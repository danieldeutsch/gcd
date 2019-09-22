from typing import Dict, List, Optional, Any

from allennlp.models import SemanticRoleLabeler
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from gcd.data.dataset_readers.srl import core_arg_kinds
from gcd.inference.constraints import ConstraintSet
from gcd.inference.constraints.srl import NoDuplicatesConstraint, DisallowArgConstraint, \
    ArgumentCandidatesConstraint
import logging

from gcd.inference.beam_search import ConstrainedBeamSearch
from gcd.inference.constraints.srl.tree_constraints import get_candidate_spans

logger = logging.getLogger(__name__)


@Model.register("constrained_srl")
class Constrained_SRL_Model(SemanticRoleLabeler):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 beam_search: ConstrainedBeamSearch,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 use_no_duplicates_constraint: bool = False,
                 use_disallow_arg: bool = False,
                 use_argument_candidates: bool = False,
                 ignore_span_metric: bool = False) -> None:

        super().__init__(vocab, text_field_embedder, encoder, binary_feature_dim, embedding_dropout, initializer,
                         regularizer, label_smoothing, ignore_span_metric)

        self.constraints = []

        self.use_no_duplicates_constraint = use_no_duplicates_constraint
        self.use_disallow_arg = use_disallow_arg
        self.use_argument_candidates = use_argument_candidates

        if self.use_no_duplicates_constraint:
            for arg_kind in ["A0", "A1", "A2", "A3", "A4", "A5"]:
                self.constraints.append(NoDuplicatesConstraint(arg_kind=arg_kind))

        if self.use_disallow_arg:
            self.constraints.append(DisallowArgConstraint())

        if self.use_argument_candidates:
            self.constraints.append(ArgumentCandidatesConstraint())

        self.constrained_decoding = any([self.use_no_duplicates_constraint,
                                         self.use_disallow_arg,
                                         self.use_argument_candidates
                                         ])
        if not self.constrained_decoding:
            logger.info("Using unconstrained decoding")
        for constraint in self.constraints:
            logger.info("Using constraint %s", constraint.get_name())

        self.constraint_set = ConstraintSet(self.constraints, vocab=vocab, namespace="labels")

        self.beam_search = beam_search
        self.start_index = self.vocab.get_token_index(START_SYMBOL, namespace='labels')
        self.end_index = self.vocab.get_token_index(END_SYMBOL, namespace='labels')

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence and the verb to compute the
            frame for, under 'words' and 'verb' keys, respectively.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()

        encoded_text = self.encoder(embedded_text_with_verb_indicator, mask)
        logits = self.tag_projection_layer(encoded_text)

        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])
        output_dict = {"tokens": tokens, "logits": logits, "class_probabilities": class_probabilities}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      tags,
                                                      mask,
                                                      label_smoothing=self._label_smoothing)
            if not self.ignore_span_metric:
                if not self.training:
                    self.span_metric(class_probabilities, tags, mask)
            output_dict["loss"] = loss

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        # decode during training as well, for now.
        # self.decode(output_dict)

        words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
        target_verb_positions = [x["target_verb_position"] for x in metadata]
        legal_args_list = [x["legal_args"] for x in metadata]
        parse_list = [x["parse"] for x in metadata]
        if metadata is not None:
            output_dict["words"] = list(words)
            output_dict["verb"] = list(verbs)
            output_dict["target_verb_position"] = target_verb_positions
            output_dict["legal_args"] = legal_args_list
            output_dict["parse"] = parse_list
        return output_dict

    def _decoder_step(self,
                      inputs: torch.Tensor,
                      state: Dict[str, torch.Tensor]):
        logits = state['logits']
        curr_index = state['curr_index']

        # The index is all the same
        index = curr_index[0, 0]
        scores = F.log_softmax(logits[:, index, :], dim=1)

        output_state = dict(state)
        output_state['curr_index'] = curr_index + 1
        return scores, output_state

    # This is non-standard allennlp model, this logic should be inside forward.
    # But this is so with the allennlp's SemanticRoleLabeler model. I am just following them ...
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        # Always use batch size 1 during predict, and modify below accordingly.
        logits = output_dict['logits']
        batch_size = logits.size(0)
        assert batch_size == 1

        input_tokens = output_dict["tokens"]["tokens"][0]
        target_verb_position = output_dict["target_verb_position"][0]
        legal_args = output_dict["legal_args"][0]
        parse = output_dict["parse"][0]

        # LEGAL ARGS CAN GENUINELY BE NONE (see have.02)
        # assert legal_args is not None
        candidate_spans = get_candidate_spans(treestring=parse, verb_idx=target_verb_position)
        verb_start, verb_end = target_verb_position, target_verb_position

        self.constraint_set.setup(input_tokens,
                                  candidate_spans=candidate_spans,
                                  legal_args=legal_args,
                                  arg_kinds=core_arg_kinds,
                                  verb_start=verb_start,
                                  verb_end=verb_end)

        initial_decoding_state = {
            'logits': logits,
            'curr_index': logits.new_zeros(batch_size, 1, dtype=torch.long)
        }

        # This is a bit of a hack. We only want the beam search code to run
        # for an exact number of steps, so we change the max steps value every time
        self.beam_search.max_steps = len(input_tokens)

        constraint_sets = [self.constraint_set]
        initial_predictions = logits.new_zeros(batch_size, dtype=torch.long)
        initial_predictions.fill_(self.start_index)
        predictions, _, working_sets, violated_constraints = \
            self.beam_search.search(initial_predictions, initial_decoding_state, self._decoder_step, constraint_sets)

        # The predictions include the start and end token, which we do not want
        predictions = [prediction[1:-1] for prediction in predictions]
        # Convert to strings
        predictions = [[self.vocab.get_token_from_index(x, namespace='labels') for x in prediction] for prediction in predictions]

        output_dict["working_set"] = working_sets
        output_dict["violated_constraints"] = violated_constraints
        output_dict['tags'] = predictions
        return output_dict
