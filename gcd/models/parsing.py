import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import MatrixAttention, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax
from overrides import overrides
from typing import Dict, Optional, Tuple

from gcd.inference.beam_search import ConstrainedBeamSearch
from gcd.inference.constraints import ConstraintSet
from gcd.models import util


@Model.register('parsing')
class ParsingModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TokenEmbedder,
                 hidden_size: int,
                 num_layers: int,
                 attention: MatrixAttention,
                 constraint_set: ConstraintSet,
                 beam_search: ConstrainedBeamSearch,
                 dropout: float = 0.0,
                 lstm_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.constraint_set = constraint_set
        self._token_embedder = token_embedder
        self._nonterminal_embedder = nonterminal_embedder
        assert hidden_size % 2 == 0, hidden_size
        self.bidirectional = True
        self._encoder = torch.nn.LSTM(token_embedder.get_output_dim(),
                                      hidden_size // 2,
                                      batch_first=True,
                                      bidirectional=self.bidirectional,
                                      num_layers=num_layers,
                                      dropout=lstm_dropout)
        self._attention = attention
        self._decoder = torch.nn.LSTM(nonterminal_embedder.get_output_dim(),
                                      hidden_size,
                                      batch_first=True,
                                      num_layers=num_layers,
                                      dropout=lstm_dropout)
        self._attention_layer = torch.nn.Linear(hidden_size * 2, hidden_size)
        self._output_layer = torch.nn.Linear(hidden_size,
                                             vocab.get_vocab_size('nonterminals'), bias=True)
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.beam_search = beam_search

        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

    def _decoder_step(self,
                      nonterminals: torch.Tensor,
                      state: Dict[str, torch.Tensor]):
        is_inference = nonterminals.dim() == 1
        if is_inference:
            # shape: (group_size, 1)
            nonterminals = nonterminals.unsqueeze(-1)

        tokens_encoding = state['tokens_encoding']
        tokens_mask = state['tokens_mask']
        hidden = state['hidden'].unsqueeze(0)
        memory = state['memory'].unsqueeze(0)

        # There's no need to handle decoder padding because the loss
        # will ignore pad targets
        nonterminal_embedding = self._nonterminal_embedder(nonterminals)
        nonterminal_encoding, hidden = self._decoder(nonterminal_embedding, (hidden, memory))
        hidden = (self.dropout(hidden[0]), self.dropout(hidden[1]))

        affinities = self._attention(nonterminal_encoding, tokens_encoding)
        attention = masked_softmax(affinities, tokens_mask)

        context = attention.bmm(tokens_encoding)
        concat = torch.cat([nonterminal_encoding, context], dim=2)
        preoutput = self.dropout(torch.tanh(self._attention_layer(concat)))
        scores = self._output_layer(preoutput)

        if is_inference:
            scores = torch.log_softmax(scores, dim=2).squeeze(1)

        output_state = dict(state)
        output_state['hidden'] = hidden[0].squeeze(0)
        output_state['memory'] = hidden[1].squeeze(0)
        return scores, output_state

    def _compute_loss(self,
                      initial_decoding_state: Dict[str, torch.Tensor],
                      parses: Dict[str, torch.Tensor]) -> torch.Tensor:
        nonterminals = parses['tokens']
        scores, _ = self._decoder_step(nonterminals, initial_decoding_state)
        scores = scores[:, :-1, :].contiguous()
        scores = scores.view(-1, scores.size(2))
        targets = nonterminals[:, 1:].contiguous()
        targets = targets.view(-1)
        return self._loss(scores, targets)

    def _run_inference(self,
                       tokens: torch.Tensor,
                       initial_decoding_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Pull out a tensor to get the device and batch_size
        hidden = initial_decoding_state['hidden']
        batch_size = hidden.size(0)

        start_index = self.vocab.get_token_index(START_SYMBOL, 'nonterminals')
        initial_predictions = hidden.new_empty(batch_size, dtype=torch.long)
        initial_predictions.fill_(start_index)

        # Setup the constraints for each instance. For now, we can only handle
        # batch size 1 during inference. Cloning the constraint set is not
        # implemented.
        assert batch_size == 1
        constraint_sets = [self.constraint_set]
        constraint_sets[0].setup(tokens)

        # shape: (batch_size, beam_size, max_output_length)
        # shape: (batch_size, beam_size)
        predictions, _, working_sets, violated_constraints = \
            self.beam_search.search(initial_predictions, initial_decoding_state, self._decoder_step, constraint_sets)
        return predictions, working_sets, violated_constraints

    def _remap_hidden(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # The hidden state from the RNN is
        # (num_layers * num_directions, batch_size, hidden_size).
        # We need to take both directions from the last layer and concatenate
        # them for input to the decoder. The output size should be
        # (1, batch_size, hidden_dim * 2). This is only implemented for an
        # LSTM which has a hidden state and a cell.
        hidden, cell = hidden

        num_directions = 2 if self.bidirectional else 1
        _, batch_size, hidden_size = hidden.size()

        # Separate the num_layers and num_directions
        hidden = hidden.view(-1, num_directions, batch_size, hidden_size)
        cell = cell.view(-1, num_directions, batch_size, hidden_size)

        # Concatenate the forward and backward RNNs
        if self.bidirectional:
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        else:
            hidden = hidden.squeeze(1)
            cell = hidden.squeeze(1)
            # This is not tested
            raise NotImplementedError
        return hidden, cell

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                parses: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        tokens_mask = get_text_field_mask(tokens)
        tokens_embedding = self._token_embedder(tokens)
        tokens_embedding = self.dropout(tokens_embedding)

        # Pass the input through the encoder, taking care of padding
        packed = util.pack_sequence(tokens_embedding, tokens_mask)
        packed_scores, packed_hidden = self._encoder(packed)
        tokens_encoding, hidden = util.unpack_sequence(packed_scores, packed_hidden, tokens_mask)
        hidden = self._remap_hidden(hidden)
        hidden = (self.dropout(hidden[0]), self.dropout(hidden[1]))

        initial_decoding_state = {
            'tokens_encoding': tokens_encoding,
            'tokens_mask': tokens_mask,
            'hidden': hidden[0].squeeze(0),
            'memory': hidden[1].squeeze(0)
        }

        output_dict = {}
        if parses is not None:
            output_dict['loss'] = self._compute_loss(initial_decoding_state, parses)
        else:
            predictions, working_sets, violated_constraints = self._run_inference(tokens['tokens'], initial_decoding_state)
            output_dict['prediction'] = predictions
            output_dict['working_set'] = working_sets
            output_dict['violated_constraints'] = violated_constraints
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        start_index = self.vocab.get_token_index(START_SYMBOL, 'nonterminals')
        end_index = self.vocab.get_token_index(END_SYMBOL, 'nonterminals')

        predictions = []
        for prediction in output_dict['prediction']:
            strings = []
            for y in prediction:
                if y == start_index:
                    continue
                if y == end_index:
                    break
                strings.append(self.vocab.get_token_from_index(y, 'nonterminals'))
            predictions.append(' '.join(strings))
        output_dict['prediction'] = predictions
        return output_dict
