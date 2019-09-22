import json
from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from gcd.data.dataset_readers.propbank_utils import PropbankFramesReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False


core_arg_kinds = ["A0", "A1", "A2", "A3", "A4", "A5"]


def get_bio_from_spans(verb_annotation: List[str], year="2004", core_args_only=False):
    # core_args_only -- only keep core arguments A0 to A5, rest A* are made O
    labels = []
    # which kind of tag are we inside of?
    inside = None
    for token_ann in verb_annotation:
        if token_ann.startswith("(") and token_ann.endswith(")"):
            kind = token_ann[1:-1].split("*")[0]
            if (core_args_only and kind in core_arg_kinds) or not core_args_only:
                tag = "B"
                label = tag + "-" + kind
            else:
                label = "O"
            labels.append(label)
            continue
        elif token_ann.startswith("("):
            kind = token_ann[1:].split("*")[0]
            if (core_args_only and kind in core_arg_kinds) or not core_args_only:
                tag = "B"
                label = tag + "-" + kind
                # turn on inside
                inside = kind
            else:
                label = "O"
            labels.append(label)
            continue
        elif token_ann.endswith(")"):
            # careful, this is where 2004 and 2005 differ
            if year == "2004":
                # ends (*A1 ... with *A1)
                kind = token_ann[:-1].split("*")[1]
            elif year == "2005":
                # ends (*A1 ... with *)
                kind = inside
            else:
                raise NotImplementedError
            if (core_args_only and kind in core_arg_kinds) or not core_args_only:
                tag = "I"
                label = tag + "-" + kind
                # turn off inside
                inside = None
            else:
                label = "O"
            labels.append(label)
            continue
        elif token_ann == "*":
            if inside is not None:
                tag = "I"
                kind = inside
                label = tag + "-" + kind
            else:
                label = "O"
            labels.append(label)
            continue
    return labels


@DatasetReader.register("conll05_srl")
class SRL_CONLL05Reader(DatasetReader):
    """
    For reading the CONLL 2004 format files for SRL sequence tagging.
    An example from the README is below.
    The last two columns are span marking arguments for the verbs 'face' and 'explore', in that order.

   WORDS---->  NE--->  POS   PARTIAL_SYNT   FULL_SYNT------>   VS   TARGETS  PROPS------->
   The             *   DT    (NP*   (S*        (S(NP*          -    -        (A0*    (A0*
   $               *   $        *     *     (ADJP(QP*          -    -           *       *
   1.4             *   CD       *     *             *          -    -           *       *
   billion         *   CD       *     *             *))        -    -           *       *
   robot           *   NN       *     *             *          -    -           *       *
   spacecraft      *   NN       *)    *             *)         -    -           *)      *)
   faces           *   VBZ   (VP*)    *          (VP*          01   face      (V*)      *
   a               *   DT    (NP*     *          (NP*          -    -        (A1*       *
   six-year        *   JJ       *     *             *          -    -           *       *
   journey         *   NN       *)    *             *          -    -           *       *
   to              *   TO    (VP*   (S*        (S(VP*          -    -           *       *
   explore         *   VB       *)    *          (VP*          01   explore     *     (V*)
   Jupiter     (ORG*)  NNP   (NP*)    *       (NP(NP*)         -    -           *    (A1*
   and             *   CC       *     *             *          -    -           *       *
   its             *   PRP$  (NP*     *          (NP*          -    -           *       *
   16              *   CD       *     *             *          -    -           *       *
   known           *   JJ       *     *             *          -    -           *       *
   moons           *   NNS      *)    *)            *)))))))   -    -           *)      *)
   .               *   .        *     *)            *)         -    -           *       *

   WORDS---->  POS   FULL_SYNT------>   NE--->  VS  TARGETS  PROPS------->

    """
    _VALID_LABELS = {'srl', 'pos', 'chunk'}

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "srl",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 core_args_only: bool = False,
                 coding_scheme: str = "BIO",
                 year: str = '2005',
                 propbank_root: str = '/home1/s/shyamupa/propbank-frames',
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))
        if coding_scheme not in ("BIO", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))
        # self.add_start_end_symbols_to_output = add_start_end_symbols_to_output
        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.core_args_only = core_args_only
        self.coding_scheme = coding_scheme
        self.year = year
        logger.info("Reading instances for CONLL %s", year)
        self.label_namespace = label_namespace
        self._original_coding_scheme = "BIO"  # BIO is same as IOB2
        #
        self.prop_reader = PropbankFramesReader(root=propbank_root)
        self.prop_reader.load_all()

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        # TODO Maybe this should be a flag whether input file is jsonl or conll?
        # TODO Also does this slow things down, as _read is called repeatedly?
        if file_path.endswith("jsonl"):
            with open(file_path, "r") as data_file:
                logger.info("Reading instances from json lines in file at: %s", file_path)
                for line in data_file:
                    data = json.loads(line)
                    tokens = [Token(token) for token in data['words']]
                    pos_tags = data['pos_tags']
                    chunk_tags = data['chunk_tags']
                    ner_tags = data['ner_tags']
                    target_verb_lemma = data["target_verb_lemma"]
                    target_verb_position = data["target_verb_position"]
                    target_verb_sense = data["verb_sense"]
                    legal_args = data["legal_args"]
                    verb_annotation = data["verb_annotation"]
                    parse = data["parse"]
                    yield self.text_to_instance(tokens=tokens,
                                                pos_tags=pos_tags,
                                                chunk_tags=chunk_tags,
                                                ner_tags=ner_tags,
                                                target_verb_lemma=target_verb_lemma,
                                                target_verb_position=target_verb_position,
                                                verb_sense=target_verb_sense,
                                                legal_args=legal_args,
                                                verb_annotation=verb_annotation,
                                                parse=parse)
        else:
            with open(file_path, "r") as data_file:
                logger.info("Reading instances from lines in file at: %s", file_path)

                # Group into alternative divider / sentence chunks.
                for is_divider, lines in itertools.groupby(data_file, _is_divider):
                    # Ignore the divider chunks, so that `lines` corresponds to the words
                    # of a single sentence.
                    if not is_divider:
                        fields = [line.strip().split() for line in lines]
                        # unzipping trick returns tuples, but our Fields need lists
                        fields = [list(field) for field in zip(*fields)]
                        #   WORDS---->  POS   FULL_SYNT------>   NE--->  VS  TARGETS  PROPS------->
                        if file_path.endswith("test-wsj") or file_path.endswith("test-brown"):
                            ##### This is test data #####
                            tokens_, target_verb_tags = fields[:2]
                            verb_annotations = fields[2:]
                            tokens = [Token(token) for token in tokens_]
                            # print(clauses)
                            target_verbs = [(idx, vrb) for idx, vrb in enumerate(target_verb_tags) if vrb != "-"]
                            for target_verb, verb_annotation in zip(target_verbs,
                                                                                verb_annotations):
                                target_verb_position, target_verb_lemma = target_verb
                                # verb_sense = int(verb_sense)
                                legal_args = core_arg_kinds
                                yield self.text_to_instance(tokens=tokens,
                                                            pos_tags=None,
                                                            chunk_tags=None,
                                                            ner_tags=None,
                                                            target_verb_lemma=target_verb_lemma,
                                                            target_verb_position=target_verb_position,
                                                            verb_sense=None,
                                                            legal_args=legal_args,
                                                            verb_annotation=verb_annotation)
                        else:
                            tokens_, pos_tags, clauses, ner_tags, target_verb_sense_tags, target_verb_tags = fields[:6]
                            chunk_tags = []
                            # The rest of the columns are annotations for each verb, in order of appearance
                            verb_annotations = fields[6:]
                            # TextField requires ``Token`` objects
                            tokens = [Token(token) for token in tokens_]
                            # print(clauses)
                            target_verbs = [(idx, vrb) for idx, vrb in enumerate(target_verb_tags) if vrb != "-"]
                            verb_senses = [(idx, vrb) for idx, vrb in enumerate(target_verb_sense_tags) if vrb != "-"]

                            for verb_sense, target_verb, verb_annotation in zip(verb_senses, target_verbs,
                                                                                verb_annotations):
                                target_verb_position, target_verb_lemma = target_verb
                                position, target_verb_sense = verb_sense
                                assert position == target_verb_position
                                # verb_sense = int(verb_sense)
                                if (target_verb_lemma, target_verb_sense) in self.prop_reader.lemma_id2role:
                                    legal_args = self.prop_reader.get_legal_args(verb_lemma=target_verb_lemma,
                                                                                 sense_id=target_verb_sense)
                                else:
                                    legal_args = core_arg_kinds
                                yield self.text_to_instance(tokens=tokens,
                                                            pos_tags=pos_tags,
                                                            chunk_tags=chunk_tags,
                                                            ner_tags=ner_tags,
                                                            target_verb_lemma=target_verb_lemma,
                                                            target_verb_position=target_verb_position,
                                                            verb_sense=target_verb_sense,
                                                            legal_args=legal_args,
                                                            verb_annotation=verb_annotation)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         pos_tags: List[str] = None,
                         chunk_tags: List[str] = None,
                         ner_tags: List[str] = None,
                         target_verb_lemma: str = None,
                         target_verb_position: int = None,
                         verb_sense: str = None,
                         legal_args: List[str] = None,
                         verb_annotation: List[str] = None,
                         parse: str = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        words = [x.text for x in tokens]
        instance_fields["metadata"] = MetadataField(
            {"words": words,  # used in ai2's srl model
             "pos_tags": pos_tags,
             "chunk_tags": chunk_tags,
             "ner_tags": chunk_tags,
             "target_verb_lemma": target_verb_lemma,
             "target_verb_position": target_verb_position,
             "verb_annotation": verb_annotation,
             "verb_sense": verb_sense,
             "legal_args": legal_args,
             "verb": target_verb_lemma,  # used in ai2's srl model
             "parse": parse  # for constraints for the dev set srl
             }
        )

        # This is the position of the gold verb predicate
        # We may or may not use it (the model might predict the predicate), but the reader always sends it.
        # instance_fields["verb_pos"] = IndexField(index=target_verb_position, sequence_field=sequence)

        # TODO Allennlp uses SequenceFeatureField for a indicator vector of the verb position (Find this)
        # instance_fields["verb_indicator"] = SequenceFeatureField(index=target_verb_position, sequence_field=sequence)

        verb_indicator = np.zeros(len(tokens))
        verb_indicator[target_verb_position] = 1.0
        instance_fields["verb_indicator"] = ArrayField(array=verb_indicator)

        # everyone follows the default IOB2 == BIO format here
        coded_srl = get_bio_from_spans(verb_annotation, year=self.year, core_args_only=self.core_args_only)
        coded_chunks = chunk_tags
        coded_ner = ner_tags

        if self.coding_scheme == "BIOUL":
            # coded_srl = get_bio_from_spans(verb_annotation)
            coded_chunks = to_bioul(chunk_tags,
                                    encoding=self._original_coding_scheme) if chunk_tags is not None else None
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None

        if 'pos' in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError("Dataset reader was specified to use pos_tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['pos_tags'] = SequenceLabelField(pos_tags, sequence, "pos_tags")
        if 'chunk' in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError("Dataset reader was specified to use chunk tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['chunk_tags'] = SequenceLabelField(coded_chunks, sequence, "chunk_tags")
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError("Dataset reader was specified to use NER tags as "
                                         " features. Pass them to text_to_instance.")
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, "ner_tags")

        # Add "tag label" to instance
        if self.tag_label == 'srl' and coded_srl is not None:
            instance_fields['tags'] = SequenceLabelField(coded_srl, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'pos' and pos_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pos_tags, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'chunk' and coded_chunks is not None:
            instance_fields['tags'] = SequenceLabelField(coded_chunks, sequence,
                                                         self.label_namespace)

        return Instance(instance_fields)


if __name__ == '__main__':
    reader = SRL_CONLL05Reader(core_args_only=True)
    filepath = "/mnt/castor/seas_home/s/shyamupa/deep_srl/data/srl/conll05st-release/test-wsj"
    missed = 0
    total = 0
    instances = reader.read(filepath)
    # dump_instances(instances,"tmp.jsonl")
    # for instance in reader.read(filepath):
    #     print(instance)
    #     #     metadata = instance["metadata"]
    #     #     total += 1
    #     #     lemma, sense, legal_args = metadata["target_verb_lemma"], metadata["verb_sense"], metadata["legal_args"]
    #     #     print(legal_args)
    #     #     if legal_args is None:
    #     #         print("missed", lemma, sense)
    #     #         missed += 1
    #     # print(f"{missed}/{total}={missed/total}")
    #     break
