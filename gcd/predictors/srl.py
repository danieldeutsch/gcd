import json

from allennlp.common.util import JsonDict
from allennlp.data import Instance, Token
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('srl-tagger')
class SRLPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        # tokens = json_dict['tokens']
        # json_dict = json.loads(line)
        tokens = [Token(token) for token in json_dict['words']]
        pos_tags = json_dict['pos_tags']
        chunk_tags = json_dict['chunk_tags']
        ner_tags = json_dict['ner_tags']
        target_verb_lemma = json_dict["target_verb_lemma"]
        target_verb_position = json_dict["target_verb_position"]
        verb_annotation = json_dict["verb_annotation"]
        instance = self._dataset_reader.text_to_instance(tokens=tokens,
                                                         pos_tags=pos_tags,
                                                         chunk_tags=chunk_tags,
                                                         ner_tags=ner_tags,
                                                         target_verb_lemma=target_verb_lemma,
                                                         target_verb_position=target_verb_position,
                                                         verb_annotation=verb_annotation)
        return instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        del outputs["tokens"]
        del outputs["mask"]
        del outputs["logits"]
        del outputs["class_probabilities"]
        # del outputs["loss"]
        # del outputs["mask"]
        return json.dumps(outputs) + "\n"
