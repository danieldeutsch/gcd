from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('parsing_predictor')
class ParsingPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict['tokens']
        instance = self._dataset_reader.text_to_instance(tokens=tokens)
        return instance
