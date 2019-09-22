import json
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides
from typing import List, Optional


@DatasetReader.register('parsing_reader')
class ParsingDatasetReader(DatasetReader):
    def __init__(self,
                 lowercase_tokens: bool = True) -> None:
        super().__init__(False)
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(),
                                        start_tokens=[START_SYMBOL],
                                        end_tokens=[END_SYMBOL])
        self._token_indexers = {'tokens': SingleIdTokenIndexer(lowercase_tokens=lowercase_tokens)}
        self._nonterminals_indexers = {'tokens': SingleIdTokenIndexer(namespace='nonterminals')}

    @overrides
    def _read(self, filename: str) -> List[Instance]:
        instances = []
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                tokens = data['tokens']
                parse = data['parse']
                instances.append(self.text_to_instance(tokens, parse))
        return instances

    @overrides
    def text_to_instance(self,
                         tokens: str,
                         parses: Optional[str] = None) -> Instance:
        raw_tokens = self._tokenizer.tokenize(tokens)
        tokens_field = TextField(raw_tokens, self._token_indexers)
        fields = {'tokens': tokens_field}
        if parses is not None:
            raw_parses = self._tokenizer.tokenize(parses)
            parses_field = TextField(raw_parses, self._nonterminals_indexers)
            fields['parses'] = parses_field
        return Instance(fields)
