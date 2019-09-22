import pynini
from overrides import overrides
from typing import Iterable, List, Tuple, Union

from gcd.inference.automata import Automaton


class FSA(Automaton):
    def __init__(self):
        super().__init__()

    @overrides
    def accepts(self, input_sequence: Union[Automaton, List[int]]) -> bool:
        if isinstance(input_sequence, list):
            automaton = self.convert_to_automaton(input_sequence)
        else:
            automaton = input_sequence
        return pynini.compose(self.fst, automaton).num_states() != 0

    @overrides
    def step(self, state: int, stack: int, key: int) -> Tuple[int, int]:
        assert self._compiled
        for arc in self.fst.arcs(state):
            if arc.ilabel == key:
                return arc.nextstate, stack
        raise Exception(f'State {state} does not have action {key}')

    @overrides
    def get_valid_actions(self, state: int, stack: int) -> List[int]:
        assert self._compiled
        return [arc.ilabel for arc in self.fst.arcs(state)]

    def get_language(self) -> Iterable[str]:
        assert self._compiled
        path_iterator = pynini.StringPathIterator(self.fst, self.symbol_table, self.symbol_table)
        while not path_iterator.done():
            yield path_iterator.istring()
            path_iterator.next()
