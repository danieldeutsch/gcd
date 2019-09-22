import os
import pynini
import subprocess
from typing import List, Tuple


class Automaton(object):
    def __init__(self) -> None:
        self.fst = pynini.Fst()
        self.symbol_table = pynini.SymbolTable()
        self._compiled = False
        self.token_to_key = {}
        self.key_to_token = {}

    def add_symbol(self, symbol: str, key: int) -> None:
        assert key > 0, key
        assert not self._compiled
        self.symbol_table.add_symbol(symbol, key)
        self.token_to_key[symbol] = key
        self.key_to_token[key] = symbol

    def add_state(self) -> int:
        assert not self._compiled
        return self.fst.add_state()

    def add_arc(self, from_state: int, to_state: int, key: int) -> None:
        assert not self._compiled
        self.fst.add_arc(from_state, pynini.Arc(key, key, None, to_state))

    def set_start(self, state: int) -> None:
        assert not self._compiled
        self.fst.set_start(state)

    def set_final(self, state: int) -> None:
        assert not self._compiled
        self.fst.set_final(state)

    def get_start(self) -> int:
        return self.fst.start()

    def get_final(self) -> int:
        return self.fst.final()

    def compile(self, optimize: bool = True) -> None:
        assert not self._compiled
        self.fst.set_input_symbols(self.symbol_table)
        self.fst.set_output_symbols(self.symbol_table)
        self.fst = pynini.determinize(self.fst)
        if optimize:
            self.fst.optimize()
        assert self.fst.verify()
        self._compiled = True

    def accepts(self, automaton: 'Automaton') -> bool:
        raise NotImplementedError

    def convert_to_automaton(self, keys: List[int]) -> 'Automaton':
        assert self._compiled
        input_fst = pynini.Fst()
        input_fst.set_input_symbols(self.symbol_table)
        input_fst.set_output_symbols(self.symbol_table)
        states = [input_fst.add_state() for _ in range(len(keys) + 1)]
        input_fst.set_start(states[0])
        input_fst.set_final(states[-1])
        for from_state, to_state, key in zip(states, states[1:], keys):
            input_fst.add_arc(from_state, pynini.Arc(key, key, None, to_state))
        return input_fst

    def step(self, state: int, stack: int, key: int) -> Tuple[int, int]:
        raise NotImplementedError

    def get_valid_actions(self, state: int, stack: int) -> List[int]:
        raise NotImplementedError

    def intersect(self, automaton: 'Automaton') -> 'Automaton':
        assert self._compiled
        from gcd.inference.automata import util
        return util.intersect(self, automaton)

    def union(self, automaton: 'Automaton') -> 'Automaton':
        assert self._compiled
        from gcd.inference.automata import util
        return util.union(self, automaton)

    def save(self, filename: str, directory='.') -> None:
        filename_no_ext = os.path.splitext(filename)[0]
        self.fst.draw(f'{directory}/{filename_no_ext}.gv')
        cmd_args = ['dot', '-Tpdf', f'{directory}/{filename_no_ext}.gv', '-o', f'{directory}/{filename_no_ext}.pdf']
        subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def num_states(self):
        assert self._compiled
        return self.fst.num_states()

    def num_arcs(self):
        assert self._compiled
        total = 0
        for state in self.fst.states():
            total += self.fst.num_arcs(state)
        return total
