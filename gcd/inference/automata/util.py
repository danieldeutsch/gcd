import pynini

from gcd.inference.automata import Automaton, FSA, PDA


def _intersect_fsa_fsa(fsa1: FSA, fsa2: FSA) -> FSA:
    intersection = FSA()
    intersection.fst = pynini.intersect(fsa1.fst, fsa2.fst)
    intersection.symbol_table = fsa1.symbol_table
    intersection.token_to_key = fsa1.token_to_key
    intersection.key_to_token = fsa1.key_to_token
    intersection.compile()
    return intersection


def _union_fsa_fsa(fsa1: FSA, fsa2: FSA) -> FSA:
    union = FSA()
    # the FSA's should have the same "vocab"
    assert fsa1.token_to_key == fsa2.token_to_key
    assert fsa1.key_to_token == fsa2.key_to_token
    # assert fsa1.symbol_table == fsa2.symbol_table
    union.fst = pynini.union(fsa1.fst, fsa2.fst)
    union.symbol_table = fsa1.symbol_table
    union.token_to_key = fsa1.token_to_key
    union.key_to_token = fsa1.key_to_token
    union.compile()
    return union


def _intersect_fsa_pda(fsa: FSA, pda: PDA) -> PDA:
    intersection = PDA()
    intersection.fst = pynini.pdt_compose(pda.fst, fsa.fst, pda.parens)
    intersection.symbol_table = fsa.symbol_table
    intersection.parens = pda.parens
    intersection.open_key_to_close_key = pda.open_key_to_close_key
    intersection.close_key_to_open_key = pda.close_key_to_open_key
    intersection.token_to_key = pda.token_to_key
    intersection.key_to_token = pda.key_to_token
    intersection.compile()
    return intersection


def _intersect_pda_pda(pda1: PDA, pda2: PDA) -> PDA:
    intersection = PDA()
    intersection.fst = pynini.intersect(pda1.fst, pda2.fst)
    intersection.symbol_table = pda1.symbol_table
    intersection.parens = pda1.parens
    intersection.open_key_to_close_key = pda1.open_key_to_close_key
    intersection.close_key_to_open_key = pda1.close_key_to_open_key
    intersection.token_to_key = pda1.token_to_key
    intersection.key_to_token = pda1.key_to_token
    intersection.compile()
    return intersection


def intersect(automaton1: Automaton, automaton2: Automaton):
    if isinstance(automaton1, FSA):
        if isinstance(automaton2, FSA):
            return _intersect_fsa_fsa(automaton1, automaton2)
        elif isinstance(automaton2, PDA):
            return _intersect_fsa_pda(automaton1, automaton2)
    elif isinstance(automaton1, PDA):
        if isinstance(automaton2, FSA):
            return _intersect_fsa_pda(automaton2, automaton1)
        elif isinstance(automaton2, PDA):
            return _intersect_pda_pda(automaton1, automaton2)
    raise Exception(f'Trying to intersect two unknown automaton types: {type(automaton1)}, {type(automaton2)}')


def union(automaton1: Automaton, automaton2: Automaton):
    if isinstance(automaton1, FSA) and isinstance(automaton2, FSA):
        return _union_fsa_fsa(automaton1, automaton2)
    else:
        raise NotImplementedError
