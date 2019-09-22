import unittest
from allennlp.common.util import END_SYMBOL, START_SYMBOL

from gcd.inference.automata import FSA, PDA, util
from gcd.inference.constraints.parsing.common import \
    OPEN_PAREN_SYMBOL, CLOSE_PAREN_SYMBOL, \
    EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL


class TestUtil(unittest.TestCase):
    def test_intersect_fsa_fsa(self):
        # Build an FSA that accepts "123" and "124" and another that
        # only accepts "123"
        a1, a2, a3, a4, a5 = 1, 2, 3, 4, 5

        fsa1 = FSA()
        s0 = fsa1.add_state()
        s1 = fsa1.add_state()
        s2 = fsa1.add_state()
        s3 = fsa1.add_state()
        fsa1.add_symbol('1', a1)
        fsa1.add_symbol('2', a2)
        fsa1.add_symbol('3', a3)
        fsa1.add_symbol('4', a4)
        fsa1.add_symbol('5', a5)
        fsa1.set_start(s0)
        fsa1.set_final(s3)
        fsa1.add_arc(s0, s1, a1)
        fsa1.add_arc(s1, s2, a2)
        fsa1.add_arc(s2, s3, a3)
        fsa1.add_arc(s2, s3, a4)
        fsa1.compile()

        fsa2 = FSA()
        s0 = fsa2.add_state()
        s1 = fsa2.add_state()
        s2 = fsa2.add_state()
        s3 = fsa2.add_state()
        fsa2.add_symbol('1', a1)
        fsa2.add_symbol('2', a2)
        fsa2.add_symbol('3', a3)
        fsa2.add_symbol('4', a4)
        fsa2.add_symbol('5', a5)
        fsa2.set_start(s0)
        fsa2.set_final(s3)
        fsa2.add_arc(s0, s1, a1)
        fsa2.add_arc(s1, s2, a2)
        fsa2.add_arc(s2, s3, a3)
        fsa2.compile()

        intersection = util.intersect(fsa1, fsa2)
        # Check the language
        language = list(intersection.get_language())
        assert language == [' '.join(list(map(str, [a1, a2, a3])))]

    def test_intersect_fsa_pda(self):
        # Create a PDA that accepts "a^n b^n" and an FSA that accepts
        # inputs >= length 5
        pda = PDA()
        s0 = pda.add_state()
        s1 = pda.add_state()
        s2 = pda.add_state()
        s3 = pda.add_state()
        s4 = pda.add_state()
        s5 = pda.add_state()
        s6 = pda.add_state()
        s7 = pda.add_state()
        s8 = pda.add_state()
        start, end = 1, 2
        push, pop = 3, 4
        a, b = 5, 6
        bos, eos = 7, 8

        pda.add_symbol(EMPTY_STACK_OPEN_SYMBOL, start)
        pda.add_symbol(EMPTY_STACK_CLOSE_SYMBOL, end)
        pda.add_symbol(OPEN_PAREN_SYMBOL, push)
        pda.add_symbol(CLOSE_PAREN_SYMBOL, pop)
        pda.add_symbol('a', a)
        pda.add_symbol('b', b)
        pda.add_symbol(START_SYMBOL, bos)
        pda.add_symbol(END_SYMBOL, eos)

        pda.set_start(s0)
        pda.set_final(s8)

        # First "a"
        pda.add_arc(s0, s1, start)
        pda.add_arc(s1, s2, a)
        pda.add_arc(s2, s3, push)
        # Loop "a"
        pda.add_arc(s3, s4, a)
        pda.add_arc(s4, s3, push)
        # First "b"
        pda.add_arc(s3, s5, pop)
        pda.add_arc(s5, s6, b)
        # Loop "b"
        pda.add_arc(s6, s7, pop)
        pda.add_arc(s7, s6, b)
        # End
        pda.add_arc(s6, s8, end)

        pda.add_paren(start, end)
        pda.add_paren(push, pop)
        pda.compile()

        assert pda.accepts(pda.convert_to_automaton([a, b]))
        assert pda.accepts(pda.convert_to_automaton([a, a, b, b]))
        assert pda.accepts(pda.convert_to_automaton([a, a, a, b, b, b]))
        assert pda.accepts(pda.convert_to_automaton([a, a, a, a, b, b, b, b]))

        fsa = FSA()
        s0 = fsa.add_state()
        s1 = fsa.add_state()
        s2 = fsa.add_state()
        s3 = fsa.add_state()
        s4 = fsa.add_state()
        s5 = fsa.add_state()
        fsa.set_start(s0)
        fsa.set_final(s5)

        # Use the PDA's symbol table
        fsa.add_symbol(EMPTY_STACK_OPEN_SYMBOL, start)
        fsa.add_symbol(EMPTY_STACK_CLOSE_SYMBOL, end)
        fsa.add_symbol(OPEN_PAREN_SYMBOL, push)
        fsa.add_symbol(CLOSE_PAREN_SYMBOL, pop)
        fsa.add_symbol('a', a)
        fsa.add_symbol('b', b)
        fsa.add_symbol(START_SYMBOL, bos)
        fsa.add_symbol(END_SYMBOL, eos)

        fsa.add_arc(s0, s1, a)
        fsa.add_arc(s0, s1, b)
        fsa.add_arc(s1, s2, a)
        fsa.add_arc(s1, s2, b)
        fsa.add_arc(s2, s3, a)
        fsa.add_arc(s2, s3, b)
        fsa.add_arc(s3, s4, a)
        fsa.add_arc(s3, s4, b)
        fsa.add_arc(s4, s5, a)
        fsa.add_arc(s4, s5, b)
        fsa.add_arc(s5, s5, a)
        fsa.add_arc(s5, s5, b)
        fsa.compile()

        intersection = util.intersect(pda, fsa)
        assert not intersection.accepts(fsa.convert_to_automaton([a, b]))
        assert not intersection.accepts(fsa.convert_to_automaton([a, a, b, b]))
        assert intersection.accepts(fsa.convert_to_automaton([a, a, a, b, b, b]))
        assert intersection.accepts(fsa.convert_to_automaton([a, a, a, a, b, b, b, b]))
