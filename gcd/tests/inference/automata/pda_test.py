import torch
import unittest
from allennlp.common.util import END_SYMBOL, START_SYMBOL

from gcd.inference.automata import PDA
from gcd.inference.constraints.parsing import BalancedParenthesesConstraint, MaxLengthConstraint
from gcd.inference.constraints.parsing.util import \
    EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL, \
    OPEN_PAREN_SYMBOL, CLOSE_PAREN_SYMBOL


class TestPDA(unittest.TestCase):
    def setUp(self):
        # Build a PDA that accepts "<bos> a^n b^k c^n <eos>" where
        # n >= 1 and k >= 0
        self.pda = PDA()
        self.s0 = self.pda.add_state()
        self.s1 = self.pda.add_state()
        self.s2 = self.pda.add_state()
        self.s3 = self.pda.add_state()
        self.s4 = self.pda.add_state()
        self.s5 = self.pda.add_state()
        self.s6 = self.pda.add_state()
        self.s7 = self.pda.add_state()
        self.s8 = self.pda.add_state()
        self.s9 = self.pda.add_state()
        self.s10 = self.pda.add_state()
        self.s11 = self.pda.add_state()
        self.start, self.end = 1, 2
        self.push, self.pop = 3, 4
        self.a, self.b, self.c = 5, 6, 7
        self.bos, self.eos = 8, 9

        self.pda.add_symbol(EMPTY_STACK_OPEN_SYMBOL, self.start)
        self.pda.add_symbol(EMPTY_STACK_CLOSE_SYMBOL, self.end)
        self.pda.add_symbol(OPEN_PAREN_SYMBOL, self.push)
        self.pda.add_symbol(CLOSE_PAREN_SYMBOL, self.pop)
        self.pda.add_symbol('a', self.a)
        self.pda.add_symbol('b', self.b)
        self.pda.add_symbol('c', self.c)
        self.pda.add_symbol(START_SYMBOL, self.bos)
        self.pda.add_symbol(END_SYMBOL, self.eos)

        self.pda.set_start(self.s0)
        self.pda.set_final(self.s11)

        # start -> a -> push
        self.pda.add_arc(self.s0, self.s1, self.bos)
        self.pda.add_arc(self.s1, self.s2, self.start)
        self.pda.add_arc(self.s2, self.s3, self.a)
        self.pda.add_arc(self.s3, self.s4, self.push)
        # push "a"
        self.pda.add_arc(self.s4, self.s5, self.a)
        self.pda.add_arc(self.s5, self.s4, self.push)
        # see first "b"
        self.pda.add_arc(self.s4, self.s6, self.b)
        # loop "b"
        self.pda.add_arc(self.s6, self.s6, self.b)
        # skip "b"
        self.pda.add_arc(self.s4, self.s7, self.pop)
        # see first "c"
        self.pda.add_arc(self.s6, self.s7, self.pop)
        self.pda.add_arc(self.s7, self.s8, self.c)
        # pop "c"
        self.pda.add_arc(self.s8, self.s9, self.pop)
        self.pda.add_arc(self.s9, self.s8, self.c)
        # finish
        self.pda.add_arc(self.s8, self.s10, self.end)
        self.pda.add_arc(self.s10, self.s11, self.eos)

        self.pda.add_paren(self.start, self.end)
        self.pda.add_paren(self.push, self.pop)
        self.pda.compile()

    def test_accepts(self):
        bos, eos = self.bos, self.eos
        a, b, c = self.a, self.b, self.c
        assert self.pda.accepts(self.pda.convert_to_automaton([bos, a, b, c, eos]))
        assert self.pda.accepts(self.pda.convert_to_automaton([bos, a, a, b, c, c, eos]))
        assert self.pda.accepts(self.pda.convert_to_automaton([bos, a, a, a, b, b, c, c, c, eos]))
        assert self.pda.accepts(self.pda.convert_to_automaton([bos, a, c, eos]))
        assert self.pda.accepts(self.pda.convert_to_automaton([bos, a, a, c, c, eos]))
        assert self.pda.accepts(self.pda.convert_to_automaton([bos, a, a, a, c, c, c, eos]))

        assert not self.pda.accepts(self.pda.convert_to_automaton([bos, a, a, b, c, c]))
        assert not self.pda.accepts(self.pda.convert_to_automaton([a, a, b, c, c, eos]))
        assert not self.pda.accepts(self.pda.convert_to_automaton([a, a, b, b, c]))
        assert not self.pda.accepts(self.pda.convert_to_automaton([a, a, c, c, c]))

    @unittest.skip('"step" not implemented for non-DAGs')
    def test_get_valid_actions(self):
        bos, eos = self.bos, self.eos
        a, b, c = self.a, self.b, self.c
        push = self.push
        start = self.start

        assert list(sorted(self.pda.get_valid_actions(0, []))) == [bos]
        assert list(sorted(self.pda.get_valid_actions(1, []))) == [a]
        assert list(sorted(self.pda.get_valid_actions(2, [start]))) == [a]
        assert list(sorted(self.pda.get_valid_actions(4, [start]))) == [a, b, c]
        assert list(sorted(self.pda.get_valid_actions(4, [start, push]))) == [a, b, c]
        assert list(sorted(self.pda.get_valid_actions(3, [start, push]))) == [a, b, c]
        assert list(sorted(self.pda.get_valid_actions(5, [start, push]))) == [b, c]
        assert list(sorted(self.pda.get_valid_actions(8, [start]))) == [c]
        assert list(sorted(self.pda.get_valid_actions(8, [start, push]))) == [c]
        assert list(sorted(self.pda.get_valid_actions(6, [start]))) == [eos]
        assert list(sorted(self.pda.get_valid_actions(6, [start, push]))) == [c]
        assert list(sorted(self.pda.get_valid_actions(7, []))) == [eos]
        assert list(sorted(self.pda.get_valid_actions(9, []))) == []

    @unittest.skip('"step" not implemented for non-DAGs')
    def test_step(self):
        bos, eos = self.bos, self.eos
        a, b, c = self.a, self.b, self.c
        push = self.push
        start = self.start

        assert self.pda.step(0, [], bos) == (1, [])
        assert self.pda.step(1, [], a) == (4, [start])
        assert self.pda.step(2, [start], a) == (4, [start])
        assert self.pda.step(4, [start], a) == (4, [start, push])
        assert self.pda.step(4, [start, push], a) == (4, [start, push, push])
        assert self.pda.step(4, [start], b) == (5, [start, push])
        assert self.pda.step(4, [start], c) == (6, [start])
        assert self.pda.step(4, [start, push], c) == (6, [start, push])
        assert self.pda.step(3, [start, push], a) == (4, [start, push])
        assert self.pda.step(3, [start, push], b) == (5, [start, push])
        assert self.pda.step(3, [start, push], c) == (6, [start])
        assert self.pda.step(3, [start, push, push], c) == (6, [start, push])
        assert self.pda.step(5, [start, push], b) == (5, [start, push])
        assert self.pda.step(5, [start, push], c) == (6, [start])
        assert self.pda.step(5, [start, push, push], c) == (6, [start, push])
        assert self.pda.step(8, [start], c) == (6, [start])
        assert self.pda.step(8, [start, push], c) == (6, [start, push])
        assert self.pda.step(6, [start], eos) == (9, [])
        assert self.pda.step(6, [start, push], c) == (6, [start])
        assert self.pda.step(7, [], eos) == (9, [])

    def test_intersection(self):
        input_tokens = torch.LongTensor(1, 5)
        start, end = 1, 2
        nt, xx, close = 3, 4, 5
        token_to_key = {
            START_SYMBOL: start,
            END_SYMBOL: end,
            '(NT': nt,
            'XX': xx,
            ')': close
        }

        balanced = BalancedParenthesesConstraint(3).build(input_tokens, token_to_key)
        max_length = MaxLengthConstraint(3).build(input_tokens, token_to_key)
        intersection = balanced.intersect(max_length)

        empty_open = intersection.token_to_key[EMPTY_STACK_OPEN_SYMBOL]
        empty_close = intersection.token_to_key[EMPTY_STACK_CLOSE_SYMBOL]
        phrase_open = intersection.token_to_key[OPEN_PAREN_SYMBOL]
        phrase_close = intersection.token_to_key[CLOSE_PAREN_SYMBOL]

        assert intersection.valid_state_stacks[9] == set([0])
        assert intersection.valid_state_stacks[8] == set([0])
        assert intersection.valid_state_stacks[7] == set([0])
        assert intersection.valid_state_stacks[5] == set([1])
        assert intersection.valid_state_stacks[4] == set()
        assert intersection.valid_state_stacks[6] == set([1])
        assert intersection.valid_state_stacks[3] == set([1, 2])
        assert intersection.valid_state_stacks[2] == set([1, 2])
        assert intersection.valid_state_stacks[1] == set([0])
        assert intersection.valid_state_stacks[0] == set([0])

        assert intersection.valid_arc_stacks[8][end] == set([0])
        assert intersection.valid_arc_stacks[7][close] == set([0])
        assert intersection.valid_arc_stacks[5][empty_close] == set([1])
        assert intersection.valid_arc_stacks[4][phrase_open] == set()
        assert intersection.valid_arc_stacks[6][close] == set([1])
        assert intersection.valid_arc_stacks[3][nt] == set()
        assert intersection.valid_arc_stacks[3][xx] == set([1])
        assert intersection.valid_arc_stacks[3][phrase_close] == set([2])
        assert intersection.valid_arc_stacks[3][empty_close] == set([1])
        assert intersection.valid_arc_stacks[2][nt] == set([1, 2])
        assert intersection.valid_arc_stacks[1][empty_open] == set([0])
        assert intersection.valid_arc_stacks[0][start] == set([0])

        assert intersection.step(0, 0, start) == (1, 0)
        assert intersection.step(1, 0, nt) == (3, 1)
        assert intersection.step(2, 1, nt) == (3, 1)
        with self.assertRaises(Exception):
            assert intersection.step(3, 1, nt) == (4, 1)
        assert intersection.step(3, 1, xx) == (5, 1)
        assert intersection.step(3, 1, close) == (8, 0)
        assert intersection.step(5, 1, close) == (8, 0)
        assert intersection.step(7, 0, close) == (8, 0)
        assert intersection.step(8, 0, end) == (9, 0)

        assert list(sorted(intersection.get_valid_actions(0, 0))) == list(sorted([start]))
        assert list(sorted(intersection.get_valid_actions(1, 0))) == list(sorted([nt]))
        assert list(sorted(intersection.get_valid_actions(2, 1))) == list(sorted([nt]))
        assert list(sorted(intersection.get_valid_actions(3, 1))) == list(sorted([xx, close]))
        assert list(sorted(intersection.get_valid_actions(5, 1))) == list(sorted([close]))
        assert list(sorted(intersection.get_valid_actions(7, 0))) == list(sorted([close]))
        assert list(sorted(intersection.get_valid_actions(8, 0))) == list(sorted([end]))
