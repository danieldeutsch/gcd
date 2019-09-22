import torch
import unittest
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from gcd.inference.constraints.parsing import NumTokensConstraint


class TestNumTokensConstraint(unittest.TestCase):
    def test_num_tokens_constraint(self):
        num_tokens = 3
        input_tokens = torch.zeros(1, num_tokens + 2)

        start, end = 1, 2
        nt, xx, close = 3, 4, 5
        token_to_key = {
            START_SYMBOL: start,
            END_SYMBOL: end,
            '(NT': nt,
            'XX': xx,
            ')': close
        }
        automaton = NumTokensConstraint().build(input_tokens, token_to_key)

        assert automaton.accepts([start, xx, xx, xx, end])
        assert automaton.accepts([start, nt, xx, close, close, xx, nt, xx, end])
        assert not automaton.accepts([start, xx, xx, end])
        assert not automaton.accepts([start, xx, nt, close, xx, end])
