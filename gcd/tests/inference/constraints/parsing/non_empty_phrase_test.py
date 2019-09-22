import unittest
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from gcd.inference.constraints.parsing import NonEmptyPhraseConstraint


class TestNonEmptyPhraseConstraint(unittest.TestCase):
    def test_non_empty_phrase_constraint(self):
        start, end = 1, 2
        nt, xx, close = 3, 4, 5
        token_to_key = {
            START_SYMBOL: start,
            END_SYMBOL: end,
            '(NT': nt,
            'XX': xx,
            ')': close
        }
        automaton = NonEmptyPhraseConstraint().build(None, token_to_key)

        assert automaton.accepts([start, nt, xx, close, end])
        assert automaton.accepts([start, nt, nt, xx, close, nt, xx, close, close, end])
        assert automaton.accepts([start, nt, nt, end])
        assert not automaton.accepts([start, nt, close, end])
        assert not automaton.accepts([start, nt, nt, close, close, end])
        assert not automaton.accepts([start, nt, xx, close, nt, close, end])
