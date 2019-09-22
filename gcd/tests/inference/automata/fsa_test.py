import unittest

from gcd.inference.automata import FSA


class TestFSA(unittest.TestCase):
    def setUp(self):
        # Build an FSA that accepts "123" and "124"
        self.fsa = FSA()
        self.s0 = self.fsa.add_state()
        self.s1 = self.fsa.add_state()
        self.s2 = self.fsa.add_state()
        self.s3 = self.fsa.add_state()
        self.a1, self.a2, self.a3, self.a4, self.a5 = 1, 2, 3, 4, 5
        self.fsa.add_symbol('1', self.a1)
        self.fsa.add_symbol('2', self.a2)
        self.fsa.add_symbol('3', self.a3)
        self.fsa.add_symbol('4', self.a4)
        self.fsa.add_symbol('5', self.a5)
        self.fsa.set_start(self.s0)
        self.fsa.set_final(self.s3)
        self.fsa.add_arc(self.s0, self.s1, self.a1)
        self.fsa.add_arc(self.s1, self.s2, self.a2)
        self.fsa.add_arc(self.s2, self.s3, self.a3)
        self.fsa.add_arc(self.s2, self.s3, self.a4)
        self.fsa.compile()

    def test_accepts(self):
        assert self.fsa.accepts(self.fsa.convert_to_automaton([self.a1, self.a2, self.a3]))
        assert self.fsa.accepts(self.fsa.convert_to_automaton([self.a1, self.a2, self.a4]))
        assert not self.fsa.accepts(self.fsa.convert_to_automaton([self.a1, self.a2]))
        assert not self.fsa.accepts(self.fsa.convert_to_automaton([self.a1, self.a2, self.a5]))
        assert not self.fsa.accepts(self.fsa.convert_to_automaton([]))

    def test_get_valid_actions(self):
        assert self.fsa.get_valid_actions(self.s0, []) == [self.a1]
        assert self.fsa.get_valid_actions(self.s1, []) == [self.a2]
        assert self.fsa.get_valid_actions(self.s2, []) == [self.a3, self.a4]
        assert self.fsa.get_valid_actions(self.s3, []) == []

    def test_step(self):
        assert self.fsa.step(self.s0, [], self.a1) == (self.s1, [])
        assert self.fsa.step(self.s1, [], self.a2) == (self.s2, [])
        assert self.fsa.step(self.s2, [], self.a3) == (self.s3, [])
        assert self.fsa.step(self.s2, [], self.a4) == (self.s3, [])

        with self.assertRaises(Exception):
            assert self.fsa.step(self.s0, self.a2)

    def test_language(self):
        language = list(self.fsa.get_language())
        assert len(language) == 2
        assert ' '.join(list(map(str, [self.a1, self.a2, self.a3]))) in language
        assert ' '.join(list(map(str, [self.a1, self.a2, self.a4]))) in language
