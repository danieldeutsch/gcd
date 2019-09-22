import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from gcd.inference.automata import FSA
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('non-empty-phrase')
class NonEmptyPhraseConstraint(Constraint):
    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> FSA:
        fsa = FSA()
        symbol_table = util.set_symbol_table(fsa, token_to_key)
        start_key = symbol_table[START_SYMBOL]
        end_key = symbol_table[END_SYMBOL]

        # To write this automaton, we will first write an automaton
        # to match emtpy phrases and then negate it
        s0 = fsa.add_state()
        s1 = fsa.add_state()
        s2 = fsa.add_state()
        s3 = fsa.add_state()

        # Set the start and final states
        fsa.set_start(s0)
        fsa.set_final(s3)

        # Set the transitions from and to the start and final
        fsa.add_arc(s0, s1, start_key)
        fsa.add_arc(s1, s3, end_key)
        fsa.add_arc(s2, s3, end_key)

        for token, key in symbol_table.items():
            if util.is_stack_token(token):
                continue
            if key in [start_key, end_key]:
                continue

            # Loop around s1 with any token but an open paren
            if not util.is_token_open_paren(token):
                fsa.add_arc(s1, s1, key)

            # Go from s1 to s2 if there's an open paren
            if util.is_token_open_paren(token):
                fsa.add_arc(s1, s2, key)

            # Loop on s2 if there's an open paren
            if util.is_token_open_paren(token):
                fsa.add_arc(s2, s2, key)

            # Go back from s2 to s1 on anything but a close paren or open paren
            if not util.is_token_close_paren(token) and not util.is_token_open_paren(token):
                fsa.add_arc(s2, s1, key)

        # Finalize
        fsa.compile()
        return fsa

    def get_name(self) -> str:
        return 'non-empty-phrase'
