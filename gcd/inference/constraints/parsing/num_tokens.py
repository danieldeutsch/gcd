import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from gcd.inference.automata import FSA
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('num-tokens')
class NumTokensConstraint(Constraint):
    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> FSA:
        batch_size, num_tokens = input_tokens.size()
        assert batch_size == 1, batch_size
        num_tokens -= 2  # <bos>, <eos>

        fsa = FSA()
        symbol_table = util.set_symbol_table(fsa, token_to_key)
        start_key = symbol_table[START_SYMBOL]
        end_key = symbol_table[END_SYMBOL]

        # There is a state for seeing 0, 1, ..., num_tokens tokens plus one
        # start state and one final state
        states = [fsa.add_state() for _ in range(num_tokens + 3)]

        # Set the start and final states
        fsa.set_start(states[0])
        fsa.set_final(states[-1])

        # Set the transitions from and to the start and final
        fsa.add_arc(states[0], states[1], start_key)
        fsa.add_arc(states[-2], states[-1], end_key)

        # For the middle states, set the self loop for any symbol
        # except for a preterminal, start, end, open, or close.
        for state in states[1:-1]:
            for token, key in symbol_table.items():
                if util.is_stack_token(token):
                    continue
                # Now we only have vocabulary items left
                if key in [start_key, end_key]:
                    continue
                if util.is_token_preterminal(token):
                    continue
                fsa.add_arc(state, state, key)

        # Add a transition between the intermediate states using a preterminal
        for state1, state2 in zip(states[1:-2], states[2:-1]):
            for token, key in symbol_table.items():
                if util.is_token_preterminal(token):
                    fsa.add_arc(state1, state2, key)

        # Finalize
        fsa.compile()
        return fsa

    def get_name(self) -> str:
        return 'num-tokens'
