import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from gcd.inference.automata import FSA
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('max-length')
class MaxLengthConstraint(Constraint):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> FSA:
        fsa = FSA()
        symbol_table = util.set_symbol_table(fsa, token_to_key)
        start_key = symbol_table[START_SYMBOL]
        end_key = symbol_table[END_SYMBOL]

        states = [fsa.add_state() for _ in range(self.max_length + 3)]
        fsa.set_start(states[0])
        fsa.set_final(states[-1])

        # Add starting and ending transitions
        fsa.add_arc(states[0], states[1], start_key)
        fsa.add_arc(states[-2], states[-1], end_key)

        # Add all of the intermediate transitions
        for state1, state2 in zip(states[1:-2], states[2:-1]):
            for token, key in symbol_table.items():
                if key in [start_key, end_key]:
                    continue
                if not util.is_stack_token(token):
                    fsa.add_arc(state1, state2, key)

        # Add a transition from the intermediate states to the end
        for state in states[1:-2]:
            fsa.add_arc(state, states[-1], end_key)

        # Finalize
        fsa.compile()
        return fsa

    def get_name(self) -> str:
        return f'max-length-{self.max_length}'
