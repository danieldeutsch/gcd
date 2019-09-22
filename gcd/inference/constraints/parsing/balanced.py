import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from gcd.inference.automata import PDA
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util
from gcd.inference.constraints.parsing.util import \
    CLOSE_PAREN_SYMBOL, OPEN_PAREN_SYMBOL, \
    EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL


@Constraint.register('balanced-parens')
class BalancedParenthesesConstraint(Constraint):
    def __init__(self, max_length: int) -> None:
        from gcd.inference.constraints.parsing import MaxLengthConstraint
        self.max_length_constraint = MaxLengthConstraint(max_length)

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> PDA:
        pda = PDA()
        symbol_table = util.set_symbol_table(pda, token_to_key)
        start_key = symbol_table[START_SYMBOL]
        end_key = symbol_table[END_SYMBOL]
        open_key = symbol_table[OPEN_PAREN_SYMBOL]
        close_key = symbol_table[CLOSE_PAREN_SYMBOL]
        empty_open_key = symbol_table[EMPTY_STACK_OPEN_SYMBOL]
        empty_close_key = symbol_table[EMPTY_STACK_CLOSE_SYMBOL]

        s0 = pda.add_state()
        s1 = pda.add_state()
        s2 = pda.add_state()
        s3 = pda.add_state()
        s4 = pda.add_state()
        s5 = pda.add_state()
        s6 = pda.add_state()
        s7 = pda.add_state()
        s8 = pda.add_state()

        # Set the start and final states
        pda.set_start(s0)
        pda.set_final(s8)

        # Add the start transition and empty stack push
        pda.add_arc(s0, s1, start_key)
        pda.add_arc(s1, s2, empty_open_key)

        # Get the first opening phrase
        for token, key in symbol_table.items():
            if util.is_token_open_paren(token):
                pda.add_arc(s2, s3, key)

        # Push open paren as many times as needed
        for token, key in symbol_table.items():
            if util.is_token_open_paren(token):
                pda.add_arc(s3, s4, key)
        pda.add_arc(s4, s3, open_key)

        # Pop as many times as necessary
        for token, key in symbol_table.items():
            if util.is_token_close_paren(token):
                pda.add_arc(s5, s3, key)
        pda.add_arc(s3, s5, close_key)

        # Emit any preterminal any number of times
        for token, key in symbol_table.items():
            if util.is_token_preterminal(token):
                pda.add_arc(s3, s3, key)

        # Get the last closing phrase
        pda.add_arc(s3, s6, empty_close_key)
        for token, key in symbol_table.items():
            if util.is_token_close_paren(token):
                pda.add_arc(s6, s7, key)

        # Add the empty stack pop and end token
        pda.add_arc(s7, s8, end_key)

        # Finalize
        pda.compile()

        # For now, we don't allow PDAs with unbounded stacks, so we have
        # to intersect this constraint with a maximum length constraint. This does
        # not change the expressibility of the model
        max_length_constraint = self.max_length_constraint.build(input_tokens, token_to_key)
        pda = pda.intersect(max_length_constraint)
        return pda

    def get_name(self) -> str:
        return 'balanced-parens'
