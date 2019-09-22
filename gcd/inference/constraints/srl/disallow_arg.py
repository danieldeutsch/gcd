import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict, List

from gcd.data.dataset_readers.srl import core_arg_kinds
from gcd.inference.automata import FSA
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.srl import util


# @Constraint.register('disallow-arg')
class DisallowArgConstraint(Constraint):
    """
    Disallow some arg labels for a given verb. So if the verb only takes A0 and A1 (legal_args=[A0,A1]), then using
    this constraint prevents other A* to get assigned
    """

    def __init__(self):
        """
        :param legal_args: list of legal args verb can take
        """
        pass

    def build(self, input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> FSA:

        legal_args: List[str] = kwargs["legal_args"]

        # LEGAL ARGS CAN GENUINELY BE NONE (have.02)
        # TODO Should we just take this constraint out?
        if legal_args is None or len(legal_args) == 0:
            # legal_args = []
            legal_args = core_arg_kinds
        fsa = FSA()
        symbol_table = util.set_symbol_table(fsa, token_to_key)
        start_key = symbol_table[START_SYMBOL]
        end_key = symbol_table[END_SYMBOL]

        # One state valid state + start + end
        start_state = fsa.add_state()
        valid_state = fsa.add_state()
        final_state = fsa.add_state()

        # Set the start and final states
        fsa.set_start(start_state)
        fsa.set_final(final_state)

        # Set the starting and ending transitions
        fsa.add_arc(start_state, valid_state, start_key)
        fsa.add_arc(valid_state, final_state, end_key)

        for token, key in symbol_table.items():
            if token in [START_SYMBOL, END_SYMBOL]:
                continue

            if token == "O":
                # "O" is valid for everyone
                fsa.add_arc(valid_state, valid_state, key)
            else:
                # Add the keys for all valid labels other than "O"
                label, kind = token.split("-")
                if kind in legal_args:
                    fsa.add_arc(valid_state, valid_state, key)

        # Finalize
        fsa.compile()
        return fsa

    def get_name(self) -> str:
        return "only allows legal args"


if __name__ == '__main__':
    input_tokens = torch.zeros(1, 9)
    start, end = 1, 2
    O, B_A0, I_A0, B_A1, I_A1, B_A2, I_A2 = 3, 4, 5, 6, 7, 8, 9
    token_to_key = {
        START_SYMBOL: start,
        END_SYMBOL: end,
        'O': O,
        'B-A0': B_A0,
        'I-A0': I_A0,
        'B-A1': B_A1,
        'I-A1': I_A1,
        'B-A2': B_A2,
        'I-A2': I_A2

    }
    constraint = DisallowArgConstraint()
    # automaton = constraint.build(input_tokens, token_to_key, legal_args=["A0"])
    automaton = constraint.build(input_tokens, token_to_key, legal_args=[])
    automaton.save("disallow_arg")
