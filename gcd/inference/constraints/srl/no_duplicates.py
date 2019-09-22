import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from gcd.inference.automata import FSA
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.srl import util


# @Constraint.register('no-duplicates')
class NoDuplicatesConstraint(Constraint):
    """
    Prevent a verb to have more than one A0's, more than one A1's etc.
    """
    def __init__(self, arg_kind: str):
        """
        :param arg_kind: something like A0, A1
        """
        self.arg_kind = arg_kind

    def build(self, input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> FSA:
        # batch_size, num_tokens = input_tokens.size()
        # assert batch_size == 1, batch_size
        # num_tokens -= 2  # <bos>, <eos>
        begin_label, inside_label = "B-" + self.arg_kind, "I-" + self.arg_kind
        fsa = FSA()
        symbol_table = util.set_symbol_table(fsa, token_to_key)
        start_key = symbol_table[START_SYMBOL]
        end_key = symbol_table[END_SYMBOL]

        start_state = fsa.add_state()
        not_yet_seen_state = fsa.add_state()
        inside_state = fsa.add_state()
        already_seen_state = fsa.add_state()
        end_state = fsa.add_state()

        # Set the start and final states
        fsa.set_start(start_state)
        fsa.set_final(end_state)

        # Add transitions from the start state to the first not yet seen state
        fsa.add_arc(start_state, not_yet_seen_state, start_key)

        # Add transitions to the final state
        fsa.add_arc(not_yet_seen_state, end_state, end_key)
        fsa.add_arc(inside_state, end_state, end_key)
        fsa.add_arc(already_seen_state, end_state, end_key)

        for token, key in symbol_table.items():
            if token in [START_SYMBOL, END_SYMBOL, begin_label, inside_label]:
                continue
            # Now add the keys for other labels like B-A1, B-A2, O etc.
            fsa.add_arc(not_yet_seen_state, not_yet_seen_state, key)
            fsa.add_arc(already_seen_state, already_seen_state, key)
            fsa.add_arc(inside_state, already_seen_state, key)

        begin_key = token_to_key[begin_label]
        inside_key = token_to_key[inside_label]
        # Edge from start state to inside state
        fsa.add_arc(not_yet_seen_state, inside_state, begin_key)
        fsa.add_arc(inside_state, inside_state, inside_key)

        # Finalize
        fsa.compile(optimize=False)
        return fsa

    def get_name(self)->str:
        return "no duplicates {}".format(self.arg_kind)


if __name__ == '__main__':
    input_tokens = torch.zeros(1, 5)
    start, end = 1, 2
    O, B_A0, I_A0 = 3, 4, 5
    token_to_key = {
        START_SYMBOL: start,
        END_SYMBOL: end,
        'O': O,
        'B-A0': B_A0,
        'I-A0': I_A0
    }
    constraint = NoDuplicatesConstraint("A0")
    automaton = constraint.build(input_tokens, token_to_key)
    automaton.save("nodup")
