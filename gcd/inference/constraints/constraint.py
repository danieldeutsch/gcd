import torch
from allennlp.common import Registrable
from typing import Dict

from gcd.inference.automata import Automaton


class Constraint(Registrable):
    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int],
              *args,
              **kwargs) -> Automaton:
        """
        Build the automaton which represents this constraint.

        args:
            input_tokens: the (1, num_tokens)-sized tensor which represents
                the input instance.
            token_to_key: the mapping from the target vocabulary to pynini
                keys that should be used to construct the automaton. Other
                keys can be added, but the decoded tokens will be converted
                to these keys when passed to the automaton for acceptance checking.
        """
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError
