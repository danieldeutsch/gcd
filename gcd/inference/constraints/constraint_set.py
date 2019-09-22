import torch
from allennlp.common.from_params import FromParams
from allennlp.data import Vocabulary
from typing import List, Optional, Tuple

from gcd.inference.constraints import Constraint


class ConstraintSet(FromParams):
    def __init__(self,
                 constraints: List[Constraint],
                 vocab: Vocabulary,
                 namespace: str) -> None:
        self.constraints = constraints
        self.automata = []
        self.constraint_automaton = None
        self.working_set = set()
        self.non_working_set = set(range(len(constraints)))

        self.vocab = vocab
        self.namespace = namespace
        self._build_indexes()

    def _build_indexes(self):
        self.token_to_key = {}
        self.index_to_key = {}
        self.key_to_index = {}
        self.all_indices = []

        # The keys are offset by 1 because pynini has problems if the
        # key is equal to 0
        for index, token in self.vocab.get_index_to_token_vocabulary(self.namespace).items():
            key = index + 1
            self.token_to_key[token] = key
            self.index_to_key[index] = key
            self.key_to_index[key] = index
            self.all_indices.append(index)

    def setup(self, input_tokens: torch.Tensor, *args, **kwargs) -> None:
        self.automata = []
        for constraint in self.constraints:
            automaton = constraint.build(input_tokens, self.token_to_key, *args, **kwargs)
            self.automata.append(automaton)
        self.constraint_automaton = None
        self.working_set = set()
        self.non_working_set = set(range(len(self.constraints)))

    def force_full_intersection(self) -> None:
        for idx, _ in enumerate(self.constraints):
            self.add_contraint_to_working_set(idx)

    def is_valid(self, output_tokens: List[int]) -> bool:
        output_keys = [self.index_to_key[index] for index in output_tokens]
        return all(automaton.accepts(output_keys) for automaton in self.automata)

    def get_start(self) -> int:
        if self.constraint_automaton is None:
            return None
        return self.constraint_automaton.get_start()

    def step(self, state: int, stack: int, index: int) -> Tuple[int, int]:
        if self.constraint_automaton is None:
            return None, None
        key = self.index_to_key[index]
        return self.constraint_automaton.step(state, stack, key)

    def get_valid_actions(self, state: int, stack: int) -> List[int]:
        if self.constraint_automaton is None:
            return self.all_indices
        valid_keys = self.constraint_automaton.get_valid_actions(state, stack)
        valid_actions = [self.key_to_index[key] for key in valid_keys]
        return valid_actions

    def get_violated_constraint(self, output_tokens: List[int]) -> Optional[int]:
        """Returns the index of the constraint in the non-working set which is violated."""
        if len(self.non_working_set) == 0:
            return None

        # Optimization hack: make the output_tokens into an automaton exactly once
        # so it doesn't need to be done every time in accepts
        output_keys = [self.index_to_key[index] for index in output_tokens]
        input_fst = self.automata[0].convert_to_automaton(output_keys)
        for index in self.non_working_set:
            if not self.automata[index].accepts(input_fst):
                return index
        return None

    def get_all_violated_constraints(self, output_tokens: List[int]) -> List[int]:
        violated = []
        output_keys = [self.index_to_key[index] for index in output_tokens]
        input_fst = self.automata[0].convert_to_automaton(output_keys)
        for i, automaton in enumerate(self.automata):
            if not automaton.accepts(input_fst):
                violated.append(i)
        return violated

    def add_contraint_to_working_set(self, index: int) -> None:
        # print(f'Adding {self.constraints[index].get_name()} to the working set')
        self.working_set.add(index)
        self.non_working_set.remove(index)
        if self.constraint_automaton is None:
            self.constraint_automaton = self.automata[index]
        else:
            self.constraint_automaton = self.constraint_automaton.intersect(self.automata[index])

    def add_all_constraints_to_working_set(self) -> None:
        """Adds all of the constraints in the non-working set to the working set."""
        for index in reversed(list(self.non_working_set)):
            self.add_contraint_to_working_set(index)

    def get_working_set(self) -> List[str]:
        return [self.constraints[index].get_name() for index in self.working_set]

    def get_all_violated_constraints_names(self, output_tokens: List[int]) -> List[str]:
        return [self.constraints[index].get_name() for index in self.get_all_violated_constraints(output_tokens)]
