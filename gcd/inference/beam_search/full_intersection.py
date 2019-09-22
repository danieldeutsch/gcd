import torch
from allennlp.common.util import END_SYMBOL
from allennlp.data import Vocabulary
from typing import Callable, Dict, List, Tuple

from gcd.inference.beam_search import ConstrainedBeamSearch, util
from gcd.inference.constraints import ConstraintSet


StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name


@ConstrainedBeamSearch.register('full_intersection')
class FullIntersectionBeamSearch(ConstrainedBeamSearch):
    def __init__(self,
                 vocab: Vocabulary,
                 beam_size: int,
                 namespace: str = 'tokens',
                 end_symbol: str = None,
                 max_steps: int = 500,
                 per_node_beam_size: int = None) -> None:
        self.beam_size = beam_size
        end_symbol = end_symbol or END_SYMBOL
        self._end_index = vocab.get_token_index(end_symbol, namespace)
        self.max_steps = max_steps
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self,
               start_predictions: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType,
               constraint_sets: List[ConstraintSet]):
        for constraint_set in constraint_sets:
            constraint_set.force_full_intersection()

        predictions, log_probs = self._search(start_predictions, start_state, step, constraint_sets)
        top_predictions = [util.ensure_one_end_index(prediction[0].tolist(), self._end_index) for prediction in predictions]

        # All of the constraints are in the working set by definition
        batch_size = start_predictions.size(0)
        working_sets = [constraint_set.get_working_set() for constraint_set in constraint_sets]
        violated_constraints = [[] for _ in range(batch_size)]

        for constraint_set, prediction in zip(constraint_sets, top_predictions):
            assert constraint_set.get_violated_constraint(prediction) is None

        return top_predictions, log_probs, working_sets, violated_constraints
