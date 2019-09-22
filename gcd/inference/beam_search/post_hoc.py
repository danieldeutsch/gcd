import torch
from allennlp.common.util import END_SYMBOL
from allennlp.data import Vocabulary
from typing import Callable, Dict, List, Tuple

from gcd.inference.beam_search import ConstrainedBeamSearch, util
from gcd.inference.beam_search.unconstrained import UnconstrainedBeamSearch
from gcd.inference.constraints import ConstraintSet


StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name


@ConstrainedBeamSearch.register('post_hoc')
class PostHocBeamSearch(UnconstrainedBeamSearch):
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
        predictions, log_probs = self._search(start_predictions, start_state, step, None)

        # Select the output predictions by taking the first one which does not
        # violate any constraints OR the most probable
        batch_size, beam_size, _ = predictions.size()
        best_predictions = []
        for batch, constraint_set in enumerate(constraint_sets):
            best = None
            for beam in range(beam_size):
                prediction = util.ensure_one_end_index(predictions[batch, beam].tolist(), self._end_index)
                if constraint_set.get_violated_constraint(prediction) is None:
                    best = prediction
                    break

            if best is None:
                best = util.ensure_one_end_index(predictions[batch, 0].tolist(), self._end_index)
            best_predictions.append(best)

        # The working set is empty by definition
        working_sets = [[] for _ in range(batch_size)]
        violated_constraints = []
        for prediction, constraint_set in zip(best_predictions, constraint_sets):
            violated_constraints.append(constraint_set.get_all_violated_constraints_names(prediction))

        return best_predictions, log_probs, working_sets, violated_constraints
