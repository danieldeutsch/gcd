import torch
import warnings
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL
from allennlp.data import Vocabulary
from typing import Callable, Dict, List, Tuple

from gcd.inference.beam_search import ConstrainedBeamSearch, util
from gcd.inference.constraints import ConstraintSet


StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name


# The naming of this is silly. It is an unconstrained search, but it implements
# the constrained beam search interface
@ConstrainedBeamSearch.register('unconstrained')
class UnconstrainedBeamSearch(ConstrainedBeamSearch):
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

    def _search(self,
                start_predictions: torch.Tensor,
                start_state: StateType,
                step: StepFunctionType,
                constraint_sets: List[ConstraintSet]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = start_predictions.size(0)

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(start_predictions, start_state)

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ConfigurationError(f"Target vocab size ({num_classes:d}) too small "
                                     f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                                     f"Please decrease beam_size or per_node_beam_size.")

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = \
                start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
                (batch_size * self.beam_size, num_classes),
                float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor.\
                    unsqueeze(1).\
                    expand(batch_size, self.beam_size, *last_dims).\
                    reshape(batch_size * self.beam_size, *last_dims)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                    batch_size * self.beam_size,
                    num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                    last_predictions_expanded == self._end_index,
                    log_probs_after_end,
                    class_log_probabilities
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes = \
                cleaned_log_probabilities.topk(self.per_node_beam_size)

            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probabilities = last_log_probabilities.\
                    unsqueeze(2).\
                    expand(batch_size, self.beam_size, self.per_node_beam_size).\
                    reshape(batch_size * self.beam_size, self.per_node_beam_size)

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.\
                    reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.\
                    reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices / self.per_node_beam_size

            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.\
                        view(batch_size, self.beam_size, *([1] * len(last_dims))).\
                        expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = state_tensor.\
                        reshape(batch_size, self.beam_size, *last_dims).\
                        gather(1, expanded_backpointer).\
                        reshape(batch_size * self.beam_size, *last_dims)

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # Add the starting token
        reconstructed_predictions.append(start_predictions.unsqueeze(1).unsqueeze(2).expand(-1, self.beam_size, -1))

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities

    def search(self,
               start_predictions: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType,
               constraint_sets: List[ConstraintSet]):
        predictions, log_probs = self._search(start_predictions, start_state, step, None)
        top_predictions = [util.ensure_one_end_index(prediction[0].tolist(), self._end_index) for prediction in predictions]

        # The working set is empty by definition
        batch_size = start_predictions.size(0)
        working_sets = [[] for _ in range(batch_size)]
        violated_constraints = []
        for prediction, constraint_set in zip(top_predictions, constraint_sets):
            violated_constraints.append(constraint_set.get_all_violated_constraints_names(prediction))

        return top_predictions, log_probs, working_sets, violated_constraints
