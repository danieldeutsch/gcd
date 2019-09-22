import networkx as nx
import pynini
from collections import defaultdict
from overrides import overrides
from tqdm import tqdm
from typing import List, Tuple, Union

from gcd.inference.automata import Automaton


class PDA(Automaton):
    def __init__(self):
        super().__init__()
        self.parens = pynini.PdtParentheses()
        self.open_key_to_close_key = {}
        self.close_key_to_open_key = {}

    def add_paren(self, push_key: int, pop_key: int) -> None:
        self.parens.add_pair(push_key, pop_key)
        self.open_key_to_close_key[push_key] = pop_key
        self.close_key_to_open_key[pop_key] = push_key

    @overrides
    def compile(self, optimize: bool = True) -> None:
        super().compile(optimize)

        from gcd.inference.constraints.parsing.common import \
            EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL, \
            OPEN_PAREN_SYMBOL, CLOSE_PAREN_SYMBOL
        self.empty_push = self.token_to_key[EMPTY_STACK_OPEN_SYMBOL]
        self.empty_pop = self.token_to_key[EMPTY_STACK_CLOSE_SYMBOL]
        self.phrase_push = self.token_to_key[OPEN_PAREN_SYMBOL]
        self.phrase_pop = self.token_to_key[CLOSE_PAREN_SYMBOL]

        # Construct the topological sort ordering of the nodes
        graph = nx.DiGraph()
        for state in self.fst.states():
            graph.add_node(state)
            for arc in self.fst.arcs(state):
                graph.add_edge(state, arc.nextstate)

        self._is_dag = False
        if nx.is_directed_acyclic_graph(graph):
            self._is_dag = True
            reverse_topological_sort = list(reversed(list(nx.topological_sort(graph))))

            # Compute the valid stack configurations for every node
            # and every arc in the graph backwards
            self.valid_state_stacks = defaultdict(set)
            self.valid_arc_stacks = defaultdict(lambda: defaultdict(set))

            # Start the algorithm with the final state
            final_state = reverse_topological_sort[0]
            self.valid_state_stacks[final_state] = set([0])

            for from_state in tqdm(reverse_topological_sort[1:]):
                state_stacks = set()
                for arc in self.fst.arcs(from_state):
                    to_state = arc.nextstate
                    key = arc.ilabel
                    arc_stacks = set()

                    if key == self.empty_push:
                        if 1 in self.valid_state_stacks[to_state]:
                            arc_stacks.add(0)
                    elif key == self.phrase_push:
                        for stack in self.valid_state_stacks[to_state]:
                            if stack > 1:
                                arc_stacks.add(stack - 1)
                    elif key == self.empty_pop:
                        arc_stacks.add(1)
                    elif key == self.phrase_pop:
                        for stack in self.valid_state_stacks[to_state]:
                            arc_stacks.add(stack + 1)
                    else:
                        for stack in self.valid_state_stacks[to_state]:
                            arc_stacks.add(stack)

                    self.valid_arc_stacks[from_state][key] = arc_stacks
                    state_stacks.update(arc_stacks)

                self.valid_state_stacks[from_state] = state_stacks

    @overrides
    def accepts(self, input_sequence: Union[Automaton, List[int]]) -> bool:
        if isinstance(input_sequence, list):
            automaton = self.convert_to_automaton(input_sequence)
        else:
            automaton = input_sequence
        intersection = pynini.pdt_compose(self.fst, automaton, self.parens)
        return pynini.pdt_expand(intersection, self.parens).num_states() != 0

    @overrides
    def step(self, input_state: int, input_stack: int, key: int) -> Tuple[int, int]:
        assert self._compiled
        if not self._is_dag:
            raise ValueError(f'"step" is not implemented for non-DAGs')

        search_stack = [(input_state, input_stack)]
        visited = defaultdict(set)

        while search_stack:
            state, stack = search_stack.pop()
            for arc in self.fst.arcs(state):
                arc_key, to_state = arc.ilabel, arc.nextstate
                valid_stacks = self.valid_arc_stacks[state][arc_key]

                # If the stack we have is not valid for this arc, we
                # cannot take this arc
                if stack not in valid_stacks:
                    continue
                if key == arc_key:
                    return to_state, stack
                if arc_key in visited[to_state]:
                    continue

                if arc_key == self.empty_push or arc_key == self.phrase_push:
                    search_stack.append([to_state, stack + 1])
                    visited[to_state].add(arc_key)
                elif arc_key == self.empty_pop:
                    if stack == 1:
                        search_stack.append([to_state, 0])
                        visited[to_state].add(arc_key)
                elif arc_key == self.phrase_pop:
                    if stack > 1:
                        search_stack.append([to_state, stack - 1])
                        visited[to_state].add(arc_key)
                else:
                    # This is a different vocabulary item, so we cannot
                    # take this action
                    pass
        raise Exception(f'Could not step from {input_state} with stack {input_stack}')

    def get_valid_actions(self, state: int, stack: int) -> List[int]:
        assert self._compiled
        if not self._is_dag:
            raise ValueError(f'"step" is not implemented for non-DAGs')

        search_stack = [(state, stack)]
        visited = defaultdict(set)
        valid_keys_to_metadata = {}

        while search_stack:
            state, stack = search_stack.pop()
            for arc in self.fst.arcs(state):
                key, to_state = arc.ilabel, arc.nextstate
                valid_stacks = self.valid_arc_stacks[state][key]
                if stack not in valid_stacks:
                    continue

                if key in visited[to_state]:
                    continue
                if key == self.empty_push or key == self.phrase_push:
                    search_stack.append([to_state, stack + 1])
                    visited[to_state].add(key)
                elif key == self.empty_pop:
                    if stack == 1:
                        search_stack.append([to_state, 0])
                        visited[to_state].add(key)
                elif key == self.phrase_pop:
                    if stack > 1:
                        search_stack.append([to_state, stack - 1])
                        visited[to_state].add(key)
                else:
                    if key in valid_keys_to_metadata:
                        assert valid_keys_to_metadata[key] == (to_state, stack)
                    else:
                        valid_keys_to_metadata[key] = (to_state, stack)
                    visited[to_state].add(key)

        return list(valid_keys_to_metadata.keys())
