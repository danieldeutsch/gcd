import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict, List, Tuple

from gcd.inference.automata import FSA
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.srl import util
from gcd.inference.constraints.srl.union_find import UnionFind
from intervaltree import IntervalTree

Span = Tuple[int, int]


def make_fsa_for_span(candidate_span: Tuple[int, int], num_tokens: int, arg_kinds: List[str],
                      token_to_key: Dict[str, int]) -> FSA:
    fsa = FSA()
    symbol_table = util.set_symbol_table(fsa, token_to_key)
    start_key = symbol_table[START_SYMBOL]
    end_key = symbol_table[END_SYMBOL]

    # There is a state for seeing 0, 1, ..., num_tokens tokens plus
    # a starting and stopping state
    start_state = fsa.add_state()
    states = [fsa.add_state() for _ in range(num_tokens + 1)]
    final_state = fsa.add_state()

    # Set the start and final states
    fsa.set_start(start_state)
    fsa.set_final(final_state)

    # Add the starting and ending transtiions
    fsa.add_arc(start_state, states[0], start_key)
    fsa.add_arc(states[-1], final_state, end_key)

    non_trivial_states = states
    # print("len(non_trivial_states)", len(non_trivial_states))
    start_idx, end_idx = candidate_span
    # tokens upto span can take any label
    for s_idx, state in enumerate(non_trivial_states[:start_idx]):
        for token, key in symbol_table.items():
            next_state = non_trivial_states[s_idx + 1]
            # everything else goes
            fsa.add_arc(state, next_state, key)

    if start_idx < num_tokens:
        for kind in arg_kinds:
            # traverse the edge that labels the span's first token
            new_state = fsa.add_state()
            fsa.add_arc(non_trivial_states[start_idx], new_state, symbol_table["B-" + kind])
            fsa.add_arc(non_trivial_states[start_idx], non_trivial_states[start_idx + 1], symbol_table["O"])
            # traverse the edges that labels the span's other tokens
            for s_idx in range(start_idx + 1, end_idx):
                next_new_state = fsa.add_state()
                fsa.add_arc(new_state, next_new_state, symbol_table["I-" + kind])
                new_state = next_new_state
                fsa.add_arc(non_trivial_states[s_idx], non_trivial_states[s_idx + 1], symbol_table["O"])
            fsa.add_arc(new_state, non_trivial_states[end_idx + 1], symbol_table["I-" + kind])
            fsa.add_arc(non_trivial_states[end_idx], non_trivial_states[end_idx + 1], symbol_table["O"])

    # now you are past the span, everything goes
    for s_idx in range(end_idx + 1, num_tokens):
        for token, key in symbol_table.items():
            fsa.add_arc(non_trivial_states[s_idx], non_trivial_states[s_idx + 1], key)
    # Finalize
    fsa.compile()
    return fsa


def partition_spans(spans: List[Span]) -> Tuple[List[List[Span]], List[Span]]:
    """
    partitions a list of spans into

    1. a list of span clusters, where each cluster contains spans that overlap somehow

    2. a list of spans that are non-overlapping.
    :param spans:
    :return:
    """
    uf = UnionFind()
    spans_so_far = IntervalTree()
    for span in spans:
        start, end = span
        overlaps_with = spans_so_far.overlap(begin=start, end=end)
        if len(overlaps_with) > 0:
            for parent in list(overlaps_with):
                parent_span = parent.begin, parent.end
                # print(parent)
                # print(span)
                uf.union(parent_span, span)
        else:
            spans_so_far.addi(begin=start, end=end)
            uf.union(span)
    # parent to cluster dict
    p2c = {}
    for span in spans:
        parent = uf[span]
        if parent not in p2c:
            p2c[parent] = []
        p2c[parent].append(span)
    # non overlap spans are those whose cluster contain just them
    non_overlap_spans: List[Span] = [parent for parent in p2c if len(p2c[parent]) == 1]
    # rest overlap
    overlap_groups: List[List[Span]] = [p2c[parent] for parent in p2c if len(p2c[parent]) > 1]
    # print(parent2cluster)
    return overlap_groups, non_overlap_spans


# @Constraint.register('arg-candidates')
class ArgumentCandidatesConstraint(Constraint):
    def __init__(self):
        pass

    def build(self, input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> FSA:
        # print(input_tokens.size())
        num_tokens = input_tokens.size()[0]
        # assert batch_size == 1, batch_size
        candidate_spans = kwargs["candidate_spans"]
        arg_kinds = kwargs["arg_kinds"]
        # print("candidate_spans", candidate_spans)
        if candidate_spans == [(num_tokens, num_tokens)]:
            fsa = make_fsa_for_span(candidate_span=candidate_spans[0],
                                    num_tokens=num_tokens,
                                    arg_kinds=arg_kinds,
                                    token_to_key=token_to_key)
            return fsa
        overlap_clusters, non_overlap_spans = partition_spans(spans=candidate_spans)
        # print("overlap_clusters", overlap_clusters)
        # print("non_overlap_spans", non_overlap_spans)
        # this is where I will collect FSAs that will be intersected at the end
        intersect_list: List[FSA] = []

        # for each cluster, union the FSA of the spans inside it
        for cluster in overlap_clusters:
            span: Span = cluster[0]
            fsa = make_fsa_for_span(candidate_span=span,
                                    num_tokens=num_tokens,
                                    arg_kinds=arg_kinds,
                                    token_to_key=token_to_key)
            for span in cluster[1:]:
                f = make_fsa_for_span(candidate_span=span,
                                      num_tokens=num_tokens,
                                      arg_kinds=arg_kinds,
                                      token_to_key=token_to_key)
                fsa = fsa.union(f)
            intersect_list.append(fsa)
        # print("intersect_list after clusters", intersect_list)
        # for the remaining non-overlapping spans, just add their FSA to the intersect list
        for span in non_overlap_spans:
            f = make_fsa_for_span(candidate_span=span,
                                  num_tokens=num_tokens,
                                  arg_kinds=arg_kinds,
                                  token_to_key=token_to_key)
            intersect_list.append(f)
        # print("intersect_list after non-overlap", intersect_list)
        fsa = intersect_list[0]
        for f in intersect_list[1:]:
            fsa = fsa.intersect(f)

        return fsa

    def get_name(self) -> str:
        return "argument candidates"


if __name__ == '__main__':
    pass
    input_tokens = torch.zeros(12)
    start, end = 1, 2
    O, B_A0, I_A0, B_A1, I_A1 = 3, 4, 5, 6, 7
    token_to_key = {
        'O': O,
        'B-A0': B_A0,
        'I-A0': I_A0,
        'B-A1': B_A1,
        'I-A1': I_A1,
    }
    constraint = ArgumentCandidatesConstraint()
    # candidate_spans = [(3, 5), (3, 6), (1, 2)]
    # candidate_spans = [(3, 5)]
    # candidate_spans = [(9, 10), (11, 22), (1, 4)]  # Non-overlapping spans
    # candidate_spans = [(8, 10), (9, 10), (11, 22), (1, 4)]  # one overlapping span
    # candidate_spans = [(8, 10), (9, 10)]  # one overlapping span
    candidate_spans = [(12, 12)]  # degenerate span
    arg_kinds = ["A0", "A1"]
    automaton = constraint.build(input_tokens=input_tokens, token_to_key=token_to_key,
                                 candidate_spans=candidate_spans, arg_kinds=arg_kinds)
    automaton.save("argcands")

    # in_tokens = torch.zeros(30)
    # O, B_A0, I_A0, B_A1, I_A1, B_A2, I_A2, B_A3, I_A3, B_A4, I_A4, B_A5, I_A5 = 3, 4, 5, 6, 7, 8, \
    #                                                                             9, 10, 11, 12, 13, 14, 15
    # token_to_key_dict = {
    #     'O': O,
    #     'B-A0': B_A0,
    #     'I-A0': I_A0,
    #     'B-A1': B_A1,
    #     'I-A1': I_A1,
    #     # 'B-A2': B_A2,
    #     # 'I-A2': I_A2,
    #     # 'B-A3': B_A3,
    #     # 'I-A3': I_A3,
    #     # 'B-A4': B_A4,
    #     # 'I-A4': I_A4,
    #     # 'B-A5': B_A5,
    #     # 'I-A5': I_A5,
    # }
    # constraint = ArgumentCandidatesConstraint()
    # # candidate_spans = [(3, 5), (3, 6), (1, 2)]
    # # candidate_spans = [(3, 5)]
    # candidate_spans = [(9, 10), (11, 22), (1, 4)]  # Non-overlapping spans
    # arg_kinds = ["A0", "A1"]  # core_arg_kinds
    # automaton = constraint.build(input_tokens=in_tokens, token_to_key=token_to_key_dict,
    #                              candidate_spans=candidate_spans, arg_kinds=arg_kinds)
    # automaton.save("argcands")
