# coding: utf-8
from typing import List, Tuple

import nltk
from nltk.tree import ParentedTree


def get_sisters(node: ParentedTree) -> List[ParentedTree]:
    sisters = []
    # root node does not have sisters
    if node.parent() is None:
        return sisters

    for child in get_children(node.parent()):
        if child == node:
            continue
        sisters.append(child)
    return sisters


def get_children(node: ParentedTree) -> List[ParentedTree]:
    return [child for child in node]


def get_verbs_non_terminal_node(ptree: ParentedTree, verb_idx: int) -> ParentedTree:
    tree_location = ptree.leaf_treeposition(verb_idx)
    verbs_non_terminal = ptree[tree_location[:-1]]
    return verbs_non_terminal


def get_candidate_spans(treestring: str, verb_idx: int) -> List[Tuple[int, int]]:
    treestring = add_indices_to_terminals(treestring)
    cands = get_candidates(treestring, verb_idx)
    spans = []
    for cand in cands:
        start_tok, end_tok = cand[0], cand[-1]
        start = int(start_tok.rsplit("_", 1)[-1])
        end = int(end_tok.rsplit("_", 1)[-1])
        spans.append((start, end))

    if len(spans) == 0:
        tree = ParentedTree.fromstring(treestring)
        num_tokens = len(tree.leaves())
        spans = [(num_tokens, num_tokens)]  # this is a dummy span
    return spans


def get_candidates(treestring: str, verb_idx: int) -> List[List[str]]:
    tree = ParentedTree.fromstring(treestring)
    # Designate the predicate as the current node
    current = get_verbs_non_terminal_node(tree, verb_idx=verb_idx)
    candidates = []
    while current is not None:
        # collect its sisters (constituents attached at the same level as the predicate)
        for sister in get_sisters(current):
            if sister.label() == "CC":
                # unless its sisters are coordinated with the predicate.
                continue
            if sister.label() == "PP":
                # If a sister is a PP, also collect its immediate children
                for child in get_children(sister):
                    candidates += [child.leaves()]
            if sister is not None and sister.label() not in [".", "``", ",", ":"]:
                candidates += [sister.leaves()]
        current = current.parent()

    # remove candidates which are just a single token, because they will anyway respect the constraint
    new_candidates = []
    for cand in candidates:
        if len(cand) == 1:
            continue
        else:
            new_candidates.append(cand)
    candidates = new_candidates

    return candidates


def traverse_tree(tree):
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            traverse_tree(subtree)
        else:
            print(subtree, type(subtree))


def add_indices_to_terminals(treestring):
    tree = ParentedTree.fromstring(treestring)
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        non_terminal = tree[tree_location[:-1]]
        non_terminal[0] = non_terminal[0] + "_" + str(idx)
    return str(tree)
