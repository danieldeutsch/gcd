from typing import List


def ensure_one_end_index(sequence: List[int], end_index: int):
    stripped = []
    for token in sequence:
        if token != end_index:
            stripped.append(token)
        else:
            stripped.append(token)
            break

    if stripped[-1] != end_index:
        stripped.append(end_index)

    return stripped
