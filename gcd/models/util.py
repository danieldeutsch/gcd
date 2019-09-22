import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List


def pack_sequence(X, mask):
    assert X.dim() >= 2
    assert X.size(0) == mask.size(0)
    assert X.size(1) == mask.size(1)

    # Sort X from longest to smallest lengths
    lengths = mask.float().sum(dim=-1)
    seq_lengths, seq_idx = lengths.sort(0, descending=True)
    seq_lengths = seq_lengths.int().data.tolist()
    X = X[seq_idx]

    packed = pack_padded_sequence(X, seq_lengths, batch_first=True)
    return packed


def unpack_sequence(packed_scores, hidden, mask):
    unpacked, _ = pad_packed_sequence(packed_scores, batch_first=True)

    lengths = mask.float().sum(dim=-1)
    _, seq_idx = lengths.sort(0, descending=True)
    _, original_idx = seq_idx.sort(0, descending=False)

    scores = unpacked[original_idx]
    h, c = hidden
    h = h[:, original_idx, :]
    c = c[:, original_idx, :]
    return scores, (h, c)


def masked_argmax(tensor: torch.Tensor, mask: torch.ByteTensor) -> int:
    assert tensor.dim() == 1
    assert mask.dim() == 1
    nonzero = mask.nonzero().squeeze(1)
    index = torch.argmax(tensor[mask]).item()
    return nonzero[index].item()


def masked_topk(tensor: torch.Tensor, mask: torch.ByteTensor, k: int) -> List[int]:
    assert tensor.dim() == 1
    assert mask.dim() == 1
    nonzero = mask.nonzero().squeeze(1)
    k = min(k, nonzero.numel())
    top_scores, top_indices = torch.topk(tensor[mask], k=k)
    top_indices = [nonzero[index] for index in top_indices]
    return top_scores, top_indices
