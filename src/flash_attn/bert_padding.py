"""Pure-PyTorch padding helpers required by veRL's training batch path."""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class _IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.input_size = values.shape
        flat = values.reshape(values.shape[0], -1)
        gathered = torch.gather(flat, 0, repeat(indices, "z -> z d", d=flat.shape[1]))
        return gathered.reshape(-1, *values.shape[1:])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        flat = grad_output.reshape(grad_output.shape[0], -1)
        grad = torch.zeros(
            (ctx.input_size[0], flat.shape[1]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad.scatter_add_(0, repeat(indices, "z -> z d", d=flat.shape[1]), flat)
        return grad.reshape(ctx.input_size), None


index_first_axis = _IndexFirstAxis.apply


def _index_put_first_axis(
    values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int
) -> torch.Tensor:
    output = torch.zeros(
        (first_axis_dim, *values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )
    output[indices] = values
    return output


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    output = _index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    unused_mask: torch.Tensor | None = None,
):
    all_masks = attention_mask if unused_mask is None else attention_mask + unused_mask
    lengths = all_masks.sum(dim=-1, dtype=torch.int32)
    used_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    packed = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)
    return packed, indices, cu_seqlens, int(lengths.max().item()), used_lengths


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]

