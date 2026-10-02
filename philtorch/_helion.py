import os

import helion
import helion.language as hl
import torch
from torch import Tensor

# Helion's own default is "full" autotuning, which makes the first call slow.
# Use "quick" unless the user chooses with HELION_AUTOTUNE_EFFORT, which must
# be set before philtorch is imported.
_AUTOTUNE_SETTINGS = (
    {} if os.environ.get("HELION_AUTOTUNE_EFFORT") else {"autotune_effort": "quick"}
)


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    **_AUTOTUNE_SETTINGS,
    static_shapes=False,
    dot_precision="ieee",
)
def lti_shared_A_recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
) -> Tensor:
    """
    Args:
        A (Tensor): State matrix of shape (M, M).
        zi (Tensor): Initial state of shape (B, M).
        x (Tensor): Input sequence of shape (B, T, M).

    Returns:
        Tensor: State sequence (B, T, M).
    """
    AT = A.transpose(-2, -1)
    batch = zi.shape[0]
    T = x.shape[1]
    M = A.shape[-1]
    output = torch.cat([zi.unsqueeze(1), x], dim=1)

    for tile_b in hl.tile(batch):
        for t in hl.grid(1, T + 1):
            for tile_m in hl.tile(M):
                output[tile_b, t, tile_m] += torch.matmul(
                    # output[tile_b, t, tile_m],
                    output[tile_b, t - 1, :],
                    AT[:, tile_m],
                )
    return output[:, 1:]


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    **_AUTOTUNE_SETTINGS,
    static_shapes=False,
    dot_precision="ieee",
)
def lti_recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
) -> Tensor:
    """
    Args:
        A (Tensor): State matrix of shape (B, M, M).
        zi (Tensor): Initial state of shape (B, M).
        x (Tensor): Input sequence of shape (B, T, M).

    Returns:
        Tensor: State sequence (B, T, M).
    """
    batch = zi.shape[0]
    # AT = A.transpose(-2, -1)
    T = x.shape[1]
    M = A.shape[-1]
    output = torch.cat([zi.unsqueeze(1), x], dim=1).unsqueeze(-1)

    for tile_b in hl.tile(batch):
        for t in hl.grid(1, T + 1):
            for tile_m in hl.tile(M):
                output[tile_b, t, tile_m, :] += torch.bmm(
                    # output[tile_b, t, tile_m, :],
                    A[tile_b, tile_m, :],
                    output[tile_b, t - 1, :, :],
                )
    return output[:, 1:].squeeze(-1)


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    **_AUTOTUNE_SETTINGS,
    static_shapes=False,
    dot_precision="ieee",
)
def lpv_shared_A_recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
) -> Tensor:
    """
    Args:
        A (Tensor): State matrix of shape (T, M, M).
        zi (Tensor): Initial state of shape (B, M).
        x (Tensor): Input sequence of shape (B, T, M).

    Returns:
        Tensor: State sequence (B, T, M).
    """
    AT = A.transpose(-2, -1)
    batch = zi.shape[0]
    T = x.shape[1]
    M = A.shape[-1]
    output = torch.cat([zi.unsqueeze(1), x], dim=1)

    for tile_b in hl.tile(batch):
        for t in hl.grid(1, T + 1):
            for tile_m in hl.tile(M):
                output[tile_b, t, tile_m] += torch.matmul(
                    # output[tile_b, t, tile_m],
                    output[tile_b, t - 1, :],
                    AT[t - 1, :, tile_m],
                )
    return output[:, 1:]


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    **_AUTOTUNE_SETTINGS,
    static_shapes=False,
    dot_precision="ieee",
)
def lpv_recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
) -> Tensor:
    """
    Args:
        A (Tensor): State matrix of shape (B, T, M, M).
        zi (Tensor): Initial state of shape (B, M).
        x (Tensor): Input sequence of shape (B, T, M).

    Returns:
        Tensor: State sequence (B, T, M).
    """
    batch = zi.shape[0]
    # AT = A.transpose(-2, -1)
    T = x.shape[1]
    M = A.shape[-1]
    output = torch.cat([zi.unsqueeze(1), x], dim=1).unsqueeze(-1)

    for tile_b in hl.tile(batch):
        for t in hl.grid(1, T + 1):
            for tile_m in hl.tile(M):
                output[tile_b, t, tile_m, :] += torch.bmm(
                    # output[tile_b, t, tile_m, :],
                    A[tile_b, t - 1, tile_m, :],
                    output[tile_b, t - 1, :, :],
                )
    return output[:, 1:].squeeze(-1)
