from torch import Tensor
import torch
import helion
import helion.language as hl


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    autotune_effort="quick",
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
                output[tile_b, t, tile_m] = torch.addmm(
                    output[tile_b, t, tile_m],
                    output[tile_b, t - 1, :],
                    AT[:, tile_m],
                )
    return output[:, 1:]


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    autotune_effort="quick",
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
                output[tile_b, t, tile_m, :] = torch.baddbmm(
                    output[tile_b, t, tile_m, :],
                    A[tile_b, tile_m, :],
                    output[tile_b, t - 1, :, :],
                )
    return output[:, 1:].squeeze(-1)


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    autotune_effort="quick",
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
                output[tile_b, t, tile_m] = torch.addmm(
                    output[tile_b, t, tile_m],
                    output[tile_b, t - 1, :],
                    AT[t - 1, :, tile_m],
                )
    return output[:, 1:]


@helion.kernel(
    # config=helion.Config(block_sizes=[4, 16]),
    autotune_effort="quick",
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
    output = torch.cat([zi.unsqueeze(1), x], dim=1)

    for tile_b in hl.tile(batch):
        for t in hl.grid(1, T + 1):
            for tile_m in hl.tile(M):
                output[tile_b, t, tile_m] = torch.baddbmm(
                    output[tile_b, t, tile_m].unsqueeze(-1),
                    A[tile_b, t - 1, tile_m, :],
                    output[tile_b, t - 1, :].unsqueeze(-1),
                ).squeeze(-1)
    return output[:, 1:]
