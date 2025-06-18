import torch
from torch import Tensor
from torchlpc import sample_wise_lpc
from typing import Optional, Union, Tuple


def lpv_fir(
    b: Tensor, x: Tensor, zi: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    raise NotImplementedError(
        "LPV FIR filter is not implemented yet. Please use a different method."
    )


def lpv_allpole(
    a: Tensor, x: Tensor, zi: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    raise NotImplementedError(
        "LPV all-pole filter is not implemented yet. Please use a different method."
    )
