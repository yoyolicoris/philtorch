import pytest
import torch

from philtorch.lti import diag_state_space, state_space


@pytest.mark.cuda
def test_diag_state_space_leaves_a_transposed_A_unchanged():
    # torch.linalg.eig on CUDA overwrites a transposed input in place.
    a = 0.3 * torch.randn(4, 3, 3, device="cuda", dtype=torch.double)
    A, x = a.mT, torch.randn(4, 20, device="cuda", dtype=torch.double)
    before = A.clone()
    y = diag_state_space(x, A=A)
    torch.testing.assert_close(A, before, rtol=0, atol=0)
    torch.testing.assert_close(y, state_space(before, x))
