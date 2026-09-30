def resolve_cuda_build(
    force_cuda_value, cuda_home, cuda_available, cuda_arch_list=None
):
    if force_cuda_value not in {"0", "1"}:
        raise RuntimeError(
            "PHILTORCH_FORCE_CUDA must be either '0' or '1'; "
            f"got {force_cuda_value!r}."
        )

    force_cuda = force_cuda_value == "1"
    if force_cuda and cuda_home is None:
        raise RuntimeError(
            "PHILTORCH_FORCE_CUDA=1 was requested, but CUDA_HOME is not set "
            "or the CUDA toolkit could not be found."
        )
    if (
        force_cuda
        and not cuda_available
        and (not cuda_arch_list or cuda_arch_list == "native")
    ):
        raise RuntimeError(
            "PHILTORCH_FORCE_CUDA=1 was requested on a host with no visible GPU, "
            "but TORCH_CUDA_ARCH_LIST is not set, so there is no safe target "
            "architecture to infer. Set TORCH_CUDA_ARCH_LIST to the target "
            'architectures, e.g. TORCH_CUDA_ARCH_LIST="8.0 8.6".'
        )

    return cuda_home is not None and (force_cuda or cuda_available)
