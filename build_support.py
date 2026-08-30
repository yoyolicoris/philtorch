def resolve_cuda_build(force_cuda_value, cuda_home, cuda_available):
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

    return cuda_home is not None and (force_cuda or cuda_available)
