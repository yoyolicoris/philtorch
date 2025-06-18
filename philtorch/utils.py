from functools import reduce
from typing import Callable, Tuple


def chain_functions(*functions: Callable) -> Callable:
    def closure(*args: Tuple) -> Tuple:
        return reduce(
            lambda acc, func: func(*acc) if isinstance(acc, tuple) else func(acc),
            functions,
            args,
        )

    return closure
