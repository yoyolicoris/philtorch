"""Test-case generators shared by the test modules."""


def each_choice(*factors):
    """Cases in which every option of every factor appears at least once.

    Case i takes option i of each factor, cycling the shorter factors, so there
    are as many cases as the longest factor has options. It covers every branch
    that depends on one factor alone, as a shape check of independent inputs
    does, without the factors' full product.
    """
    n = max(len(factor) for factor in factors)
    return [tuple(factor[i % len(factor)] for factor in factors) for i in range(n)]
