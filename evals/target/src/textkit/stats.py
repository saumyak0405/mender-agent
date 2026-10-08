"""Small statistics helpers."""
import math


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    if p < 0 or p > 100:
        raise ValueError("p must be within [0, 100]")
    ordered = sorted(values)
    if p == 0:
        return ordered[0]
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[rank - 1]


def clamp(x: float, low: float, high: float) -> float:
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(x, high))


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            out.append(total / window)
    return out


def variance(values: list[float]) -> float:
    """Population variance."""
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)
