import pytest

from textkit.stats import clamp, mean, median, moving_average, percentile, variance


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5


def test_percentile_median_rank():
    assert percentile([15, 20, 35, 40, 50], 50) == 35


def test_percentile_bounds():
    data = [5, 1, 9, 3]
    assert percentile(data, 0) == 1
    assert percentile(data, 100) == 9


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_clamp_inside():
    assert clamp(5, 0, 10) == 5


def test_clamp_below_and_above():
    assert clamp(-3, 0, 10) == 0
    assert clamp(42, 0, 10) == 10


def test_clamp_bad_bounds():
    with pytest.raises(ValueError):
        clamp(1, 5, 0)


def test_moving_average():
    assert moving_average([1, 2, 3, 4, 5], 2) == [1.5, 2.5, 3.5, 4.5]


def test_moving_average_window_equals_length():
    assert moving_average([2, 4, 6], 3) == [4.0]


def test_variance():
    assert variance([2, 4, 4, 4, 5, 5, 7, 9]) == 4.0
