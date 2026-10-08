import pytest

from textkit.ranges import binary_search, merge_intervals, overlaps, paginate, total_covered, total_pages


def test_overlaps_true():
    assert overlaps((1, 5), (4, 8))


def test_overlaps_touching_is_false():
    assert not overlaps((1, 5), (5, 8))


def test_merge_overlapping():
    assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]


def test_merge_touching():
    assert merge_intervals([(1, 2), (2, 4)]) == [(1, 4)]


def test_merge_unsorted_contained():
    assert merge_intervals([(5, 9), (1, 10)]) == [(1, 10)]


def test_total_covered():
    assert total_covered([(0, 5), (3, 8), (10, 12)]) == 10


def test_total_pages():
    assert total_pages(10, 3) == 4
    assert total_pages(9, 3) == 3
    assert total_pages(0, 3) == 0


def test_paginate_first_and_last():
    items = list(range(7))
    assert paginate(items, 1, 3) == [0, 1, 2]
    assert paginate(items, 3, 3) == [6]


def test_paginate_out_of_range():
    assert paginate([1, 2], 5, 3) == []


def test_binary_search_found():
    data = [1, 3, 5, 7, 9, 11]
    assert binary_search(data, 1) == 0
    assert binary_search(data, 7) == 3
    assert binary_search(data, 11) == 5


def test_binary_search_missing():
    assert binary_search([1, 3, 5], 4) == -1
    assert binary_search([], 4) == -1
