"""Nested splits partition validation matches without leaking the outer match."""

import pytest

from experiments.cross_validation import nested_splits

MATCHES = [f"DFL-MAT-J03W{suffix}" for suffix in ("MX", "N1", "OH", "OY", "PY", "QQ", "R9")]


def test_seven_outer_folds_cover_each_inner_match_once_and_separate_pair():
    folds = nested_splits(reversed(MATCHES), seed=17)
    assert folds == nested_splits(MATCHES, seed=17)
    assert len(folds) == 7
    assert [fold["outer_test"][0] for fold in folds] == sorted(MATCHES)
    for fold in folds:
        outer = fold["outer_test"][0]
        assert outer not in fold["outer_train"]
        assert len(fold["outer_train"]) == 6
        assert len(fold["inner"]) == 3
        assert sorted(len(inner["validation"]) for inner in fold["inner"]) == [2, 2, 2]
        assert sorted(match for inner in fold["inner"] for match in inner["validation"]) == fold["outer_train"]
        for inner in fold["inner"]:
            assert len(inner["train"]) == 4
            assert set(inner["train"]).isdisjoint(inner["validation"])
            assert set(inner["train"]) | set(inner["validation"]) == set(fold["outer_train"])
        if set(MATCHES[:2]).issubset(fold["outer_train"]):
            assert not any(set(MATCHES[:2]).issubset(inner["validation"]) for inner in fold["inner"])


def test_small_samples_and_invalid_ids():
    folds = nested_splits(["a", "b", "c"], inner_folds=3)
    assert all(len(fold["inner"]) == 2 for fold in folds)
    with pytest.raises(ValueError, match="distinct"):
        nested_splits(["a", "a", "b"])
    with pytest.raises(ValueError, match="invalid"):
        nested_splits(["a", "b", "bad/match"])
    with pytest.raises(ValueError, match="inner_folds"):
        nested_splits(["a", "b", "c"], inner_folds=1)
