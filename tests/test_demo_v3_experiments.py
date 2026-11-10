import numpy as np
import pytest

from train_demo_v3_experiments import _sha256_arrays, _topk_hits


def test_demo_v3_top_five_accuracy_and_probability_validation():
    probabilities = np.asarray(
        [
            [0.01, 0.02, 0.03, 0.04, 0.90],
            [0.05, 0.80, 0.05, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    hits, max_sum_error = _topk_hits(np.asarray([0, 4]), probabilities, k=5)
    assert hits == 2
    assert max_sum_error == pytest.approx(0.0)


def test_demo_v3_top_five_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="non-finite"):
        _topk_hits(np.asarray([0]), np.asarray([[np.nan, 0.0, 0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="outside"):
        _topk_hits(np.asarray([0]), np.asarray([[1.1, 0.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="sum"):
        _topk_hits(np.asarray([0]), np.asarray([[0.5, 0.0, 0.0, 0.0, 0.0]]))


def test_demo_v3_array_hash_is_stable_and_content_sensitive():
    first = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    second = np.asarray([2, 1], dtype=np.int32)
    assert _sha256_arrays(first, second) == _sha256_arrays(first.copy(), second.copy())
    assert _sha256_arrays(first, second) != _sha256_arrays(first, second[::-1])
