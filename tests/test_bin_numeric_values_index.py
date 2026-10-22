"""The bin index of `bin_numeric_values` matches `search_sorted` without a per-row window.

`_get_val_bin_idx` used to compute `bin_endpoints.list.explode().search_sorted(value, side="right")` once per
row, through `.over()` on a unique row index. That allocates per-group state for every row (several KB),
which exhausted memory on shards of a few million rows. The current formulation counts, within each row's
endpoint list, the endpoints `<=` the value. These tests pin it to the reference semantics.
"""

import random

import polars as pl
import pytest

from MEDS_transforms.stages.bin_numeric_values.bin_numeric_values import _get_val_bin_idx


def reference_bin_idx(df: pl.DataFrame) -> pl.Series:
    """The previous, window-based formulation."""
    val_col = pl.col("numeric_value")
    bin_endpoints = pl.col("bin_endpoints")
    do_bin = bin_endpoints.is_not_null() & val_col.is_not_null()
    idx_expr = bin_endpoints.list.explode().search_sorted(val_col, side="right").over("__idx")
    return df.with_row_index("__idx").select(pl.when(do_bin).then(idx_expr).alias("idx"))["idx"]


def current_bin_idx(df: pl.DataFrame) -> pl.Series:
    return df.select(_get_val_bin_idx(pl.col("bin_endpoints")).alias("idx"))["idx"]


def random_rows(seed: int, n_rows: int) -> pl.DataFrame:
    rng = random.Random(seed)
    values, endpoints = [], []
    for _ in range(n_rows):
        n_endpoints = rng.choice([1, 2, 9, 22])
        edges = sorted(round(rng.uniform(-10, 10), 1) for _ in range(n_endpoints))
        kind = rng.random()
        if kind < 0.1:
            value = None
        elif kind < 0.15:
            value = float("nan")
        elif kind < 0.35:
            value = rng.choice(edges)  # exactly on an endpoint (ties go right, as side="right")
        else:
            value = round(rng.uniform(-12, 12), 1)
        values.append(value)
        endpoints.append(None if rng.random() < 0.05 else edges)
    return pl.DataFrame(
        {"numeric_value": values, "bin_endpoints": endpoints},
        schema={"numeric_value": pl.Float64, "bin_endpoints": pl.List(pl.Float64)},
    )


@pytest.mark.parametrize("seed", range(5))
def test_matches_search_sorted_reference(seed: int):
    df = random_rows(seed, 2_000)
    expected = reference_bin_idx(df)
    actual = current_bin_idx(df)
    assert actual.dtype == expected.dtype
    assert actual.to_list() == expected.to_list()


def test_edge_cases_match_reference():
    nan = float("nan")
    df = pl.DataFrame(
        {
            "numeric_value": [nan, -1.0, 0.0, 0.0, 2.0, 5.0, None, 1.0, nan],
            "bin_endpoints": [
                [0.0, 1.0, 2.0],  # NaN sorts after every endpoint
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],  # on the first endpoint
                [0.0, 0.0, 1.0],  # duplicated endpoints
                [0.0, 1.0, 2.0],  # on the last endpoint
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                None,
                [0.0],
            ],
        },
        schema={"numeric_value": pl.Float64, "bin_endpoints": pl.List(pl.Float64)},
    )
    assert current_bin_idx(df).to_list() == reference_bin_idx(df).to_list()
    assert current_bin_idx(df).to_list() == [3, 0, 1, 2, 3, 3, None, None, 1]


def test_float32_values_match_reference():
    df = random_rows(7, 500).with_columns(pl.col("numeric_value").cast(pl.Float32))
    assert current_bin_idx(df).to_list() == reference_bin_idx(df).to_list()
