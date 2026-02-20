"""
Data Splitter — splits OHLCV data into train/test sets for out-of-sample testing.

Supports:
  - Date-based split: train ends before a given date, test starts at that date
  - Percentage-based split: first N% of rows for training, rest for testing

Both methods preserve strict chronological order with no shuffling.
"""

from typing import Tuple

import pandas as pd


def split_data(
    data: pd.DataFrame,
    split_method: str = "percentage",
    split_date: str = None,
    train_pct: float = 70.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split OHLCV data into training and testing sets.

    Args:
        data: OHLCV DataFrame with a DatetimeIndex (must be sorted chronologically).
        split_method: "date" for date-based split, "percentage" for ratio split.
        split_date: Cut-off date string (e.g. "2019-01-01"). Required when
                     split_method is "date".
        train_pct: Percentage of data for training (e.g. 70.0). Used when
                    split_method is "percentage".

    Returns:
        (train_data, test_data) — two non-overlapping DataFrames.

    Raises:
        ValueError: on invalid arguments or if either split is empty.
    """
    if data is None or data.empty:
        raise ValueError("Input data is empty.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex.")

    # Ensure chronological order
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    if split_method == "date":
        if not split_date:
            raise ValueError("split_date is required for date-based splitting.")

        cut = pd.Timestamp(split_date)

        if cut <= data.index[0]:
            raise ValueError(
                f"split_date ({split_date}) is at or before data start "
                f"({data.index[0].date()}). Training set would be empty."
            )
        if cut > data.index[-1]:
            raise ValueError(
                f"split_date ({split_date}) is after data end "
                f"({data.index[-1].date()}). Test set would be empty."
            )

        train_data = data[data.index < cut].copy()
        test_data = data[data.index >= cut].copy()

    elif split_method == "percentage":
        if not (1.0 <= train_pct <= 99.0):
            raise ValueError(
                f"train_pct must be between 1 and 99, got {train_pct}."
            )

        n = len(data)
        split_idx = int(n * train_pct / 100.0)

        if split_idx == 0:
            raise ValueError("Training set would be empty with the given percentage.")
        if split_idx >= n:
            raise ValueError("Test set would be empty with the given percentage.")

        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()

    else:
        raise ValueError(
            f"Unknown split_method '{split_method}'. Use 'date' or 'percentage'."
        )

    # Final safety checks
    if train_data.empty:
        raise ValueError("Training set is empty after splitting.")
    if test_data.empty:
        raise ValueError("Test set is empty after splitting.")

    # Assert no overlap
    assert train_data.index.max() < test_data.index.min(), (
        "Data leak detected: training set overlaps with test set."
    )

    return train_data, test_data
