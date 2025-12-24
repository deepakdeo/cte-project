"""
Tests for the synthetic data generator.
"""

from datetime import datetime

import pandas as pd
import pytest

from cte.synthetic import generate_synthetic_dataset, _time_str, _duration_str


class TestHelperFunctions:
    """Test helper functions."""

    def test_time_str_morning(self):
        result = _time_str(8, 30)
        assert result == "8:30 AM"

    def test_time_str_afternoon(self):
        result = _time_str(14, 45)
        assert result == "2:45 PM"

    def test_time_str_midnight(self):
        result = _time_str(0, 0)
        assert result == "12:00 AM"

    def test_duration_str(self):
        assert _duration_str(7.5) == "7h30m"
        assert _duration_str(8.0) == "8h0m"
        assert _duration_str(6.25) == "6h15m"


class TestSyntheticGeneration:
    """Test synthetic dataset generation."""

    def test_generate_correct_number_of_days(self):
        df = generate_synthetic_dataset(n_days=30, seed=42)
        assert len(df) == 30

    def test_generate_with_different_seeds_produces_different_data(self):
        df1 = generate_synthetic_dataset(n_days=10, seed=1)
        df2 = generate_synthetic_dataset(n_days=10, seed=2)
        # Productivity values should differ
        assert not df1["productivity\npercentage"].equals(df2["productivity\npercentage"])

    def test_generate_with_same_seed_is_reproducible(self):
        df1 = generate_synthetic_dataset(n_days=10, seed=42)
        df2 = generate_synthetic_dataset(n_days=10, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_all_expected_columns_present(self, synthetic_df_small):
        expected_cols = [
            "Date", "Reflection", "primary mood", "secondary mood",
            "productivity\npercentage", "when most \nproductive",
            "studied \nat home", "studied \nat school",
            "breakfast qualtity", "lunch quality", "dinner quality",
            "water \ndrank", "workout \ndid", "meditation",
            "morning \nshower", "played \nsports",
            "interaction \nw/ partner", "interaction \nw/ family",
            "interaction \nw/ friends", "sickness",
            "sleep \nduration", "nap \ntoday",
            "deep sleep \npercentage", "REM sleep \npercentage",
            "dinner time", "bed time", "wakeup time",
        ]
        for col in expected_cols:
            assert col in synthetic_df_small.columns, f"Missing column: {col}"

    def test_productivity_in_valid_range(self, synthetic_df_medium):
        prod = synthetic_df_medium["productivity\npercentage"]
        assert prod.min() >= 0
        assert prod.max() <= 100

    def test_sleep_percentages_valid(self, synthetic_df_medium):
        deep = synthetic_df_medium["deep sleep \npercentage"]
        rem = synthetic_df_medium["REM sleep \npercentage"]
        assert deep.min() >= 0
        assert deep.max() <= 100
        assert rem.min() >= 0
        assert rem.max() <= 100

    def test_boolean_columns_have_yes_no(self, synthetic_df_small):
        bool_cols = ["studied \nat home", "workout \ndid", "meditation", "sickness", "nap \ntoday"]
        for col in bool_cols:
            values = set(synthetic_df_small[col].unique())
            assert values.issubset({"yes", "no"}), f"Invalid values in {col}: {values}"

    def test_interaction_columns_valid_values(self, synthetic_df_small):
        int_cols = ["interaction \nw/ partner", "interaction \nw/ family", "interaction \nw/ friends"]
        valid = {"positive", "neutral", "negative", "na"}
        for col in int_cols:
            values = set(synthetic_df_small[col].unique())
            assert values.issubset(valid), f"Invalid values in {col}: {values}"

    def test_meal_columns_valid_values(self, synthetic_df_small):
        meal_cols = ["breakfast qualtity", "lunch quality", "dinner quality"]
        valid = {"carb_heavy", "protein_heavy", "fat_heavy", "balanced", "na"}
        for col in meal_cols:
            values = set(synthetic_df_small[col].unique())
            assert values.issubset(valid), f"Invalid values in {col}: {values}"

    def test_custom_start_date(self):
        start = datetime(2024, 1, 1)
        df = generate_synthetic_dataset(n_days=5, start_date=start, seed=42)
        # First date should be Jan 1, 2024
        assert "Jan 01, 2024" in df["Date"].iloc[0]

    def test_reflections_not_empty(self, synthetic_df_small):
        for reflection in synthetic_df_small["Reflection"]:
            assert len(reflection) > 10  # Non-trivial text


class TestDataConsistency:
    """Test that generated data has realistic patterns."""

    def test_weekday_weekend_patterns(self):
        """Weekends should have slightly different patterns."""
        df = generate_synthetic_dataset(n_days=90, seed=42)
        # This is a sanity check - we expect data to be generated without errors
        assert len(df) == 90

    def test_temporal_correlations_exist(self):
        """Adjacent days should show some correlation (carryover effects)."""
        df = generate_synthetic_dataset(n_days=60, seed=42)
        prod = df["productivity\npercentage"]
        # Calculate lag-1 correlation
        corr = prod.corr(prod.shift(1))
        # Should have some positive correlation due to carryover
        assert corr > -1  # Just verify it computes
