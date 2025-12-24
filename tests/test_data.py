"""
Tests for the data cleaning pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from cte.data import (
    _coerce_yes_no,
    _coerce_percent_or_number,
    _parse_duration_to_hours,
    _parse_time_to_minutes,
    _norm_token,
    _map_interaction,
    clean_csv,
    RENAME_MAP,
)


class TestTokenNormalization:
    """Test token normalization."""

    def test_norm_token_lowercase(self):
        assert _norm_token("YES") == "yes"
        assert _norm_token("No") == "no"

    def test_norm_token_strips_whitespace(self):
        assert _norm_token("  yes  ") == "yes"

    def test_norm_token_handles_none(self):
        assert _norm_token(None) is None

    def test_norm_token_handles_nan(self):
        assert _norm_token(np.nan) is None


class TestBooleanCoercion:
    """Test yes/no coercion."""

    def test_yes_variants(self):
        s = pd.Series(["yes", "Yes", "YES", "y", "Y", "true", "True", "1"])
        result = _coerce_yes_no(s)
        assert result.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    def test_no_variants(self):
        s = pd.Series(["no", "No", "NO", "n", "N", "false", "False", "0"])
        result = _coerce_yes_no(s)
        assert result.tolist() == [0, 0, 0, 0, 0, 0, 0, 0]

    def test_handles_none(self):
        s = pd.Series(["yes", None, "no"])
        result = _coerce_yes_no(s)
        assert result.iloc[0] == 1
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 0


class TestPercentCoercion:
    """Test percentage/number coercion."""

    def test_integer_string(self):
        s = pd.Series(["75", "100", "0"])
        result = _coerce_percent_or_number(s)
        assert result.tolist() == [75.0, 100.0, 0.0]

    def test_with_percent_sign(self):
        s = pd.Series(["75%", "50 %", " 25% "])
        result = _coerce_percent_or_number(s)
        assert result.tolist() == [75.0, 50.0, 25.0]

    def test_float_values(self):
        s = pd.Series(["75.5", "33.33"])
        result = _coerce_percent_or_number(s)
        assert result.tolist() == [75.5, 33.33]

    def test_handles_none(self):
        s = pd.Series([50, None])
        result = _coerce_percent_or_number(s)
        assert result.iloc[0] == 50.0
        assert np.isnan(result.iloc[1])


class TestDurationParsing:
    """Test duration string parsing."""

    def test_hours_and_minutes(self):
        assert _parse_duration_to_hours("7h38m") == pytest.approx(7 + 38/60, rel=1e-3)
        assert _parse_duration_to_hours("8h0m") == 8.0

    def test_colon_format(self):
        assert _parse_duration_to_hours("7:30") == 7.5
        assert _parse_duration_to_hours("6:15") == 6.25

    def test_only_hours(self):
        assert _parse_duration_to_hours("8h") == 8.0

    def test_only_minutes(self):
        assert _parse_duration_to_hours("90m") == 1.5

    def test_handles_na(self):
        assert np.isnan(_parse_duration_to_hours("na"))
        assert np.isnan(_parse_duration_to_hours(None))
        assert np.isnan(_parse_duration_to_hours(""))


class TestTimeParsing:
    """Test time string parsing to minutes after midnight."""

    def test_am_times(self):
        s = pd.Series(["6:30 AM", "8:00 AM", "11:45 AM"])
        result = _parse_time_to_minutes(s)
        assert result.tolist() == [6*60+30, 8*60, 11*60+45]

    def test_pm_times(self):
        s = pd.Series(["1:00 PM", "8:30 PM", "11:59 PM"])
        result = _parse_time_to_minutes(s)
        assert result.tolist() == [13*60, 20*60+30, 23*60+59]

    def test_noon_and_midnight(self):
        s = pd.Series(["12:00 PM", "12:00 AM"])
        result = _parse_time_to_minutes(s)
        assert result.tolist() == [12*60, 0]


class TestInteractionMapping:
    """Test social interaction encoding."""

    def test_positive_negative_neutral(self):
        s = pd.Series(["positive", "negative", "neutral"])
        result = _map_interaction(s, "test")
        assert result["test_score"].tolist() == [1.0, -1.0, 0.0]
        assert result["test_no_interaction"].tolist() == [0, 0, 0]

    def test_na_handling(self):
        s = pd.Series(["na", "NA", "positive"])
        result = _map_interaction(s, "test")
        # na should have score 0 and no_interaction flag 1
        assert result["test_score"].iloc[0] == 0.0
        assert result["test_no_interaction"].iloc[0] == 1
        assert result["test_no_interaction"].iloc[2] == 0


class TestFullPipeline:
    """Test the full cleaning pipeline."""

    def test_clean_csv_with_synthetic_data(self, synthetic_df_small, temp_dir):
        # Save synthetic data to CSV
        in_path = temp_dir / "test_input.csv"
        out_path = temp_dir / "test_output.parquet"
        synthetic_df_small.to_csv(in_path, index=False)

        # Run cleaning pipeline
        result = clean_csv(in_path, out_path)

        # Check output exists
        assert out_path.exists()

        # Check key columns are present
        assert "date" in result.columns
        assert "productivity_pct" in result.columns
        assert "sleep_duration_h" in result.columns

        # Check data types
        assert result["productivity_pct"].dtype in [float, np.float64]

    def test_clean_csv_preserves_row_count(self, synthetic_df_medium, temp_dir):
        in_path = temp_dir / "test_input.csv"
        out_path = temp_dir / "test_output.parquet"
        synthetic_df_medium.to_csv(in_path, index=False)

        result = clean_csv(in_path, out_path)
        assert len(result) == len(synthetic_df_medium)

    def test_rename_map_completeness(self):
        """Verify RENAME_MAP covers all expected raw columns."""
        expected_raw = [
            "Date", "Reflection", "primary mood", "secondary mood",
            "productivity\npercentage", "when most \nproductive",
        ]
        for col in expected_raw:
            assert col in RENAME_MAP, f"Missing mapping for: {col}"
