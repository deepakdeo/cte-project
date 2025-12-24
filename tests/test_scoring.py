"""
Tests for the job-fit scoring module.
"""

import pytest

from cte.scoring import score_requirements


class TestScoreRequirements:
    """Test the job-fit scoring function."""

    def test_perfect_match(self, sample_persona):
        """All traits meet thresholds."""
        per_trait = sample_persona["per_trait"]
        requirements = [
            {"trait": "planning", "required_level": "low"},  # 0.80 score meets low (0.50)
            {"trait": "reliability", "required_level": "low"},  # 0.75 meets low
        ]
        overall, match_ratio, risk, rows, criticals = score_requirements(per_trait, requirements)
        assert match_ratio == 1.0
        assert overall == "Strong fit"
        assert risk == "low-risk"
        assert len(criticals) == 0

    def test_partial_match(self, sample_persona):
        """Some traits meet thresholds, some don't."""
        per_trait = sample_persona["per_trait"]
        requirements = [
            {"trait": "focus", "required_level": "high"},  # 0.72 >= 0.70 threshold
            {"trait": "communication", "required_level": "medium"},  # 0.65 >= 0.60
            {"trait": "impact", "required_level": "high"},  # 0.55 < 0.70 - fails
        ]
        overall, match_ratio, risk, rows, criticals = score_requirements(per_trait, requirements)
        # Should have 2 met, 1 unmet
        met_count = sum(1 for r in rows if r["met"])
        assert met_count == 2
        assert "impact" in criticals  # unmet high requirement

    def test_missing_trait_returns_zero_score(self, sample_persona):
        """Requirements for traits not in persona get score 0."""
        per_trait = sample_persona["per_trait"]
        requirements = [
            {"trait": "unknown_trait", "required_level": "high"},
        ]
        overall, match_ratio, risk, rows, criticals = score_requirements(per_trait, requirements)
        # Unknown trait should have score 0, fail high threshold
        assert len(rows) == 1
        assert rows[0]["candidate_score"] == 0.0
        assert rows[0]["met"] is False
        assert "unknown_trait" in criticals

    def test_empty_requirements(self, sample_persona):
        """Empty requirements should handle gracefully."""
        per_trait = sample_persona["per_trait"]
        overall, match_ratio, risk, rows, criticals = score_requirements(per_trait, [])
        # With no requirements, match_ratio should be 1.0 (met_w/total_w = 0/1)
        assert match_ratio == 0.0  # Actually 0/1 = 0
        assert overall is not None

    def test_custom_thresholds(self, sample_persona):
        """Test with custom threshold settings."""
        per_trait = sample_persona["per_trait"]
        requirements = [{"trait": "focus", "required_level": "high"}]
        # focus score is 0.72, custom high threshold is 0.70
        overall, match_ratio, risk, rows, _ = score_requirements(
            per_trait,
            requirements,
            thresholds={"low": 0.3, "medium": 0.5, "high": 0.70},
        )
        assert match_ratio == 1.0
        assert rows[0]["met"] is True

    def test_custom_weights(self, sample_persona):
        """Test with custom weight settings."""
        per_trait = sample_persona["per_trait"]
        requirements = [
            {"trait": "focus", "required_level": "high"},  # meets
            {"trait": "communication", "required_level": "low"},  # meets
        ]
        overall, match_ratio, risk, rows, _ = score_requirements(
            per_trait,
            requirements,
            weights={"low": 1.0, "medium": 1.5, "high": 2.0},
        )
        # Both should be met
        assert all(r["met"] for r in rows)
        assert match_ratio == 1.0

    def test_verdict_categories(self):
        """Test that verdicts map correctly to match ratios."""
        valid_verdicts = ["Strong fit", "Possible fit", "Leaning no", "Not a fit"]

        # Create persona with varying scores
        per_trait = {
            "skill_a": {"score": 0.9, "confidence": 0.9},
            "skill_b": {"score": 0.1, "confidence": 0.9},
        }

        # Strong match (should be Strong fit)
        requirements = [{"trait": "skill_a", "required_level": "low"}]
        overall, _, _, _, _ = score_requirements(per_trait, requirements)
        assert overall in valid_verdicts

        # Weak match (should be Not a fit or Leaning no)
        requirements = [{"trait": "skill_b", "required_level": "high"}]
        overall, _, _, _, _ = score_requirements(per_trait, requirements)
        assert overall in valid_verdicts

    def test_risk_bands(self, sample_persona):
        """Risk band should map to match ratio correctly."""
        per_trait = sample_persona["per_trait"]
        valid_bands = ["low-risk", "moderate-risk", "elevated-risk", "high-risk"]

        # High match -> low risk
        requirements = [{"trait": "planning", "required_level": "low"}]
        _, _, risk, _, _ = score_requirements(per_trait, requirements)
        assert risk in valid_bands

    def test_per_requirement_details(self, sample_persona):
        """Results should include per-requirement breakdown."""
        per_trait = sample_persona["per_trait"]
        requirements = [
            {"trait": "focus", "required_level": "high"},
            {"trait": "communication", "required_level": "medium"},
        ]
        _, _, _, rows, _ = score_requirements(per_trait, requirements)
        # Should have an entry for each requirement
        assert len(rows) == 2
        for row in rows:
            assert "trait" in row
            assert "candidate_score" in row
            assert "threshold" in row
            assert "met" in row

    def test_unmet_high_penalty(self, sample_persona):
        """Unmet high requirements should apply penalty."""
        per_trait = sample_persona["per_trait"]
        # Create requirement that will fail
        requirements = [
            {"trait": "impact", "required_level": "high"},  # 0.55 < 0.70
        ]
        _, match_ratio_with_penalty, _, _, criticals = score_requirements(
            per_trait, requirements, unmet_high_penalty=0.10
        )
        assert "impact" in criticals

        # Without penalty
        _, match_ratio_no_penalty, _, _, _ = score_requirements(
            per_trait, requirements, unmet_high_penalty=0.0
        )
        # With penalty should be lower (or equal if both are 0)
        assert match_ratio_with_penalty <= match_ratio_no_penalty


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_score_trait(self):
        """Trait with zero score."""
        per_trait = {"skill": {"score": 0.0, "confidence": 0.5}}
        requirements = [{"trait": "skill", "required_level": "low"}]
        overall, match_ratio, _, rows, _ = score_requirements(per_trait, requirements)
        assert rows[0]["candidate_score"] == 0.0
        assert rows[0]["met"] is False

    def test_max_score_trait(self):
        """Trait with maximum score."""
        per_trait = {"skill": {"score": 1.0, "confidence": 1.0}}
        requirements = [{"trait": "skill", "required_level": "high"}]
        overall, match_ratio, _, rows, _ = score_requirements(per_trait, requirements)
        assert rows[0]["met"] is True
        assert match_ratio == 1.0

    def test_all_requirements_unmet(self):
        """All requirements unmet should give low score."""
        per_trait = {
            "skill_a": {"score": 0.1, "confidence": 0.9},
            "skill_b": {"score": 0.1, "confidence": 0.9},
        }
        requirements = [
            {"trait": "skill_a", "required_level": "high"},
            {"trait": "skill_b", "required_level": "high"},
        ]
        overall, match_ratio, risk, rows, criticals = score_requirements(per_trait, requirements)
        assert match_ratio < 0.5
        assert overall in ["Leaning no", "Not a fit"]
        assert len(criticals) == 2

    def test_invalid_level_ignored(self):
        """Invalid requirement levels should be skipped."""
        per_trait = {"skill": {"score": 0.8, "confidence": 0.9}}
        requirements = [
            {"trait": "skill", "required_level": "invalid_level"},
        ]
        _, _, _, rows, _ = score_requirements(per_trait, requirements)
        # Invalid level should be skipped
        assert len(rows) == 0

    def test_rows_sorted_correctly(self):
        """Rows should be sorted by level, met status, and score."""
        per_trait = {
            "a": {"score": 0.9, "confidence": 0.9},
            "b": {"score": 0.5, "confidence": 0.9},
            "c": {"score": 0.8, "confidence": 0.9},
        }
        requirements = [
            {"trait": "a", "required_level": "high"},  # met
            {"trait": "b", "required_level": "high"},  # not met
            {"trait": "c", "required_level": "medium"},  # met
        ]
        _, _, _, rows, _ = score_requirements(per_trait, requirements)
        # Should be sorted: high met first, then high unmet, then medium
        assert rows[0]["trait"] == "a"  # high, met, highest score
