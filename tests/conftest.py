"""
Shared test fixtures for CTE test suite.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from cte.synthetic import generate_synthetic_dataset


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def synthetic_df_small():
    """Generate a small synthetic dataset for testing."""
    return generate_synthetic_dataset(n_days=10, seed=123)


@pytest.fixture
def synthetic_df_medium():
    """Generate a medium synthetic dataset for testing."""
    return generate_synthetic_dataset(n_days=30, seed=456)


@pytest.fixture
def sample_persona():
    """Sample persona for scoring tests."""
    return {
        "per_trait": {
            "focus": {"score": 0.72, "confidence": 0.8},
            "communication": {"score": 0.65, "confidence": 0.75},
            "teamwork": {"score": 0.58, "confidence": 0.7},
            "planning": {"score": 0.80, "confidence": 0.85},
            "adaptability": {"score": 0.62, "confidence": 0.6},
            "learning_mindset": {"score": 0.70, "confidence": 0.8},
            "impact": {"score": 0.55, "confidence": 0.65},
            "reliability": {"score": 0.75, "confidence": 0.9},
            "resilience": {"score": 0.60, "confidence": 0.7},
            "independence": {"score": 0.68, "confidence": 0.75},
        }
    }


@pytest.fixture
def sample_requirements():
    """Sample JD requirements for scoring tests."""
    return {
        "focus": "high",
        "communication": "medium",
        "teamwork": "high",
        "planning": "low",
        "adaptability": "medium",
    }
