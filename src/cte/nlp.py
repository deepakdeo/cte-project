# src/cte/nlp.py
"""
NLP utilities for sentiment analysis and emotional trait extraction.

This module provides multiple sentiment analysis approaches:
1. analyze_sentiment() - OpenAI API (original, unchanged)
2. analyze_sentiment_local() - Local ML models (VADER + transformers)
3. analyze_sentiment_hybrid() - Smart choice between local/API
4. extract_emotional_signals() - 5 emotional traits from text
5. analyze_sentiment_trends() - Burnout detection from time series

All functions are backward compatible with existing code.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Original imports (keep for backward compatibility)
from .openai_util import chat_json, parse_json_maybe

# New imports for local sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# =============================================================================
# ORIGINAL FUNCTION (UNCHANGED - BACKWARD COMPATIBLE)
# =============================================================================

# System prompt: make it deterministic and schema-bound
SA_SYSTEM = """You are a sentiment rater. Output ONLY valid JSON.
You must classify short reflective text as:
- label: one of ["positive","neutral","negative"]
- score: float in [0,1] where 0=very negative, 0.5=neutral, 1=very positive
Add a short rationale string (<=100 chars)."""

def _sa_schema() -> Dict[str, Any]:
    # for clarity; not strictly used by the model, but helpful reference
    return {"label": "positive|neutral|negative", "score": 0.0, "rationale": ""}

def analyze_sentiment(text: str, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    """
    ORIGINAL FUNCTION - Uses OpenAI API for sentiment analysis.
    
    Returns dict: {"label": str, "score": float, "rationale": str}
    If the text is empty/whitespace, returns neutral 0.5.
    
    This function is UNCHANGED for backward compatibility.
    """
    t = (text or "").strip()
    if not t:
        return {"label": "neutral", "score": 0.5, "rationale": "empty text"}
    
    user_prompt = f"""Rate the following reflection:
---TEXT START---
{t}
---TEXT END---
Respond as JSON with keys: label, score, rationale."""
    
    raw = chat_json(SA_SYSTEM, user_prompt, model=model, max_tokens=200)
    data = parse_json_maybe(raw) or {}
    
    label = str(data.get("label", "neutral")).lower()
    if label not in ("positive", "neutral", "negative"):
        label = "neutral"
    
    try:
        score = float(data.get("score", 0.5))
    except Exception:
        score = 0.5
    score = max(0.0, min(1.0, score))
    
    rationale = str(data.get("rationale", ""))[:120]
    
    return {"label": label, "score": score, "rationale": rationale}


# =============================================================================
# NEW: LOCAL SENTIMENT ANALYSIS (FREE, FAST, ACCURATE)
# =============================================================================

class LocalSentimentAnalyzer:
    """
    Local sentiment analyzer using VADER and/or Transformers.
    Free, fast, offline sentiment analysis.
    """
    
    def __init__(self, method: str = "auto"):
        """
        Initialize local sentiment analyzer.
        
        Args:
            method: "vader", "transformers", or "auto" (tries transformers first)
        """
        self.method = method
        self.analyzer = None
        
        if method == "auto":
            if TRANSFORMERS_AVAILABLE:
                self.method = "transformers"
            elif VADER_AVAILABLE:
                self.method = "vader"
            else:
                raise ImportError("No local sentiment library available")
        
        if self.method == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers not installed")
            # Use cached model to avoid re-loading
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        
        elif self.method == "vader":
            if not VADER_AVAILABLE:
                raise ImportError("vaderSentiment not installed")
            self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using local models.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with keys: label, score, positive, negative, neutral, compound
        """
        if not text or not text.strip():
            return {
                'label': 'neutral',
                'score': 0.5,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        if self.method == "transformers":
            return self._analyze_transformers(text)
        else:
            return self._analyze_vader(text)
    
    def _analyze_transformers(self, text: str) -> Dict[str, Any]:
        """Analyze using transformer model (more accurate)"""
        # Truncate to 512 tokens max
        text = text[:2000]
        
        result = self.analyzer(text)[0]
        label = result['label']  # POSITIVE or NEGATIVE
        conf = result['score']  # Confidence
        
        if label == 'POSITIVE':
            return {
                'label': 'positive',
                'score': (0.5 + conf / 2),  # 0.5 to 1.0
                'positive': conf,
                'negative': 1 - conf,
                'neutral': 0.0,
                'compound': conf
            }
        else:
            return {
                'label': 'negative',
                'score': (0.5 - conf / 2),  # 0.0 to 0.5
                'positive': 1 - conf,
                'negative': conf,
                'neutral': 0.0,
                'compound': -conf
            }
    
    def _analyze_vader(self, text: str) -> Dict[str, Any]:
        """Analyze using VADER (faster, rule-based)"""
        scores = self.analyzer.polarity_scores(text)
        
        # Determine label
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
            score = 0.5 + (compound * 0.5)  # Map 0.05-1.0 to 0.525-1.0
        elif compound <= -0.05:
            label = 'negative'
            score = 0.5 + (compound * 0.5)  # Map -1.0--0.05 to 0.0-0.475
        else:
            label = 'neutral'
            score = 0.5
        
        return {
            'label': label,
            'score': max(0.0, min(1.0, score)),
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }


# Global instance (lazy loaded)
_local_analyzer = None

def analyze_sentiment_local(text: str, method: str = "auto") -> Dict[str, Any]:
    """
    Analyze sentiment using local ML models (FREE, FAST, ACCURATE).
    
    This is often MORE accurate than OpenAI for sentiment because it uses
    models specifically trained on 67k+ sentiment examples.
    
    Args:
        text: Text to analyze
        method: "vader", "transformers", or "auto" (tries transformers first)
    
    Returns:
        Dict with keys:
        - label: "positive", "negative", or "neutral"
        - score: float 0-1 (0=very negative, 0.5=neutral, 1=very positive)
        - positive: positive score component
        - negative: negative score component
        - neutral: neutral score component
        - compound: compound score (-1 to 1)
    
    Example:
        >>> analyze_sentiment_local("Today was great!")
        {'label': 'positive', 'score': 0.89, 'compound': 0.78, ...}
    """
    global _local_analyzer
    
    if _local_analyzer is None:
        _local_analyzer = LocalSentimentAnalyzer(method=method)
    
    return _local_analyzer.analyze(text)


# =============================================================================
# NEW: HYBRID SENTIMENT (SMART CHOICE BETWEEN LOCAL/API)
# =============================================================================

def analyze_sentiment_hybrid(
    text: str, 
    prefer_local: bool = True,
    complexity_threshold: int = 200,
    model: str = "gpt-4.1-mini"
) -> Dict[str, Any]:
    """
    Smart hybrid sentiment analysis: uses local for simple cases, API for complex.
    
    Strategy:
    - Short/simple text → local (free, fast, accurate)
    - Long/complex text → API (detailed rationale)
    - Always returns consistent format
    
    Args:
        text: Text to analyze
        prefer_local: If True, tries local first (saves API costs)
        complexity_threshold: Character count to switch to API
        model: OpenAI model to use if needed
    
    Returns:
        Dict with keys: label, score, rationale (if API used)
    
    Example:
        >>> analyze_sentiment_hybrid("Good day!")  # Uses local
        {'label': 'positive', 'score': 0.85}
        
        >>> analyze_sentiment_hybrid("Complex situation...")  # Uses API if long
        {'label': 'positive', 'score': 0.75, 'rationale': '...'}
    """
    t = (text or "").strip()
    if not t:
        return {"label": "neutral", "score": 0.5, "rationale": "empty text"}
    
    # Decide which method to use
    use_local = prefer_local and len(t) < complexity_threshold
    
    if use_local and (VADER_AVAILABLE or TRANSFORMERS_AVAILABLE):
        # Use local analysis
        result = analyze_sentiment_local(t)
        # Convert to API-compatible format
        return {
            "label": result['label'],
            "score": result['score'],
            "rationale": f"Local analysis: {result['label']} (compound: {result.get('compound', 0):.2f})"
        }
    else:
        # Use API (original function)
        return analyze_sentiment(t, model=model)


# =============================================================================
# NEW: EMOTIONAL TRAIT EXTRACTION (5 TRAITS FROM TEXT + SENTIMENT)
# =============================================================================

def extract_emotional_signals(text: str, sentiment: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract 5 emotional trait signals from text and sentiment.
    
    Traits extracted (0.0 to 1.0):
    - optimism: Positive outlook (from sentiment)
    - emotional_stability: Consistency of emotions
    - confidence: Self-assurance and achievement language
    - stress_tolerance: Performance under pressure
    - resilience: Bouncing back from difficulties
    
    Args:
        text: Text to analyze
        sentiment: Pre-computed sentiment dict (optional, will compute if None)
    
    Returns:
        Dict mapping trait names to scores (0.0 to 1.0)
    
    Example:
        >>> extract_emotional_signals("Stressed but finished everything!")
        {'optimism': 0.60, 'confidence': 0.70, 'stress_tolerance': 0.75,
         'resilience': 0.70, 'emotional_stability': 0.55}
    """
    if not text or not text.strip():
        return {
            'optimism': 0.5,
            'emotional_stability': 0.5,
            'confidence': 0.5,
            'stress_tolerance': 0.5,
            'resilience': 0.5,
        }
    
    # Get sentiment if not provided
    if sentiment is None:
        sentiment = analyze_sentiment_local(text)
    
    compound = sentiment.get('compound', 0.0)
    text_lower = text.lower()
    
    # Initialize signals
    signals = {
        'optimism': 0.5,
        'emotional_stability': 0.5,
        'confidence': 0.5,
        'stress_tolerance': 0.5,
        'resilience': 0.5,
    }
    
    # OPTIMISM: directly from positive sentiment
    signals['optimism'] = 0.5 + (compound * 0.3)
    
    # EMOTIONAL STABILITY: inverse of sentiment volatility
    signals['emotional_stability'] = (
        0.3 * sentiment.get('neutral', 0.5) +
        0.4 * (1 - sentiment.get('negative', 0)) +
        0.3 * sentiment.get('positive', 0)
    )
    
    # CONFIDENCE: positive words + achievement language
    confidence_keywords = [
        'confident', 'proud', 'accomplished', 'succeeded', 
        'nailed', 'crushed', 'achieved', 'excellent', 'great job',
        'did well', 'strong', 'capable'
    ]
    confidence_boost = sum(1 for kw in confidence_keywords if kw in text_lower) * 0.1
    signals['confidence'] = min(0.9, 0.5 + (compound * 0.2) + confidence_boost)
    
    # STRESS TOLERANCE: negative sentiment but still productive
    stress_keywords = [
        'stressed', 'overwhelmed', 'pressure', 'difficult', 
        'challenging', 'tough', 'hard', 'intense', 'demanding'
    ]
    productive_keywords = [
        'finished', 'completed', 'done', 'delivered', 
        'productive', 'focused', 'accomplished', 'achieved'
    ]
    
    has_stress = any(kw in text_lower for kw in stress_keywords)
    was_productive = any(kw in text_lower for kw in productive_keywords)
    
    if has_stress and was_productive:
        signals['stress_tolerance'] = 0.75  # High - performed despite stress
    elif has_stress and not was_productive:
        signals['stress_tolerance'] = 0.35  # Low - stress impacted performance
    else:
        signals['stress_tolerance'] = 0.55  # Neutral
    
    # RESILIENCE: bouncing back from negative
    resilience_keywords = [
        'despite', 'recovered', 'pushed through', 'overcame', 
        'persevered', 'kept going', 'bounced back', 'got back',
        'moved on', 'learned from'
    ]
    if any(kw in text_lower for kw in resilience_keywords):
        signals['resilience'] = 0.8
    elif compound < -0.3:  # Very negative
        signals['resilience'] = 0.3
    else:
        signals['resilience'] = 0.5 + (compound * 0.2)
    
    # Clamp all values to [0, 1]
    for key in signals:
        signals[key] = max(0.0, min(1.0, signals[key]))
    
    return signals


# =============================================================================
# NEW: SENTIMENT TREND ANALYSIS (BURNOUT DETECTION)
# =============================================================================

def analyze_sentiment_trends(sentiment_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze sentiment trends over time for burnout detection.
    
    Args:
        sentiment_data: List of dicts with keys: 'ts' (timestamp), 'sentiment_compound'
                       Or a pandas DataFrame with these columns
    
    Returns:
        Dict with:
        - avg_sentiment: Average sentiment score
        - trend: "improving", "declining", or "stable"
        - volatility: Standard deviation of sentiment
        - positive_days: Count of positive sentiment days
        - negative_days: Count of negative sentiment days
        - neutral_days: Count of neutral sentiment days
        - burnout_risk: "low", "medium", or "high"
        - total_days: Total number of data points
    
    Example:
        >>> data = [{'ts': '2024-01-01', 'sentiment_compound': 0.5}, ...]
        >>> analyze_sentiment_trends(data)
        {'avg_sentiment': 0.42, 'trend': 'improving', 'burnout_risk': 'low', ...}
    """
    # Handle both list of dicts and DataFrame
    try:
        import pandas as pd
        if isinstance(sentiment_data, pd.DataFrame):
            df = sentiment_data
        else:
            df = pd.DataFrame(sentiment_data)
    except ImportError:
        # Fallback without pandas
        if not sentiment_data:
            return _empty_trend_result()
        
        compounds = [d.get('sentiment_compound', 0) for d in sentiment_data]
        avg_sentiment = sum(compounds) / len(compounds) if compounds else 0
        
        return {
            'avg_sentiment': avg_sentiment,
            'trend': 'insufficient_data',
            'volatility': 0.0,
            'positive_days': sum(1 for c in compounds if c >= 0.05),
            'negative_days': sum(1 for c in compounds if c <= -0.05),
            'neutral_days': sum(1 for c in compounds if -0.05 < c < 0.05),
            'burnout_risk': 'low' if avg_sentiment > 0 else 'medium',
            'total_days': len(compounds)
        }
    
    if df.empty or 'sentiment_compound' not in df.columns or len(df) < 2:
        return _empty_trend_result()
    
    # Calculate statistics
    avg_sentiment = df['sentiment_compound'].mean()
    volatility = df['sentiment_compound'].std()
    
    positive_days = (df['sentiment_compound'] >= 0.05).sum()
    negative_days = (df['sentiment_compound'] <= -0.05).sum()
    neutral_days = len(df) - positive_days - negative_days
    
    # Determine trend (last 7 days vs previous 7 days)
    if len(df) >= 14:
        recent = df.tail(7)['sentiment_compound'].mean()
        previous = df.iloc[-14:-7]['sentiment_compound'].mean()
        
        if recent > previous + 0.1:
            trend = 'improving'
        elif recent < previous - 0.1:
            trend = 'declining'
        else:
            trend = 'stable'
    else:
        trend = 'insufficient_data'
    
    # Burnout risk detection
    recent_week = df.tail(7)
    burnout_risk = 'low'
    
    if len(recent_week) >= 5:
        recent_avg = recent_week['sentiment_compound'].mean()
        negative_streak = (recent_week['sentiment_compound'] < -0.1).sum()
        
        if recent_avg < -0.3 or negative_streak >= 5:
            burnout_risk = 'high'
        elif recent_avg < -0.1 or negative_streak >= 3:
            burnout_risk = 'medium'
    
    return {
        'avg_sentiment': float(avg_sentiment),
        'trend': trend,
        'volatility': float(volatility),
        'positive_days': int(positive_days),
        'negative_days': int(negative_days),
        'neutral_days': int(neutral_days),
        'burnout_risk': burnout_risk,
        'total_days': len(df)
    }


def _empty_trend_result() -> Dict[str, Any]:
    """Return empty/default trend result"""
    return {
        'avg_sentiment': 0.0,
        'trend': 'insufficient_data',
        'volatility': 0.0,
        'positive_days': 0,
        'negative_days': 0,
        'neutral_days': 0,
        'burnout_risk': 'low',
        'total_days': 0
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_methods() -> List[str]:
    """Return list of available sentiment analysis methods."""
    methods = ['api']  # Always have API
    if VADER_AVAILABLE:
        methods.append('vader')
    if TRANSFORMERS_AVAILABLE:
        methods.append('transformers')
    return methods


def is_local_available() -> bool:
    """Check if any local sentiment method is available."""
    return VADER_AVAILABLE or TRANSFORMERS_AVAILABLE


# =============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# =============================================================================

__all__ = [
    # Original (unchanged)
    'analyze_sentiment',
    
    # New local analysis
    'analyze_sentiment_local',
    'analyze_sentiment_hybrid',
    
    # New trait extraction
    'extract_emotional_signals',
    'analyze_sentiment_trends',
    
    # Utilities
    'get_available_methods',
    'is_local_available',
]
