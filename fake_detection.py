"""
Fake/replica detection for luxury watch listings.
Rule-based: title keywords, price heuristics, seller trust.
Output: fake_risk_score (0–100, higher = more suspicious), fake_risk_level, fake_reasons.
"""
from __future__ import annotations

import re
from typing import Any


# Phrases that often indicate replica / homage / non-genuine (case-insensitive)
FAKE_KEYWORDS = [
    "replica",
    "replicas",
    "fake",
    "counterfeit",
    "copy",
    "homage",
    "inspired by",
    "style of",
    "aftermarket dial",
    "aftermarket bezel",
    "franken",
    "frankenwatch",
    "custom build",
    "modded",
    "china made",
    "no box no papers",
]

# Strong indicators: likely explicit replica listing
STRONG_FAKE_KEYWORDS = ["replica", "fake", "counterfeit", "copy watch", "replica watch"]


def _parse_feedback_percent(s: str | None) -> float:
    if not s:
        return 0.0
    n = re.sub(r"[^\d.]", "", str(s))
    try:
        return float(n) if n else 0.0
    except ValueError:
        return 0.0


def _parse_feedback_count(s: str | None) -> int:
    if not s:
        return 0
    n = re.sub(r"[^\d]", "", str(s))
    try:
        return int(n) if n else 0
    except ValueError:
        return 0


def compute_fake_risk(item: dict[str, Any]) -> dict[str, Any]:
    """
    Add fake_risk_score (0–100), fake_risk_level (low/medium/high), fake_reasons (list).
    Higher score = more suspicious. 0 = no red flags.
    """
    reasons: list[str] = []
    score = 0.0
    title = (item.get("title") or "").lower()
    price = item.get("price_value") or item.get("price")
    if price is None and item.get("priceString"):
        try:
            price = float(re.sub(r"[^\d.]", "", str(item["priceString"])))
        except (TypeError, ValueError):
            price = None
    feedback_pct = _parse_feedback_percent(item.get("sellerFeedbackPercent"))
    feedback_count = _parse_feedback_count(item.get("sellerFeedbackCount"))

    # 1) Explicit replica/fake keywords → high risk
    for kw in STRONG_FAKE_KEYWORDS:
        if kw in title:
            score = min(100, score + 80)
            reasons.append(f"Title contains '{kw}'")
            break
    for kw in FAKE_KEYWORDS:
        if kw in title and score < 80:
            score = min(100, score + 35)
            if not any(kw in r for r in reasons):
                reasons.append(f"Title: '{kw}'")

    # 2) Suspiciously low price for "luxury" (rough heuristic: Rolex/Omega typically $1k+ for real)
    if price is not None and price < 500 and any(b in title for b in ["rolex", "omega", "cartier", "patek"]):
        score = min(100, score + 25)
        reasons.append("Very low price for luxury brand")
    elif price is not None and price < 200 and ("luxury" in title or "automatic" in title):
        score = min(100, score + 15)
        reasons.append("Very low price for luxury/automatic claim")

    # 3) Seller: very low feedback count + high-ticket
    if feedback_count < 50 and price is not None and price > 2000:
        score = min(100, score + 20)
        reasons.append("Low seller feedback count for high-price item")
    if feedback_pct < 90 and feedback_pct > 0:
        score = min(100, score + 10)
        reasons.append("Seller feedback below 90%")

    # 4) "No papers" / "no box" can be legit but often used with fakes
    if "no papers" in title or "no box" in title:
        score = min(100, score + 5)
        reasons.append("No box/papers (verify authenticity)")

    score = round(min(100, score), 1)
    if score >= 60:
        level = "high"
    elif score >= 25:
        level = "medium"
    else:
        level = "low"

    return {
        "fake_risk_score": score,
        "fake_risk_level": level,
        "fake_reasons": reasons[:5],
    }
