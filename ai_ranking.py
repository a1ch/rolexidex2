"""
AI-powered ranking of luxury watch listings using LLM analysis.
Supports OpenAI and Anthropic. Returns quality, pricing, trend, and overall scores plus short summaries.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Optional: use openai or anthropic
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


def _get_openai_client(api_key: str) -> OpenAI | None:
    if OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _get_anthropic_client(api_key: str) -> Anthropic | None:
    if Anthropic is None:
        return None
    return Anthropic(api_key=api_key)


def _build_watch_summary(item: dict[str, Any], index: int) -> str:
    """One-line summary of a listing for the LLM."""
    title = (item.get("title") or "")[:120]
    price = item.get("priceString") or item.get("price") or "?"
    list_price = item.get("listPriceString") or ""
    cond = item.get("condition") or item.get("condition_normalized") or "?"
    seller = item.get("sellerName") or "?"
    feedback = item.get("sellerFeedbackPercent") or "?"
    sold = item.get("soldCount") or item.get("sold_count_num") or "?"
    listing_type = item.get("listingType") or "?"
    return (
        f"[{index}] Title: {title} | Price: {price} | List: {list_price} | "
        f"Condition: {cond} | Seller: {seller} ({feedback}) | Sold: {sold} | Type: {listing_type}"
    )


def _parse_ai_response(text: str, n_expected: int) -> list[dict[str, Any]]:
    """Parse LLM JSON response into list of score dicts."""
    # Try to find a JSON array in the response
    text = text.strip()
    # Strip markdown code block if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract array with regex
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []
    if not isinstance(data, list):
        return []
    # Normalize keys (LLM might use different names)
    result = []
    for i, row in enumerate(data[:n_expected]):
        if isinstance(row, dict):
            result.append({
                "quality_score": _safe_float(row.get("quality_score") or row.get("quality")),
                "pricing_score": _safe_float(row.get("pricing_score") or row.get("pricing")),
                "trend_score": _safe_float(row.get("trend_score") or row.get("trend")),
                "overall_ai_score": _safe_float(row.get("overall_ai_score") or row.get("overall_score") or row.get("overall")),
                "summary": _safe_str(row.get("summary") or row.get("one_line") or row.get("reason")),
            })
        else:
            result.append({
                "quality_score": 5.0,
                "pricing_score": 5.0,
                "trend_score": 5.0,
                "overall_ai_score": 5.0,
                "summary": "",
            })
    return result


def _safe_float(x: Any) -> float:
    if x is None:
        return 5.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 5.0


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()[:500]


def run_ai_ranking_openai(
    items: list[dict[str, Any]],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_items: int = 30,
) -> list[dict[str, Any]]:
    """Run AI ranking using OpenAI. Returns items with added ai_* fields."""
    if not _get_openai_client(api_key):
        raise RuntimeError("openai package not installed. pip install openai")
    return _run_ai_ranking(items, api_key, "openai", model, max_items)


def run_ai_ranking_anthropic(
    items: list[dict[str, Any]],
    api_key: str,
    model: str = "claude-3-5-haiku-20241022",
    max_items: int = 30,
) -> list[dict[str, Any]]:
    """Run AI ranking using Anthropic. Returns items with added ai_* fields."""
    if not _get_anthropic_client(api_key):
        raise RuntimeError("anthropic package not installed. pip install anthropic")
    return _run_ai_ranking(items, api_key, "anthropic", model, max_items)


def _run_ai_ranking(
    items: list[dict[str, Any]],
    api_key: str,
    provider: str,
    model: str,
    max_items: int,
) -> list[dict[str, Any]]:
    """Common logic: build prompt, call LLM, parse and merge scores."""
    subset = items[:max_items]
    if not subset:
        return items

    lines = [_build_watch_summary(item, i) for i, item in enumerate(subset)]
    listing_text = "\n".join(lines)

    prompt = f"""You are an expert in luxury watches and eBay marketplace deals. Analyze these eBay watch listings and score each one.

Listings (each line is one listing, [index] at start):
{listing_text}

For each listing index 0 to {len(subset)-1}, provide:
1. quality_score (1-10): perceived quality from title, condition, brand cues (e.g. Rolex, Omega).
2. pricing_score (1-10): how good the price is vs list price and typical market (discount, value).
3. trend_score (1-10): demand/popularity from seller feedback and sold count.
4. overall_ai_score (1-10): overall deal quality combining quality, pricing, and trend.
5. summary: one short sentence (max 15 words) on why it's a good or weak deal.

Return ONLY a JSON array of objects, one per listing in order. No other text.
Example format: [{{"quality_score": 7, "pricing_score": 8, "trend_score": 6, "overall_ai_score": 7, "summary": "Strong discount, trusted seller."}}, ...]
"""

    if provider == "openai":
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
    else:
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (msg.content[0].text if msg.content else "") or ""

    parsed = _parse_ai_response(text, len(subset))
    if len(parsed) != len(subset):
        # Pad with defaults if LLM returned fewer
        while len(parsed) < len(subset):
            parsed.append({
                "quality_score": 5.0,
                "pricing_score": 5.0,
                "trend_score": 5.0,
                "overall_ai_score": 5.0,
                "summary": "",
            })

    # Merge back into items (only subset gets AI fields)
    result = []
    for i, item in enumerate(items):
        out = dict(item)
        if i < len(parsed):
            p = parsed[i]
            out["ai_quality_score"] = p["quality_score"]
            out["ai_pricing_score"] = p["pricing_score"]
            out["ai_trend_score"] = p["trend_score"]
            out["ai_overall_score"] = p["overall_ai_score"]
            out["ai_summary"] = p["summary"]
        else:
            out["ai_quality_score"] = None
            out["ai_pricing_score"] = None
            out["ai_trend_score"] = None
            out["ai_overall_score"] = None
            out["ai_summary"] = None
        result.append(out)
    return result


def items_to_dataframe_ai(items: list[dict[str, Any]]) -> "pd.DataFrame":
    """Build DataFrame from AI-scored items, sorted by ai_overall_score (then deal_score)."""
    import pandas as pd
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    if "ai_overall_score" in df.columns:
        df = df.sort_values(
            "ai_overall_score",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    return df
