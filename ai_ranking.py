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

# Cheapest usable defaults (override via function args if you prefer).
# OpenAI: gpt-5-nano is very cheap when available; fall back to gpt-4o-mini.
OPENAI_MODEL_DEFAULT = "gpt-5-nano"
OPENAI_MODEL_FALLBACKS = ("gpt-4o-mini",)

# Anthropic: Haiku-class is the cheap tier; fall back if an ID 404s.
ANTHROPIC_MODEL_DEFAULT = "claude-haiku-4-5-20251001"
ANTHROPIC_MODEL_FALLBACKS = (
    "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022",
)

# Smaller batches = valid JSON + full scores (one huge response often truncates → all 5s).
AI_BATCH_SIZE = 7


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
            # Accept numeric strings; authenticity may be named differently
            result.append({
                "quality_score": _safe_float(row.get("quality_score") or row.get("quality")),
                "pricing_score": _safe_float(row.get("pricing_score") or row.get("pricing")),
                "trend_score": _safe_float(row.get("trend_score") or row.get("trend")),
                "overall_ai_score": _safe_float(row.get("overall_ai_score") or row.get("overall_score") or row.get("overall")),
                "summary": _safe_str(row.get("summary") or row.get("one_line") or row.get("reason")),
                "authenticity_risk": _safe_float(
                    row.get("authenticity_risk")
                    or row.get("authenticity")
                    or row.get("fake_risk")
                    or 5.0
                ),
                "authenticity_note": _safe_str(row.get("authenticity_note") or row.get("auth_note") or ""),
            })
        else:
            result.append({
                "quality_score": 5.0,
                "pricing_score": 5.0,
                "trend_score": 5.0,
                "overall_ai_score": 5.0,
                "summary": "",
                "authenticity_risk": 5.0,
                "authenticity_note": "",
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
    model: str = OPENAI_MODEL_DEFAULT,
    max_items: int = 30,
) -> list[dict[str, Any]]:
    """Run AI ranking using OpenAI. Returns items with added ai_* fields."""
    if not _get_openai_client(api_key):
        raise RuntimeError("openai package not installed. pip install openai")
    return _run_ai_ranking(items, api_key, "openai", model, max_items)


def run_ai_ranking_anthropic(
    items: list[dict[str, Any]],
    api_key: str,
    model: str = ANTHROPIC_MODEL_DEFAULT,
    max_items: int = 30,
) -> list[dict[str, Any]]:
    """Run AI ranking using Anthropic. Returns items with added ai_* fields."""
    if not _get_anthropic_client(api_key):
        raise RuntimeError("anthropic package not installed. pip install anthropic")
    return _run_ai_ranking(items, api_key, "anthropic", model, max_items)


def _anthropic_message_text(msg: Any) -> str:
    """Join all text blocks (avoid empty answer if first block isn't text)."""
    parts: list[str] = []
    for block in getattr(msg, "content", None) or []:
        txt = getattr(block, "text", None)
        if txt:
            parts.append(txt)
    return "".join(parts)


def _build_batch_prompt(batch: list[dict[str, Any]], offset: int) -> str:
    lines = [_build_watch_summary(item, offset + i) for i, item in enumerate(batch)]
    listing_text = "\n".join(lines)
    n = len(batch)
    return f"""You are an expert in luxury watches and eBay marketplace deals. Score each listing below.

Listings (each line is one row; [index] is the global index):
{listing_text}

There are EXACTLY {n} listings in this batch. Return EXACTLY {n} JSON objects in a single array, in the SAME ORDER.

For EACH listing provide integers/floats 1-10 (use varied scores when listings differ; do NOT default every field to 5):
1. quality_score (1-10): quality from title, condition, brand cues.
2. pricing_score (1-10): price vs list/discount and value.
3. trend_score (1-10): seller feedback / demand signals.
4. overall_ai_score (1-10): overall deal quality.
5. summary: max 15 words.
6. authenticity_risk (1-10): 10 = likely genuine, 1 = strong replica/red-flag signals.
7. authenticity_note: short phrase.

Return ONLY valid JSON: one array, no markdown, no commentary. Example schema:
[{{"quality_score":7,"pricing_score":8,"trend_score":6,"overall_ai_score":7,"summary":"...","authenticity_risk":8,"authenticity_note":"..."}}]
"""


def _default_pad_row() -> dict[str, Any]:
    return {
        "quality_score": 5.0,
        "pricing_score": 5.0,
        "trend_score": 5.0,
        "overall_ai_score": 5.0,
        "summary": "(AI parse failed for this row)",
        "authenticity_risk": 5.0,
        "authenticity_note": "",
    }


def _call_openai_batch(client: Any, models: list[str], prompt: str) -> str:
    last: Exception | None = None
    for m in models:
        if not m:
            continue
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.55,
                max_tokens=4096,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            err = str(e).lower()
            if "model" in err or "404" in str(e) or "does not exist" in err:
                last = e
                continue
            raise
    if last:
        raise last
    return ""


def _call_anthropic_batch(client: Any, models: list[str], prompt: str) -> str:
    last_err: Exception | None = None
    for m in models:
        if not m:
            continue
        try:
            msg = client.messages.create(
                model=m,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = _anthropic_message_text(msg).strip()
            if text:
                return text
        except Exception as e:
            err = str(e).lower()
            if "404" in str(e) or "not_found" in err:
                last_err = e
                continue
            raise
    if last_err:
        raise last_err
    return ""


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

    if provider == "openai":
        client = OpenAI(api_key=api_key)
        models_try: list[str] = []
        for m in (model, *OPENAI_MODEL_FALLBACKS):
            if m and m not in models_try:
                models_try.append(m)
    else:
        client = Anthropic(api_key=api_key)
        models_try = []
        for m in (model, *ANTHROPIC_MODEL_FALLBACKS):
            if m and m not in models_try:
                models_try.append(m)

    parsed_all: list[dict[str, Any]] = []
    for start in range(0, len(subset), AI_BATCH_SIZE):
        batch = subset[start : start + AI_BATCH_SIZE]
        prompt = _build_batch_prompt(batch, start)
        if provider == "openai":
            text = _call_openai_batch(client, models_try, prompt)
        else:
            text = _call_anthropic_batch(client, models_try, prompt)

        parsed = _parse_ai_response(text, len(batch))
        if len(parsed) != len(batch):
            while len(parsed) < len(batch):
                parsed.append(_default_pad_row())
            parsed = parsed[: len(batch)]
        parsed_all.extend(parsed)

    parsed = parsed_all
    if len(parsed) != len(subset):
        while len(parsed) < len(subset):
            parsed.append(_default_pad_row())
        parsed = parsed[: len(subset)]

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
            out["ai_authenticity_risk"] = p.get("authenticity_risk", 5.0)
            out["ai_authenticity_note"] = p.get("authenticity_note", "")
        else:
            out["ai_quality_score"] = None
            out["ai_pricing_score"] = None
            out["ai_trend_score"] = None
            out["ai_overall_score"] = None
            out["ai_summary"] = None
            out["ai_authenticity_risk"] = None
            out["ai_authenticity_note"] = None
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
