"""
Apify eBay scraper and deal-scoring logic for luxury/fancy watches.
Uses automation-lab/ebay-scraper actor.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from apify_client import ApifyClient


# Default search queries for "fancy" / luxury watches
DEFAULT_WATCH_QUERIES = [
    "Rolex",
    "Omega luxury watch",
    "Tag Heuer",
    "Breitling",
    "Cartier watch",
    "luxury automatic watch",
]

# Condition quality weight (higher = better for deal score)
CONDITION_SCORE = {
    "new": 1.0,
    "open_box": 0.85,
    "refurbished": 0.75,
    "used": 0.5,
    "for_parts": 0.2,
}


def _normalize_condition(raw: str | None) -> str:
    if not raw or not isinstance(raw, str):
        return "used"
    raw_lower = raw.lower()
    if "new" in raw_lower and "like" not in raw_lower:
        return "new"
    if "open box" in raw_lower or "open-box" in raw_lower:
        return "open_box"
    if "refurbish" in raw_lower:
        return "refurbished"
    if "part" in raw_lower or "for part" in raw_lower:
        return "for_parts"
    return "used"


def _parse_price_string(s: str | None) -> float | None:
    if not s:
        return None
    numbers = re.sub(r"[^\d.]", "", str(s))
    try:
        return float(numbers) if numbers else None
    except ValueError:
        return None


def _parse_sold_count(s: str | None) -> int:
    if not s:
        return 0
    numbers = re.sub(r"[^\d]", "", str(s))
    try:
        return int(numbers) if numbers else 0
    except ValueError:
        return 0


def _parse_feedback_percent(s: str | None) -> float:
    if not s:
        return 0.0
    numbers = re.sub(r"[^\d.]", "", str(s))
    try:
        return float(numbers) if numbers else 0.0
    except ValueError:
        return 0.0


def run_ebay_scrape(
    api_token: str,
    search_queries: list[str] | None = None,
    max_products_per_search: int = 50,
    max_search_pages: int = 3,
    listing_type: str = "all",
    min_price: int | None = None,
    max_price: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run Apify eBay scraper and return list of product items.
    Actor: automation-lab/ebay-scraper
    """
    queries = search_queries or DEFAULT_WATCH_QUERIES
    client = ApifyClient(api_token)
    actor_client = client.actor("automation-lab/ebay-scraper")
    run_input = {
        "searchQueries": queries,
        "maxProductsPerSearch": max_products_per_search,
        "maxSearchPages": max_search_pages,
        "sort": "best_match",
        "listingType": listing_type,
    }
    if min_price is not None:
        run_input["minPrice"] = min_price
    if max_price is not None:
        run_input["maxPrice"] = max_price

    run_result = actor_client.call(run_input=run_input)
    if run_result is None:
        return []

    dataset_client = client.dataset(run_result["defaultDatasetId"])
    items = list(dataset_client.iterate_items())
    return items


def score_deal(item: dict[str, Any]) -> dict[str, Any]:
    """
    Compute deal score and components for one listing.
    Returns item dict with added keys: condition_normalized, price_value,
    list_price_value, discount_pct, condition_score, seller_score, trend_score, deal_score.
    """
    price = item.get("price")
    if price is None:
        price = _parse_price_string(item.get("priceString"))
    list_price = _parse_price_string(item.get("listPriceString"))
    if list_price is None or list_price <= 0:
        list_price = price

    condition_norm = _normalize_condition(item.get("condition"))
    condition_score = CONDITION_SCORE.get(condition_norm, 0.5)

    feedback_pct = _parse_feedback_percent(item.get("sellerFeedbackPercent"))
    seller_score = min(1.0, feedback_pct / 100.0) if feedback_pct else 0.5

    sold = _parse_sold_count(item.get("soldCount"))
    trend_score = min(1.0, (sold / 100.0) * 0.5 + 0.5) if sold else 0.5  # more sold = slight boost

    discount_pct = 0.0
    if list_price and list_price > 0 and price is not None and price < list_price:
        discount_pct = (1 - price / list_price) * 100
    price_score = min(1.0, (discount_pct / 50.0) * 0.5 + 0.5) if discount_pct else 0.5  # discount boost

    # Combined deal score: weight price (discount), condition, seller, trend
    deal_score = (
        price_score * 0.35
        + condition_score * 0.30
        + seller_score * 0.20
        + trend_score * 0.15
    )
    deal_score = round(deal_score * 100, 1)

    out = dict(item)
    out["condition_normalized"] = condition_norm
    out["price_value"] = price
    out["list_price_value"] = list_price
    out["discount_pct"] = round(discount_pct, 1)
    out["condition_score"] = condition_score
    out["seller_score"] = seller_score
    out["trend_score"] = trend_score
    out["deal_score"] = deal_score
    out["sold_count_num"] = sold
    return out


def items_to_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Score all items and return a DataFrame sorted by deal_score descending."""
    if not items:
        return pd.DataFrame()
    scored = [score_deal(i) for i in items]
    df = pd.DataFrame(scored)
    df = df.sort_values("deal_score", ascending=False).reset_index(drop=True)
    return df
