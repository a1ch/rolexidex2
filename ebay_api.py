"""
eBay Browse API as a data source using your eBay API keys (OAuth client credentials).
Returns listings in the same format as the Apify scraper so the rest of the app works unchanged.
"""
from __future__ import annotations

import base64
import os
import urllib.parse
import urllib.request
import urllib.error
from typing import Any

# Production; use api.sandbox.ebay.com for sandbox
IDENTITY_URL_PROD = "https://api.ebay.com/identity/v1/oauth2/token"
IDENTITY_URL_SANDBOX = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
BROWSE_URL_PROD = "https://api.ebay.com/buy/browse/v1/item_summary/search"
BROWSE_URL_SANDBOX = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"
SCOPE = "https://api.ebay.com/oauth/api_scope"  # Browse API public access

DEFAULT_MARKETPLACE_ID = "EBAY_US"


def _use_sandbox_from_env() -> bool:
    v = os.environ.get("EBAY_USE_SANDBOX", "").strip().lower()
    return v in {"1", "true", "yes", "y"}


def _identity_url() -> str:
    return IDENTITY_URL_SANDBOX if _use_sandbox_from_env() else IDENTITY_URL_PROD


def _browse_url() -> str:
    return BROWSE_URL_SANDBOX if _use_sandbox_from_env() else BROWSE_URL_PROD


def get_oauth_token(client_id: str, client_secret: str) -> str:
    """Get Application access token via client credentials grant."""
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": SCOPE,
    }).encode()
    req = urllib.request.Request(
        _identity_url(),
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {credentials}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = __import__("json").load(resp)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            raw = e.read()
            if isinstance(raw, bytes):
                body = raw.decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        # Don't leak secrets; include only response body.
        raise RuntimeError(
            f"eBay OAuth token request failed: HTTP {e.code} {e.reason}.{(' Response: ' + body[:500]) if body else ''} "
            f"Check that your Client ID/Secret pair is correct and that sandbox/production endpoints match your keys."
        ) from e
    return out["access_token"]


def _item_summary_to_listing(s: dict[str, Any]) -> dict[str, Any]:
    """Map Browse API itemSummary to our app's listing shape (Apify-compatible)."""
    item_id = s.get("itemId", "")
    title = s.get("title", "")
    condition = s.get("condition", "") or s.get("conditionId", "")
    price_obj = s.get("price", {}) or {}
    value = price_obj.get("value")
    currency = price_obj.get("currency", "USD")
    price_string = f"{currency} {value}" if value is not None else ""
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
    # List price / original price (seller discount)
    listing_price = s.get("unitPrice", {}) or {}
    list_value = listing_price.get("value")
    list_price_string = f"{listing_price.get('currency', 'USD')} {list_value}" if list_value else ""
    image_obj = s.get("image", {}) or {}
    thumbnail = image_obj.get("imageUrl", "")
    item_web_url = s.get("itemWebUrl", "") or f"https://www.ebay.com/itm/{item_id}"
    buying_options = s.get("buyingOptions") or []
    listing_type = "Buy It Now" if "FIXED_PRICE" in buying_options else "Auction" if "AUCTION" in buying_options else ""

    return {
        "itemId": item_id,
        "title": title,
        "price": value,
        "priceString": price_string,
        "listPriceString": list_price_string or price_string,
        "condition": condition,
        "sellerName": "",
        "sellerFeedbackPercent": "",
        "soldCount": "",
        "listingType": listing_type,
        "url": item_web_url,
        "thumbnail": thumbnail,
    }


def run_ebay_api_search(
    client_id: str,
    client_secret: str,
    search_queries: list[str],
    limit_per_query: int = 200,
    max_total: int = 2000,
) -> list[dict[str, Any]]:
    """
    Search eBay via Browse API; returns list of listings in Apify-compatible format.
    """
    token = get_oauth_token(client_id, client_secret)
    all_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    marketplace_id = os.environ.get("EBAY_MARKETPLACE_ID", DEFAULT_MARKETPLACE_ID).strip() or DEFAULT_MARKETPLACE_ID
    browse_url = _browse_url()

    for q in search_queries:
        if len(all_items) >= max_total:
            break
        offset = 0
        limit = min(limit_per_query, 200)  # API often caps at 200 per request
        while offset < limit_per_query and len(all_items) < max_total:
            params = urllib.parse.urlencode({
                "q": q[:100],
                "limit": min(200, limit_per_query - offset),
                "offset": offset,
            })
            url = f"{browse_url}?{params}"
            headers = {
                "Authorization": f"Bearer {token}",
                # Required for marketplaces outside of US; safe to pass for US too.
                "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
            }
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = __import__("json").load(resp)
            except Exception as e:
                # Try to include response body for easier debugging (e.g. invalid token / wrong keys).
                body = ""
                if hasattr(e, "read"):
                    try:
                        raw = e.read()
                        if isinstance(raw, bytes):
                            body = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        body = ""
                details = f" eBay API error body: {body[:500]}" if body else ""
                raise RuntimeError(f"eBay API search failed: {e}.{details}") from e

            summaries = data.get("itemSummaries") or []
            if not summaries:
                break
            for s in summaries:
                item_id = s.get("itemId")
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    all_items.append(_item_summary_to_listing(s))
            offset += len(summaries)
            if len(summaries) < 200:
                break

    return all_items
