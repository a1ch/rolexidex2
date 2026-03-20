"""
Database layer for persisting watch listings.
Uses SQLite by default (local); set DATABASE_URL in secrets for Postgres (e.g. Streamlit Cloud).
Refresh 1–2x/day via "Refresh data" or an external scheduler.
"""
from __future__ import annotations

import math
import os
import socket
import urllib.parse
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text


def _sanitize(val: Any) -> Any:
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

# Default: SQLite. Set DATABASE_URL in secrets for Postgres (Streamlit Cloud).


def _get_secret_db_url() -> str:
    """DATABASE_URL from env (CI/scripts) first, then Streamlit secrets."""
    env_u = os.environ.get("DATABASE_URL", "").strip()
    if env_u:
        return env_u.replace("postgres://", "postgresql://", 1)
    try:
        import streamlit as st
        if "DATABASE_URL" in st.secrets:
            u = str(st.secrets.get("DATABASE_URL", "")).strip()
            if u:
                return u.replace("postgres://", "postgresql://", 1)
    except Exception:
        pass
    return ""


def is_remote_database_configured() -> bool:
    """True if DATABASE_URL is set (e.g. Supabase). Lets the app load saved data without scrape keys."""
    return bool(_get_secret_db_url().strip())


def get_engine():
    url = _get_secret_db_url() or "sqlite:///watches.db"
    if "sqlite" in url:
        return create_engine(url, connect_args={"check_same_thread": False})
    # Postgres (e.g. Supabase): ensure postgresql:// and optional SSL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    kwargs = {"pool_pre_ping": True}
    if "supabase" in url.lower():
        kwargs["connect_args"] = {"sslmode": "require"}

        # Streamlit Cloud sometimes ends up trying IPv6 routes for Supabase hosts.
        # If IPv6 isn't reachable, you may see: "Cannot assign requested address".
        # Force IPv4 resolution and also pass `hostaddr` to libpq when possible.
        try:
            parsed = urllib.parse.urlsplit(url)
            host = parsed.hostname
            if host and "supabase.co" in host.lower():
                ipv4_addrs = socket.getaddrinfo(host, None, socket.AF_INET)
                if ipv4_addrs:
                    ipv4 = ipv4_addrs[0][4][0]

                    # Only pass hostaddr: connect over IPv4 but KEEP the real hostname in the URL.
                    # Rewriting host to a raw IP breaks Supabase pooler (FATAL: Tenant or user not found).
                    connect_args = kwargs.get("connect_args") or {}
                    connect_args["hostaddr"] = ipv4
                    kwargs["connect_args"] = connect_args
        except Exception:
            pass
    return create_engine(url, **kwargs)


def init_db(engine=None):
    engine = engine or get_engine()
    is_sqlite = "sqlite" in (str(engine.url).lower())
    id_col = "id INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "id SERIAL PRIMARY KEY"
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS listings (
                {id_col},
                item_id TEXT,
                title TEXT,
                price REAL,
                price_string TEXT,
                list_price_string TEXT,
                condition TEXT,
                condition_normalized TEXT,
                seller_name TEXT,
                seller_feedback_percent TEXT,
                sold_count TEXT,
                sold_count_num INTEGER,
                listing_type TEXT,
                url TEXT,
                thumbnail TEXT,
                deal_score REAL,
                discount_pct REAL,
                price_value REAL,
                list_price_value REAL,
                condition_score REAL,
                seller_score REAL,
                trend_score REAL,
                scraped_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scrape_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_scraped_at TEXT,
                num_listings INTEGER
            )
        """))
        conn.commit()
    _migrate_listings_ai_columns(engine)


_AI_COLUMN_DEFS = (
    "ai_overall_score REAL",
    "ai_quality_score REAL",
    "ai_pricing_score REAL",
    "ai_trend_score REAL",
    "ai_authenticity_risk REAL",
    "ai_summary TEXT",
    "ai_authenticity_note TEXT",
)


def _migrate_listings_ai_columns(engine) -> None:
    """Add AI ranking columns to existing listings tables (idempotent)."""
    with engine.connect() as conn:
        for coldef in _AI_COLUMN_DEFS:
            try:
                conn.execute(text(f"ALTER TABLE listings ADD COLUMN {coldef}"))
                conn.commit()
            except Exception:
                conn.rollback()
                pass


def _row_to_item(row) -> dict[str, Any]:
    """Convert DB row to dict matching scraper output for app/AI."""
    out: dict[str, Any] = {
        "itemId": row[1],
        "title": row[2],
        "price": row[3],
        "priceString": row[4],
        "listPriceString": row[5],
        "condition": row[6],
        "condition_normalized": row[7],
        "sellerName": row[8],
        "sellerFeedbackPercent": row[9],
        "soldCount": row[10],
        "listingType": row[12],
        "url": row[13],
        "thumbnail": row[14],
        "deal_score": row[15],
        "discount_pct": row[16],
        "price_value": row[17],
        "list_price_value": row[18],
        "condition_score": row[19],
        "seller_score": row[20],
        "trend_score": row[21],
        "sold_count_num": row[11] or 0,
    }
    # AI columns (migration); short tuples = pre-migration DB
    if len(row) > 23:
        out["ai_overall_score"] = row[23]
        out["ai_quality_score"] = row[24]
        out["ai_pricing_score"] = row[25]
        out["ai_trend_score"] = row[26]
        out["ai_authenticity_risk"] = row[27]
        out["ai_summary"] = row[28]
        out["ai_authenticity_note"] = row[29]
    return out


def save_listings(engine, items: list[dict[str, Any]]) -> None:
    """Replace all listings with the new scrape and update metadata."""
    init_db(engine)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM listings"))
        conn.execute(text("DELETE FROM scrape_metadata"))
        now = datetime.now(timezone.utc).isoformat()
        for it in items:
            conn.execute(text("""
                INSERT INTO listings (
                    item_id, title, price, price_string, list_price_string,
                    condition, condition_normalized, seller_name, seller_feedback_percent,
                    sold_count, sold_count_num, listing_type, url, thumbnail,
                    deal_score, discount_pct, price_value, list_price_value,
                    condition_score, seller_score, trend_score, scraped_at,
                    ai_overall_score, ai_quality_score, ai_pricing_score, ai_trend_score,
                    ai_authenticity_risk, ai_summary, ai_authenticity_note
                ) VALUES (
                    :item_id, :title, :price, :price_string, :list_price_string,
                    :condition, :condition_normalized, :seller_name, :seller_feedback_percent,
                    :sold_count, :sold_count_num, :listing_type, :url, :thumbnail,
                    :deal_score, :discount_pct, :price_value, :list_price_value,
                    :condition_score, :seller_score, :trend_score, :scraped_at,
                    :ai_overall_score, :ai_quality_score, :ai_pricing_score, :ai_trend_score,
                    :ai_authenticity_risk, :ai_summary, :ai_authenticity_note
                )
            """), {
                "item_id": it.get("itemId"),
                "title": _sanitize(it.get("title")),
                "price": _sanitize(it.get("price")),
                "price_string": it.get("priceString"),
                "list_price_string": it.get("listPriceString"),
                "condition": it.get("condition"),
                "condition_normalized": it.get("condition_normalized"),
                "seller_name": it.get("sellerName"),
                "seller_feedback_percent": it.get("sellerFeedbackPercent"),
                "sold_count": it.get("soldCount"),
                "sold_count_num": it.get("sold_count_num", 0) or 0,
                "listing_type": it.get("listingType"),
                "url": it.get("url"),
                "thumbnail": it.get("thumbnail"),
                "deal_score": _sanitize(it.get("deal_score")),
                "discount_pct": _sanitize(it.get("discount_pct")),
                "price_value": _sanitize(it.get("price_value")),
                "list_price_value": _sanitize(it.get("list_price_value")),
                "condition_score": _sanitize(it.get("condition_score")),
                "seller_score": _sanitize(it.get("seller_score")),
                "trend_score": _sanitize(it.get("trend_score")),
                "scraped_at": now,
                "ai_overall_score": _sanitize(it.get("ai_overall_score")),
                "ai_quality_score": _sanitize(it.get("ai_quality_score")),
                "ai_pricing_score": _sanitize(it.get("ai_pricing_score")),
                "ai_trend_score": _sanitize(it.get("ai_trend_score")),
                "ai_authenticity_risk": _sanitize(it.get("ai_authenticity_risk")),
                "ai_summary": (it.get("ai_summary") or "")[:8000] if it.get("ai_summary") else None,
                "ai_authenticity_note": (it.get("ai_authenticity_note") or "")[:4000] if it.get("ai_authenticity_note") else None,
            })
        conn.execute(text("""
            INSERT INTO scrape_metadata (id, last_scraped_at, num_listings) VALUES (1, :at, :n)
        """), {"at": now, "n": len(items)})
        conn.commit()


def get_listings(engine) -> list[dict[str, Any]]:
    """Load all listings from DB, ordered by deal_score desc."""
    init_db(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, item_id, title, price, price_string, list_price_string,
                   condition, condition_normalized, seller_name, seller_feedback_percent,
                   sold_count, sold_count_num, listing_type, url, thumbnail,
                   deal_score, discount_pct, price_value, list_price_value,
                   condition_score, seller_score, trend_score, scraped_at,
                   ai_overall_score, ai_quality_score, ai_pricing_score, ai_trend_score,
                   ai_authenticity_risk, ai_summary, ai_authenticity_note
            FROM listings ORDER BY deal_score DESC
        """)).fetchall()
    return [_row_to_item(r) for r in rows]


def get_last_scraped_at(engine) -> str | None:
    """When we last refreshed the DB."""
    init_db(engine)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT last_scraped_at FROM scrape_metadata WHERE id = 1")).fetchone()
    return row[0] if row else None


def get_db_stats(engine) -> dict[str, Any]:
    """Row count and last refresh without loading all listings."""
    init_db(engine)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM listings")).scalar_one()
        row = conn.execute(
            text("SELECT last_scraped_at, num_listings FROM scrape_metadata WHERE id = 1")
        ).fetchone()
    return {
        "listing_count": int(n or 0),
        "last_scraped_at": row[0] if row else None,
        "meta_num_listings": int(row[1] or 0) if row else 0,
    }
