#!/usr/bin/env python3
"""
One-shot: scrape eBay and save listings to your database (Supabase/Postgres or SQLite).

Data source (one of):
  - Apify: set APIFY_API_KEY
  - eBay API: set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET

Required: DATABASE_URL (env or .streamlit/secrets.toml).

Optional env:
  SEARCH_QUERIES="Rolex\\nOmega"  (newline-separated, or use defaults)
  MAX_PRODUCTS=50  (MAX_PAGES only used for Apify)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_local_secrets() -> None:
    p = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not p.exists():
        return
    try:
        import tomllib
        with open(p, "rb") as f:
            data = tomllib.load(f)
        for k, v in data.items():
            if isinstance(v, str) and v and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        print("Could not load .streamlit/secrets.toml:", e, file=sys.stderr)


def main() -> int:
    _load_local_secrets()
    api = os.environ.get("APIFY_API_KEY", "").strip()
    ebay_id = os.environ.get("EBAY_CLIENT_ID", "").strip()
    ebay_secret = os.environ.get("EBAY_CLIENT_SECRET", "").strip()
    db = os.environ.get("DATABASE_URL", "").strip()

    use_apify = bool(api)
    use_ebay_api = bool(ebay_id and ebay_secret)

    if not use_apify and not use_ebay_api:
        print(
            "Missing credentials: set APIFY_API_KEY (Apify) or EBAY_CLIENT_ID + EBAY_CLIENT_SECRET (eBay API) in env or .streamlit/secrets.toml",
            file=sys.stderr,
        )
        return 1
    if not db:
        print("Missing DATABASE_URL for Postgres/Supabase (env or secrets.toml)", file=sys.stderr)
        return 1

    from scraper import run_ebay_scrape, items_to_dataframe, DEFAULT_WATCH_QUERIES
    from database import get_engine, save_listings, init_db

    queries = os.environ.get("SEARCH_QUERIES", "").strip()
    if queries:
        qlist = [x.strip() for x in queries.replace("\\n", "\n").split("\n") if x.strip()]
    else:
        qlist = list(DEFAULT_WATCH_QUERIES)
    max_p = int(os.environ.get("MAX_PRODUCTS", "50"))
    max_pages = int(os.environ.get("MAX_PAGES", "3"))

    if use_apify:
        print("Scraping eBay via Apify…", qlist)
        items = run_ebay_scrape(
            api_token=api,
            search_queries=qlist,
            max_products_per_search=max_p,
            max_search_pages=max_pages,
            listing_type="all",
        )
    else:
        from ebay_api import run_ebay_api_search
        print("Fetching from eBay API…", qlist)
        items = run_ebay_api_search(
            client_id=ebay_id,
            client_secret=ebay_secret,
            search_queries=qlist,
            limit_per_query=min(max_p, 200),
            max_total=max_p * len(qlist),
        )

    if not items:
        print("No items returned.", file=sys.stderr)
        return 2
    df = items_to_dataframe(items)
    engine = get_engine()
    init_db(engine)
    save_listings(engine, df.to_dict("records"))
    print(f"Saved {len(df)} listings to database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
