"""
Luxury Watch Deals — Streamlit website.
Scrape eBay via Apify, rank by rule-based and AI analysis (quality, pricing, trends).
Deploy to GitHub + Streamlit Community Cloud.
"""
import os
import re
import streamlit as st
import pandas as pd

from scraper import (
    run_ebay_scrape,
    items_to_dataframe,
    DEFAULT_WATCH_QUERIES,
)
from ebay_api import run_ebay_api_search
from ai_ranking import (
    run_ai_ranking_openai,
    run_ai_ranking_anthropic,
    items_to_dataframe_ai,
)
from database import (
    get_engine,
    init_db,
    get_listings,
    save_listings,
    get_last_scraped_at,
    get_db_stats,
    is_remote_database_configured,
)
from datetime import datetime, timezone, timedelta

st.set_page_config(
    page_title="Luxury Watch Deals | AI Rankings",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Secrets: Streamlit Cloud + .streamlit/secrets.toml (root keys). Env vars as fallback.
def _get_secret(key: str) -> str:
    """Get API key from Streamlit secrets or environment. Returns empty if not set."""
    try:
        if key in st.secrets:
            val = st.secrets[key]
            return str(val).strip() if val else ""
    except (AttributeError, TypeError, FileNotFoundError):
        pass
    return os.environ.get(key, "").strip()


def _ai_error_message(e: Exception) -> str:
    """User-friendly message when AI (OpenAI/Anthropic) fails."""
    s = str(e).lower()
    if "429" in s or "quota" in s or "insufficient_quota" in s or "rate limit" in s:
        return "That provider’s quota/limits are hit. Switch to **Anthropic** in the sidebar (or add credits to OpenAI). Rule-based ranking still works without AI."
    return str(e)


# Custom CSS
st.markdown("""
<style>
    .deal-score { font-size: 1.4rem; font-weight: 700; color: #0e8c80; }
    .ai-score { font-size: 1.2rem; font-weight: 600; color: #7c4dff; }
    .metric-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .watch-title { font-weight: 600; color: #eaeaea; }
    a { color: #4fc3f7; }
    div[data-testid="stExpander"] { border: 1px solid #333; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


def _ensure_fake_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add fake_risk_score/level/reasons if missing (e.g. loaded from DB)."""
    if "fake_risk_score" in df.columns:
        return df
    from fake_detection import compute_fake_risk
    rows = df.to_dict("records")
    for r in rows:
        risk = compute_fake_risk(r)
        r["fake_risk_score"] = risk["fake_risk_score"]
        r["fake_risk_level"] = risk["fake_risk_level"]
        r["fake_reasons"] = risk["fake_reasons"]
    return pd.DataFrame(rows)


def _render_rule_based(df: pd.DataFrame) -> None:
    """Rule-based ranking view."""
    st.subheader("Rankings by deal score (rule-based)")
    st.caption("Formula: 35% price/discount + 30% condition + 20% seller feedback + 15% popularity. **Fake risk** = rule-based replica/fake detection.")
    display_cols = [
        "deal_score", "fake_risk_score", "fake_risk_level", "title", "priceString", "listPriceString", "discount_pct",
        "condition_normalized", "sellerFeedbackPercent", "soldCount", "listingType", "url",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].head(1000),
        column_config={
            "deal_score": st.column_config.NumberColumn("Deal score", format="%.1f"),
            "fake_risk_score": st.column_config.NumberColumn("Fake risk", format="%.0f"),
            "fake_risk_level": st.column_config.TextColumn("Risk level"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "priceString": st.column_config.TextColumn("Price"),
            "listPriceString": st.column_config.TextColumn("List price"),
            "discount_pct": st.column_config.NumberColumn("Discount %", format="%.1f"),
            "condition_normalized": st.column_config.TextColumn("Condition"),
            "sellerFeedbackPercent": st.column_config.TextColumn("Seller %"),
            "soldCount": st.column_config.TextColumn("Sold"),
            "listingType": st.column_config.TextColumn("Type"),
            "url": st.column_config.LinkColumn("View auction", display_text="→ Open"),
        },
        use_container_width=True,
        hide_index=True,
    )


def _render_ai_ranking(df: pd.DataFrame) -> None:
    """AI ranking view (requires ai_* columns)."""
    if "ai_overall_score" not in df.columns or df["ai_overall_score"].isna().all():
        st.info("Run **AI analysis** in the sidebar (and set an OpenAI or Anthropic API key) to see AI rankings.")
        return
    df_ai = df.dropna(subset=["ai_overall_score"]).copy()
    if df_ai.empty:
        st.warning("No AI-scored items yet.")
        return
    st.subheader("Rankings by AI analysis")
    st.caption("Scores from LLM: quality, pricing, trends, and overall deal assessment.")
    display_cols = [
        "ai_overall_score", "ai_quality_score", "ai_pricing_score", "ai_trend_score",
        "ai_authenticity_risk", "title", "priceString", "listPriceString", "ai_summary",
        "condition_normalized", "sellerFeedbackPercent", "url",
    ]
    available = [c for c in display_cols if c in df_ai.columns]
    st.dataframe(
        df_ai[available].head(1000),
        column_config={
            "ai_overall_score": st.column_config.NumberColumn("AI overall", format="%.1f"),
            "ai_authenticity_risk": st.column_config.NumberColumn("AI authenticity (10=likely genuine)", format="%.1f"),
            "ai_quality_score": st.column_config.NumberColumn("Quality", format="%.1f"),
            "ai_pricing_score": st.column_config.NumberColumn("Pricing", format="%.1f"),
            "ai_trend_score": st.column_config.NumberColumn("Trend", format="%.1f"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "priceString": st.column_config.TextColumn("Price"),
            "listPriceString": st.column_config.TextColumn("List price"),
            "ai_summary": st.column_config.TextColumn("AI summary", width="medium"),
            "condition_normalized": st.column_config.TextColumn("Condition"),
            "sellerFeedbackPercent": st.column_config.TextColumn("Seller %"),
            "url": st.column_config.LinkColumn("View auction", display_text="→ Open"),
        },
        use_container_width=True,
        hide_index=True,
    )


def _render_datasheets(df: pd.DataFrame, use_ai: bool) -> None:
    """Expandable watch cards; use_ai=True shows AI scores when present."""
    n_cards = min(20, len(df))
    for i in range(n_cards):
        row = df.iloc[i]
        title_short = (row.get("title") or "")[:70]
        price_short = str(row.get("priceString") or "").strip()
        if use_ai and "ai_overall_score" in row and pd.notna(row.get("ai_overall_score")):
            label = f"#{i+1} — {title_short}… | AI: {row['ai_overall_score']:.1f}" + (f" | {price_short}" if price_short else "")
        else:
            label = f"#{i+1} — {title_short}… | Score: {row['deal_score']}" + (f" | {price_short}" if price_short else "")
        with st.expander(label):
            url = row.get("url") or ""
            # Fallback: construct a usable eBay link when the dataset doesn't include `url`.
            if not url:
                item_id = row.get("itemId") or ""
                if item_id:
                    url = f"https://www.ebay.com/itm/{item_id}"
            title_full = (row.get("title") or "").strip()
            if url and title_full:
                st.markdown(f"[**{title_full}**]({url})")
            col_a, col_b = st.columns(2)
            with col_a:
                if row.get("thumbnail"):
                    st.image(row["thumbnail"], width=200)
                if use_ai and "ai_overall_score" in row and pd.notna(row.get("ai_overall_score")):
                    st.metric("AI overall", f"{row['ai_overall_score']:.1f}")
                    st.write("**Quality:**", f"{row.get('ai_quality_score', '—'):.1f}" if pd.notna(row.get("ai_quality_score")) else "—")
                    st.write("**Pricing:**", f"{row.get('ai_pricing_score', '—'):.1f}" if pd.notna(row.get("ai_pricing_score")) else "—")
                    st.write("**Trend:**", f"{row.get('ai_trend_score', '—'):.1f}" if pd.notna(row.get("ai_trend_score")) else "—")
                    if row.get("ai_summary"):
                        st.caption(f"*{row['ai_summary']}*")
                    if pd.notna(row.get("ai_authenticity_risk")):
                        st.write("**AI authenticity:**", f"{row['ai_authenticity_risk']:.1f}/10", "—", row.get("ai_authenticity_note") or "")
                else:
                    st.metric("Deal score", f"{row['deal_score']}")
                st.write("**Price:**", row.get("priceString", "—"))
                st.write("**List price:**", row.get("listPriceString", "—"))
                st.write("**Discount:**", f"{row.get('discount_pct', 0)}%")
            with col_b:
                st.write("**Condition:**", row.get("condition_normalized", "—"))
                st.write("**Seller:**", row.get("sellerName", "—"))
                st.write("**Feedback:**", row.get("sellerFeedbackPercent", "—"))
                st.write("**Sold:**", row.get("soldCount", "—"))
                st.write("**Type:**", row.get("listingType", "—"))
                risk = row.get("fake_risk_score")
                level = row.get("fake_risk_level", "")
                if pd.notna(risk):
                    color = "red" if level == "high" else "orange" if level == "medium" else "green"
                    st.markdown(f"**Likelihood of fake:** :{color}[{risk:.0f}% — {level}]")
                    reasons = row.get("fake_reasons") or []
                    if reasons:
                        for r in reasons[:3]:
                            st.caption(f"• {r}")

                # Lightweight "history" + "when made" hints derived from the listing title.
                # (We can't know real production year/history without papers/serial/model data.)
                title_lc = (title_full or "").lower()
                year_m = re.search(r"(19\d{2}|20\d{2})", title_lc)
                made_year = year_m.group(1) if year_m else None
                if made_year:
                    st.caption(f"• Watch made (claimed): {made_year}")

                history_flags: list[str] = []
                if "no papers" in title_lc:
                    history_flags.append("No papers (verify authenticity)")
                elif "papers" in title_lc:
                    history_flags.append("Includes papers (claimed)")

                if "no box" in title_lc:
                    history_flags.append("No box (verify completeness)")
                elif "box" in title_lc:
                    history_flags.append("Includes box (claimed)")

                if "serviced" in title_lc or "service" in title_lc:
                    history_flags.append("Service/serviced mentioned")

                if "warranty" in title_lc:
                    history_flags.append("Warranty mentioned")

                if history_flags:
                    st.caption("**History/features:**")
                    for flag in history_flags[:5]:
                        st.caption(f"• {flag}")
            if row.get("url"):
                st.link_button("Open on eBay", row["url"], type="secondary")


def _render_admin_tab(
    engine,
    df: pd.DataFrame,
    has_ai: bool,
    last_scraped: str | None,
) -> None:
    """Data preview, export, Supabase link, health, scheduled refresh docs."""
    st.subheader("Data & admin")
    st.caption("Check your database, export listings, and verify connections.")

    # Health
    st.markdown("#### Connection health")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.write("**Apify**", "✓ configured" if _get_secret("APIFY_API_KEY") else "✗ missing")
    with h2:
        if not engine:
            st.write("**Database**", "✗")
        elif "sqlite" in str(engine.url).lower():
            st.write("**Database**", "✓ SQLite (local)")
        else:
            st.write("**Database**", "✓ Postgres / Supabase")
    with h3:
        st.write("**OpenAI**", "✓" if _get_secret("OPENAI_API_KEY") else "—")
    with h4:
        st.write("**Anthropic**", "✓" if _get_secret("ANTHROPIC_API_KEY") else "—")
    st.caption("**Data source:** Apify ✓" if _get_secret("APIFY_API_KEY") else "**Data source:** Apify —")
    st.caption("**eBay API** (Client ID + Secret): ✓" if (_get_secret("EBAY_CLIENT_ID") and _get_secret("EBAY_CLIENT_SECRET")) else "**eBay API**: —")

    supabase_ref = _get_secret("SUPABASE_PROJECT_REF")
    dash = _get_secret("SUPABASE_DASHBOARD_URL")
    if dash:
        st.link_button("Open Supabase dashboard", dash, type="primary")
    elif supabase_ref:
        st.link_button(
            "Open Supabase dashboard",
            f"https://supabase.com/dashboard/project/{supabase_ref}",
            type="primary",
        )
    else:
        st.info("Add **SUPABASE_PROJECT_REF** (e.g. `itaefhfsnfjmfhzctrbs`) or **SUPABASE_DASHBOARD_URL** to secrets for a one-click link to your project.")

    # DB stats
    st.markdown("#### Database snapshot")
    if engine:
        try:
            stats = get_db_stats(engine)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rows in DB", stats["listing_count"])
            with c2:
                st.metric("Last saved count", stats["meta_num_listings"])
            with c3:
                ls = stats["last_scraped_at"] or last_scraped or "—"
                st.metric("Last refresh (DB)", str(ls)[:19] if ls and ls != "—" else "—")
        except Exception as e:
            st.warning(f"Could not read DB stats: {e}")
    else:
        st.warning("No database engine (set DATABASE_URL for Supabase).")

    # Export
    st.markdown("#### Export current listings")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "luxury_watch_listings.csv",
        "text/csv",
        use_container_width=True,
    )

    st.markdown("#### Full data preview")
    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("#### Automated refresh (GitHub)")
    st.markdown(
        """
        This repo includes **`.github/workflows/refresh-database.yml`**.  
        In GitHub: **Settings → Secrets and variables → Actions**, add:
        - `APIFY_API_KEY`
        - `DATABASE_URL` (your Supabase URI)

        The workflow runs **twice daily** (06:00 & 18:00 UTC) and can be triggered manually under **Actions**.
        """
    )


def main():
    st.title("⌚ Luxury Watch Deals")
    st.caption("Luxury watch listings from **Apify** or **eBay API** — rank by rule-based and **AI** analysis.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        data_source = st.radio(
            "Data source",
            ["Apify (scraper)", "eBay API (your keys)"],
            help="eBay API uses your eBay Developer keys (Browse API).",
        )
        api_key = _get_secret("APIFY_API_KEY")
        if not api_key:
            api_key = st.text_input(
                "Apify API Key",
                type="password",
                help="Or add APIFY_API_KEY to Streamlit secrets",
            )
        else:
            st.caption("✓ Apify API key loaded from secrets")

        ebay_client_id = _get_secret("EBAY_CLIENT_ID")
        ebay_client_secret = _get_secret("EBAY_CLIENT_SECRET")
        if not ebay_client_id:
            ebay_client_id = st.text_input("eBay Client ID (App ID)", type="password", help="From eBay Developer Program")
        else:
            st.caption("✓ eBay Client ID from secrets")
        if not ebay_client_secret:
            ebay_client_secret = st.text_input("eBay Client Secret (Cert ID)", type="password", help="From eBay Developer Program")
        else:
            st.caption("✓ eBay Client Secret from secrets")

        st.divider()
        st.subheader("eBay API config")
        ebay_marketplace_id = _get_secret("EBAY_MARKETPLACE_ID")
        if not ebay_marketplace_id:
            ebay_marketplace_id = st.text_input(
                "eBay Marketplace ID",
                value="EBAY_US",
                help="Sent as `X-EBAY-C-MARKETPLACE-ID` (e.g. EBAY_US, EBAY_GB).",
            )
        else:
            st.caption(f"✓ Marketplace ID from secrets: `{ebay_marketplace_id}`")
        os.environ["EBAY_MARKETPLACE_ID"] = str(ebay_marketplace_id).strip() or "EBAY_US"

        ebay_use_sandbox_secret = _get_secret("EBAY_USE_SANDBOX").lower()
        ebay_use_sandbox_default = ebay_use_sandbox_secret in {"1", "true", "yes", "y"}
        ebay_use_sandbox = st.checkbox(
            "Use eBay sandbox endpoints",
            value=ebay_use_sandbox_default,
            help="Enable if your keys are sandbox keys (so we call api.sandbox.ebay.com).",
        )
        if ebay_use_sandbox:
            os.environ["EBAY_USE_SANDBOX"] = "true"
        else:
            os.environ.pop("EBAY_USE_SANDBOX", None)

        st.subheader("Search")
        custom_queries = st.text_area(
            "Search queries (one per line)",
            value="\n".join(DEFAULT_WATCH_QUERIES),
            height=120,
        )
        st.divider()
        st.subheader("Brand filter")
        brand_filter_enabled = st.checkbox(
            "Only show high-end brand watches (title must match)",
            value=True,
            help="Filters results after scraping/loading by requiring the listing title to contain at least one brand keyword.",
        )
        default_brand_keywords = [
            "rolex",
            "omega",
            "tag heuer",
            "breitling",
            "cartier",
            "patek philippe",
            "audemars",
            "iwc",
            "vacheron",
            "panerai",
            "glashutte",
            "richard mille",
        ]
        brand_keywords_text = st.text_area(
            "Brand keywords (one per line)",
            value="\n".join(default_brand_keywords),
            height=120,
        )
        max_products = st.slider("Max products per query", 10, 3000, 500)
        max_pages = st.slider("Max pages per query", 1, 20, 5)
        listing_type = st.selectbox("Listing type", ["all", "buy_it_now", "auction"], index=0)
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min price (USD)", min_value=0, value=0, step=100)
        with col2:
            max_price = st.number_input("Max price (USD)", min_value=0, value=0, step=500)
        min_price = min_price or None
        max_price = max_price or None
        run_clicked = st.button(
            "Start scanning",
            type="primary",
            use_container_width=True,
            disabled=not (
                (data_source == "eBay API (your keys)" and ebay_client_id and ebay_client_secret)
                or (data_source == "Apify (scraper)" and api_key)
            ),
        )

        st.divider()
        st.subheader("AI ranking")
        ai_provider = st.selectbox(
            "AI provider",
            ["OpenAI", "Anthropic"],
            help="Used for quality, pricing, and trend analysis.",
        )
        openai_key = _get_secret("OPENAI_API_KEY")
        if not openai_key:
            openai_key = st.text_input("OpenAI API Key", type="password", help="Or add OPENAI_API_KEY to secrets")
        else:
            st.caption("✓ OpenAI key from secrets")
        anthropic_key = _get_secret("ANTHROPIC_API_KEY")
        if not anthropic_key:
            anthropic_key = st.text_input("Anthropic API Key", type="password", help="Or add ANTHROPIC_API_KEY to secrets")
        else:
            st.caption("✓ Anthropic key from secrets")
        max_ai_items = st.slider("Max listings to analyze with AI", 10, 50, 25)
        run_ai_after = st.checkbox(
            "Run AI automatically after scrape & refresh",
            value=False,
            help="Uses your chosen AI provider; needs API key in secrets.",
        )
        run_ai_clicked = st.button("Run AI analysis", use_container_width=True)
        st.divider()
        st.caption("Data freshness")
        stale_after_hours = st.number_input("Warn if data older than (hours)", min_value=1, max_value=168, value=24)

    use_ebay_api = data_source == "eBay API (your keys)"
    can_scan = (use_ebay_api and bool(ebay_client_id and ebay_client_secret)) or (
        not use_ebay_api and bool(api_key)
    )
    has_cloud_db = is_remote_database_configured()

    if not can_scan and not has_cloud_db:
        st.info(
            "**No data source yet.** Add **`DATABASE_URL`** in Streamlit secrets to load listings saved by the scheduled job, "
            "or add an **Apify** key / **eBay API** keys in the sidebar to scan eBay."
        )
        return
    if not can_scan and has_cloud_db:
        st.success(
            "Loading from your **database** (DATABASE_URL). Add Apify or eBay keys in the sidebar if you want to run a new scan from this app."
        )

    # Main-area button (same as sidebar "Run scrape")
    scan_row1, scan_row2 = st.columns([1, 2])
    with scan_row1:
        main_start_scan = st.button(
            "▶ Start scanning",
            type="primary",
            use_container_width=True,
            disabled=not can_scan,
            help="Requires Apify or eBay API keys in the sidebar."
            if not can_scan
            else "Fetch listings using the search queries and limits in the sidebar.",
        )
    with scan_row2:
        st.caption("Uses **Data source**, queries, and max products from the sidebar → Settings.")
    run_clicked = run_clicked or main_start_scan
    if st.session_state.pop("_trigger_scrape_next", False):
        run_clicked = True

    # Preload last scrape from DB when we have no data (first load or empty session)
    try:
        engine = get_engine()
    except Exception:
        engine = None
    need_data = "watch_df" not in st.session_state or st.session_state.get("watch_df") is None or (
        isinstance(st.session_state.get("watch_df"), pd.DataFrame) and st.session_state["watch_df"].empty
    )
    preload_error: str | None = None
    if engine and need_data:
        try:
            cached = get_listings(engine)
            if cached:
                df_cached = pd.DataFrame(cached).sort_values("deal_score", ascending=False).reset_index(drop=True)
                st.session_state["watch_df"] = df_cached
                st.session_state["watch_items"] = cached
        except Exception as e:
            preload_error = str(e)

    if preload_error and need_data:
        st.error(f"**Could not load from database:** {preload_error}")
        st.caption("Check **DATABASE_URL** in secrets (use `postgresql://` for Supabase) and that the DB is reachable.")

    # Run scrape (and save to DB)
    if run_clicked:
        if not can_scan:
            st.error("Add **Apify** or **eBay API** keys in the sidebar to scan.")
            st.stop()
        queries = [q.strip() for q in custom_queries.splitlines() if q.strip()]
        if not queries:
            st.error("Add at least one search query.")
            return
        if use_ebay_api:
            with st.spinner("Fetching from eBay API…"):
                try:
                    items = run_ebay_api_search(
                        client_id=ebay_client_id,
                        client_secret=ebay_client_secret,
                        search_queries=queries,
                        limit_per_query=min(max_products, 200),
                        max_total=max_products * len(queries),
                    )
                except Exception as e:
                    st.error(f"eBay API failed: {e}")
                    return
        else:
            with st.spinner("Scraping eBay via Apify…"):
                try:
                    items = run_ebay_scrape(
                        api_token=api_key,
                        search_queries=queries,
                        max_products_per_search=max_products,
                        max_search_pages=max_pages,
                        listing_type=listing_type,
                        min_price=min_price,
                        max_price=max_price,
                    )
                except Exception as e:
                    st.error(f"Scrape failed: {e}")
                    return
        if not items:
            st.warning("No results. Try different queries or filters.")
            return
        df = items_to_dataframe(items)
        st.session_state["watch_df"] = df
        st.session_state["watch_items"] = items
        if "watch_df_ai" in st.session_state:
            del st.session_state["watch_df_ai"]
        # Persist to DB for fast load next time
        if engine:
            try:
                save_listings(engine, df.to_dict("records"))
            except Exception:
                pass
        if run_ai_after:
            key = openai_key if ai_provider == "OpenAI" else anthropic_key
            if key:
                with st.spinner("Running AI analysis…"):
                    try:
                        if ai_provider == "OpenAI":
                            scored = run_ai_ranking_openai(items, key, max_items=max_ai_items)
                        else:
                            scored = run_ai_ranking_anthropic(items, key, max_items=max_ai_items)
                        df_ai = items_to_dataframe_ai(scored)
                        st.session_state["watch_df"] = df_ai
                        st.session_state["watch_df_ai"] = df_ai
                        st.success("AI ranking complete.")
                    except Exception as e:
                        st.error("AI failed: " + _ai_error_message(e))

    # "Load last scrape" from DB (no new scrape — use after scheduled refresh or to restore view)
    load_db_clicked = st.sidebar.button("Load last scrape from DB", use_container_width=True) if engine else False
    if load_db_clicked and engine:
        try:
            cached = get_listings(engine)
            if cached:
                df_cached = pd.DataFrame(cached).sort_values("deal_score", ascending=False).reset_index(drop=True)
                st.session_state["watch_df"] = df_cached
                st.session_state["watch_items"] = cached
                if "watch_df_ai" in st.session_state:
                    del st.session_state["watch_df_ai"]
                st.sidebar.success(f"Loaded {len(cached)} listings from database.")
            else:
                st.sidebar.warning("Database is empty. Run a scrape first.")
        except Exception as e:
            st.sidebar.error(f"Could not load from DB: {e}")
        st.rerun()

    # "Refresh data" button: re-scrape and save to DB (use 1–2x/day to save load time)
    refresh_clicked = st.sidebar.button("Refresh data (re-scrape & save)", use_container_width=True)
    has_refresh_creds = (api_key and not use_ebay_api) or (use_ebay_api and ebay_client_id and ebay_client_secret)
    if refresh_clicked and engine and has_refresh_creds:
        queries = [q.strip() for q in custom_queries.splitlines() if q.strip()]
        if queries:
            with st.spinner("Refreshing from eBay…"):
                try:
                    if use_ebay_api:
                        items = run_ebay_api_search(
                            client_id=ebay_client_id,
                            client_secret=ebay_client_secret,
                            search_queries=queries,
                            limit_per_query=min(max_products, 200),
                            max_total=max_products * len(queries),
                        )
                    else:
                        items = run_ebay_scrape(
                            api_token=api_key,
                            search_queries=queries,
                            max_products_per_search=max_products,
                            max_search_pages=max_pages,
                            listing_type=listing_type,
                            min_price=min_price,
                            max_price=max_price,
                        )
                    if items:
                        df_new = items_to_dataframe(items)
                        save_listings(engine, df_new.to_dict("records"))
                        st.session_state["watch_df"] = df_new
                        st.session_state["watch_items"] = list(df_new.to_dict("records"))
                        if "watch_df_ai" in st.session_state:
                            del st.session_state["watch_df_ai"]
                        st.success("Data refreshed and saved.")
                        if run_ai_after:
                            key = openai_key if ai_provider == "OpenAI" else anthropic_key
                            if key:
                                with st.spinner("Running AI analysis…"):
                                    try:
                                        if ai_provider == "OpenAI":
                                            scored = run_ai_ranking_openai(items, key, max_items=max_ai_items)
                                        else:
                                            scored = run_ai_ranking_anthropic(items, key, max_items=max_ai_items)
                                        df_ai = items_to_dataframe_ai(scored)
                                        st.session_state["watch_df"] = df_ai
                                        st.session_state["watch_df_ai"] = df_ai
                                    except Exception as e:
                                        st.warning("AI after refresh failed: " + _ai_error_message(e))
                    else:
                        st.warning("No results from scrape.")
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
            st.rerun()

    if "watch_df" not in st.session_state or st.session_state["watch_df"].empty:
        db_hint = ""
        if engine:
            try:
                stats = get_db_stats(engine)
                n = stats["listing_count"]
                db_hint = f" Your database currently has **{n}** listing(s)."
                if n == 0:
                    db_hint += (
                        " Run the **Refresh watch database** GitHub Action (with API keys + DATABASE_URL) "
                        "or scan from this app once you add keys."
                    )
            except Exception:
                pass
        st.info(
            "No listings in this session yet."
            + db_hint
            + " Click **▶ Start scanning** (needs API keys), or **Load last scrape from DB** in the sidebar."
        )
        if can_scan and st.button("▶ Start scanning", type="primary", key="start_scan_empty_state"):
            st.session_state["_trigger_scrape_next"] = True
            st.rerun()
        elif not can_scan and has_cloud_db:
            st.caption("You can only load from DB right now — use **Load last scrape from DB** in the sidebar.")
        return

    df = st.session_state["watch_df"]
    # Optional post-filter: require title to match at least one brand keyword.
    if brand_filter_enabled:
        brand_keywords = [x.strip().lower() for x in brand_keywords_text.splitlines() if x.strip()]
        if brand_keywords and "title" in df.columns and not df.empty:
            def _title_matches_any_brand(t: object) -> bool:
                tl = str(t or "").lower()
                return any(k in tl for k in brand_keywords)

            mask = df["title"].apply(_title_matches_any_brand)
            df = df.loc[mask].reset_index(drop=True)
            st.session_state["watch_df"] = df
            st.session_state["watch_items"] = df.to_dict("records")
    items = st.session_state.get("watch_items", [])
    if not items and not df.empty:
        items = df.to_dict("records")
        st.session_state["watch_items"] = items

    # Run AI analysis
    if run_ai_clicked:
        key = openai_key if ai_provider == "OpenAI" else anthropic_key
        if not key:
            st.error(f"Set {ai_provider} API key in sidebar or secrets.")
        else:
            with st.spinner("Running AI analysis on listings…"):
                try:
                    if ai_provider == "OpenAI":
                        scored = run_ai_ranking_openai(items, key, max_items=max_ai_items)
                    else:
                        scored = run_ai_ranking_anthropic(items, key, max_items=max_ai_items)
                    df_ai = items_to_dataframe_ai(scored)
                    st.session_state["watch_df"] = df_ai
                    st.session_state["watch_df_ai"] = df_ai
                    st.success("AI ranking complete.")
                except Exception as e:
                    st.error("AI analysis failed: " + _ai_error_message(e))

    # Use AI dataframe if available
    df = st.session_state.get("watch_df_ai", df)
    has_ai = "ai_overall_score" in df.columns and df["ai_overall_score"].notna().any()

    # Ensure fake-risk columns (for DB-loaded data) and apply sort preference
    df_rule = _ensure_fake_risk(st.session_state["watch_df"].copy())
    sort_order = st.radio(
        "Listing order",
        ["Best deals first", "Safest first (low fake risk)"],
        horizontal=True,
        key="sort_order",
    )
    if sort_order == "Safest first (low fake risk)":
        df_rule = df_rule.sort_values("fake_risk_score", ascending=True).reset_index(drop=True)

    # Last updated + stale warning
    last_for_admin: str | None = None
    if engine:
        try:
            last = get_last_scraped_at(engine)
            last_for_admin = last
            if last:
                last_str = last[:19] if len(last) > 19 else last
                try:
                    dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    last_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                    age = datetime.now(timezone.utc) - dt
                    if age > timedelta(hours=int(stale_after_hours)):
                        hrs = int(age.total_seconds() // 3600)
                        st.warning(
                            f"Data is about **{hrs}h** old. Click **Refresh data** to update "
                            f"(warn after **{int(stale_after_hours)}h**)."
                        )
                except Exception:
                    pass
                st.caption(f"Data last refreshed: **{last_str}** — use *Refresh data* in the sidebar to update.")
        except Exception:
            pass

    # Overview metrics
    st.subheader("Overview")
    high_risk = (df_rule["fake_risk_score"] >= 60).sum() if "fake_risk_score" in df_rule.columns else 0
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Listings", len(df))
    with c2:
        st.metric("Avg deal score", f"{df['deal_score'].mean():.1f}")
    with c3:
        st.metric("Best deal score", f"{df['deal_score'].max():.1f}")
    with c4:
        st.metric("Avg discount %", f"{df['discount_pct'].mean():.1f}%")
    with c5:
        if has_ai:
            st.metric("Avg AI score", f"{df['ai_overall_score'].mean():.1f}")
    with c6:
        st.metric("High fake risk", high_risk)

    # Likelihood-of-fake calculator
    with st.expander("Likelihood of fake calculator", expanded=False):
        st.caption("Enter listing details to get a rule-based fake/replica likelihood (0–100%). Same logic as table risk scores.")
        calc_title = st.text_input("Listing title", placeholder="e.g. Rolex Submariner Date 41mm...", key="calc_title")
        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            calc_price = st.number_input("Price (optional)", min_value=0.0, value=0.0, step=100.0, key="calc_price",
                help="Leave 0 to skip price-based checks.")
            calc_fb_pct = st.number_input("Seller feedback % (optional)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="calc_fb_pct")
        with calc_col2:
            calc_fb_count = st.number_input("Seller feedback count (optional)", min_value=0, value=0, step=10, key="calc_fb_count")
        if st.button("Calculate likelihood of fake", key="calc_btn"):
            if not (calc_title or "").strip():
                st.warning("Enter at least a title.")
            else:
                from fake_detection import compute_fake_risk
                item = {"title": (calc_title or "").strip()}
                if calc_price and calc_price > 0:
                    item["price_value"] = calc_price
                    item["price"] = calc_price
                if calc_fb_pct and calc_fb_pct > 0:
                    item["sellerFeedbackPercent"] = str(calc_fb_pct)
                if calc_fb_count and calc_fb_count > 0:
                    item["sellerFeedbackCount"] = str(calc_fb_count)
                risk = compute_fake_risk(item)
                score = risk["fake_risk_score"]
                level = risk["fake_risk_level"]
                reasons = risk.get("fake_reasons") or []
                st.metric("Likelihood of fake", f"{score:.0f}%")
                st.write("**Risk level:**", level)
                if reasons:
                    st.write("**Reasons:**")
                    for r in reasons:
                        st.caption(f"• {r}")
                if not reasons:
                    st.caption("No red flags from title, price, or seller.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Rule-based ranking", "AI ranking", "Datasheets", "Data & admin"]
    )
    with tab1:
        _render_rule_based(df_rule)
    with tab2:
        _render_ai_ranking(df)
    with tab3:
        st.subheader("Watch datasheets (top deals)")
        view = st.radio("Sort by", ["Rule-based score", "AI score", "Safest first"], horizontal=True)
        if view == "AI score" and has_ai:
            _render_datasheets(df, use_ai=True)
        elif view == "Safest first":
            safest = _ensure_fake_risk(st.session_state["watch_df"].copy()).sort_values(
                "fake_risk_score", ascending=True
            ).reset_index(drop=True)
            _render_datasheets(safest, use_ai=False)
        else:
            _render_datasheets(st.session_state["watch_df"], use_ai=False)
    with tab4:
        _render_admin_tab(engine, df, has_ai, last_for_admin)

    st.divider()
    st.caption("Data from eBay via Apify. AI uses OpenAI or Anthropic. DB refresh: sidebar or GitHub Actions — see **Data & admin** tab.")


if __name__ == "__main__":
    main()
