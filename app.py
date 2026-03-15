"""
Luxury Watch Deals — Streamlit website.
Scrape eBay via Apify, rank by rule-based and AI analysis (quality, pricing, trends).
Deploy to GitHub + Streamlit Community Cloud.
"""
import os
import streamlit as st
import pandas as pd

from scraper import (
    run_ebay_scrape,
    items_to_dataframe,
    DEFAULT_WATCH_QUERIES,
)
from ai_ranking import (
    run_ai_ranking_openai,
    run_ai_ranking_anthropic,
    items_to_dataframe_ai,
)

st.set_page_config(
    page_title="Luxury Watch Deals | AI Rankings",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Secrets: Streamlit Cloud uses st.secrets; local can use env or sidebar input
def _get_secret(key: str, sidebar_default: str = "") -> str:
    try:
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, sidebar_default)


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


def _render_rule_based(df: pd.DataFrame) -> None:
    """Rule-based ranking view."""
    st.subheader("Rankings by deal score (rule-based)")
    st.caption("Formula: 35% price/discount + 30% condition + 20% seller feedback + 15% popularity.")
    display_cols = [
        "deal_score", "title", "priceString", "listPriceString", "discount_pct",
        "condition_normalized", "sellerFeedbackPercent", "soldCount", "listingType", "url",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].head(100),
        column_config={
            "deal_score": st.column_config.NumberColumn("Deal score", format="%.1f"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "priceString": st.column_config.TextColumn("Price"),
            "listPriceString": st.column_config.TextColumn("List price"),
            "discount_pct": st.column_config.NumberColumn("Discount %", format="%.1f"),
            "condition_normalized": st.column_config.TextColumn("Condition"),
            "sellerFeedbackPercent": st.column_config.TextColumn("Seller %"),
            "soldCount": st.column_config.TextColumn("Sold"),
            "listingType": st.column_config.TextColumn("Type"),
            "url": st.column_config.LinkColumn("Link", display_text="eBay"),
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
        "title", "priceString", "listPriceString", "ai_summary", "condition_normalized",
        "sellerFeedbackPercent", "url",
    ]
    available = [c for c in display_cols if c in df_ai.columns]
    st.dataframe(
        df_ai[available].head(100),
        column_config={
            "ai_overall_score": st.column_config.NumberColumn("AI overall", format="%.1f"),
            "ai_quality_score": st.column_config.NumberColumn("Quality", format="%.1f"),
            "ai_pricing_score": st.column_config.NumberColumn("Pricing", format="%.1f"),
            "ai_trend_score": st.column_config.NumberColumn("Trend", format="%.1f"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "priceString": st.column_config.TextColumn("Price"),
            "listPriceString": st.column_config.TextColumn("List price"),
            "ai_summary": st.column_config.TextColumn("AI summary", width="medium"),
            "condition_normalized": st.column_config.TextColumn("Condition"),
            "sellerFeedbackPercent": st.column_config.TextColumn("Seller %"),
            "url": st.column_config.LinkColumn("Link", display_text="eBay"),
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
        if use_ai and "ai_overall_score" in row and pd.notna(row.get("ai_overall_score")):
            label = f"#{i+1} — {title_short}… | AI: {row['ai_overall_score']:.1f}"
        else:
            label = f"#{i+1} — {title_short}… | Score: {row['deal_score']}"
        with st.expander(label):
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
            if row.get("url"):
                st.link_button("Open on eBay", row["url"], type="secondary")


def main():
    st.title("⌚ Luxury Watch Deals")
    st.caption("Scrape eBay for luxury watches (Apify), then rank by rule-based and **AI** analysis — quality, pricing, trends.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Apify API Key",
            type="password",
            value=_get_secret("APIFY_API_KEY"),
            help="From https://console.apify.com/account/integrations",
        )
        st.divider()
        st.subheader("Search")
        custom_queries = st.text_area(
            "Search queries (one per line)",
            value="\n".join(DEFAULT_WATCH_QUERIES),
            height=120,
        )
        max_products = st.slider("Max products per query", 10, 200, 50)
        max_pages = st.slider("Max pages per query", 1, 10, 3)
        listing_type = st.selectbox("Listing type", ["all", "buy_it_now", "auction"], index=0)
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min price (USD)", min_value=0, value=0, step=100)
        with col2:
            max_price = st.number_input("Max price (USD)", min_value=0, value=0, step=500)
        min_price = min_price or None
        max_price = max_price or None
        run_clicked = st.button("Run scrape", type="primary", use_container_width=True)

        st.divider()
        st.subheader("AI ranking")
        ai_provider = st.selectbox(
            "AI provider",
            ["OpenAI", "Anthropic"],
            help="Used for quality, pricing, and trend analysis.",
        )
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=_get_secret("OPENAI_API_KEY"),
            help="Only if provider is OpenAI.",
        )
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=_get_secret("ANTHROPIC_API_KEY"),
            help="Only if provider is Anthropic.",
        )
        max_ai_items = st.slider("Max listings to analyze with AI", 10, 50, 25)
        run_ai_clicked = st.button("Run AI analysis", use_container_width=True)

    if not api_key:
        st.info("Enter your **Apify API key** in the sidebar to run a scrape.")
        return

    # Run scrape
    if run_clicked:
        queries = [q.strip() for q in custom_queries.splitlines() if q.strip()]
        if not queries:
            st.error("Add at least one search query.")
            return
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
        # Clear previous AI results when re-scraping
        if "watch_df_ai" in st.session_state:
            del st.session_state["watch_df_ai"]

    if "watch_df" not in st.session_state or st.session_state["watch_df"].empty:
        st.info("Click **Run scrape** in the sidebar to load watch listings.")
        return

    df = st.session_state["watch_df"]
    items = st.session_state.get("watch_items", [])

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
                    st.error(f"AI analysis failed: {e}")

    # Use AI dataframe if available
    df = st.session_state.get("watch_df_ai", df)
    has_ai = "ai_overall_score" in df.columns and df["ai_overall_score"].notna().any()

    # Overview metrics
    st.subheader("Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
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

    # Tabs: Rule-based vs AI ranking
    tab1, tab2, tab3 = st.tabs(["Rule-based ranking", "AI ranking", "Datasheets"])
    with tab1:
        _render_rule_based(st.session_state["watch_df"])
    with tab2:
        _render_ai_ranking(df)
    with tab3:
        st.subheader("Watch datasheets (top deals)")
        view = st.radio("Sort by", ["Rule-based score", "AI score"], horizontal=True)
        if view == "AI score" and has_ai:
            _render_datasheets(df, use_ai=True)
        else:
            _render_datasheets(st.session_state["watch_df"], use_ai=False)

    st.divider()
    st.caption("Data from eBay via Apify. AI analysis uses OpenAI or Anthropic. Deploy to GitHub and Streamlit Community Cloud — see README.")


if __name__ == "__main__":
    main()
