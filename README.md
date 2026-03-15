# Luxury Watch Deals — Website

A **Streamlit website** that scrapes eBay for luxury watches (via Apify), then ranks them using **rule-based** and **AI** analysis (quality, pricing, trends). Push to GitHub and deploy on **Streamlit Community Cloud**.

## Features

- **eBay scraping** — [Apify eBay Scraper](https://apify.com/automation-lab/ebay-scraper): search queries, filters (price, condition, listing type).
- **Rule-based ranking** — Deal score (0–100) from price/discount, condition, seller feedback, and popularity (sold count).
- **AI ranking** — LLM analysis (OpenAI or Anthropic) scores each listing on:
  - **Quality** (1–10): brand, condition, title cues.
  - **Pricing** (1–10): value vs list price and market.
  - **Trends** (1–10): seller reputation and sold count.
  - **Overall** (1–10) + short **summary**.
- **Datasheets** — Sortable tables and expandable cards with image, prices, condition, seller, and link to eBay.
- **Deployable** — Ready for GitHub and Streamlit Community Cloud (secrets for API keys).

## Local setup

1. **Clone and enter the project:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/Rolexidex2.git
   cd Rolexidex2
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **API keys** (use at least Apify; add OpenAI or Anthropic for AI ranking):
   - [Apify](https://console.apify.com/account/integrations) — API token
   - [OpenAI](https://platform.openai.com/api-keys) — for AI ranking (e.g. `gpt-4o-mini`)
   - [Anthropic](https://console.anthropic.com/) — optional alternative for AI ranking

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   For local secrets (optional), create `.streamlit/secrets.toml` (do **not** commit it):

   ```toml
   APIFY_API_KEY = "your_apify_token"
   OPENAI_API_KEY = "your_openai_key"
   # ANTHROPIC_API_KEY = "your_anthropic_key"
   ```

   Or set environment variables: `APIFY_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

## Deploy to GitHub

1. Create a new repository on GitHub (e.g. `Rolexidex2`).
2. From your project folder:

   ```bash
   git init
   git add .
   git commit -m "Luxury watch deals app - Streamlit + Apify + AI ranking"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/Rolexidex2.git
   git push -u origin main
   ```

3. Do **not** commit `.streamlit/secrets.toml` or any file with API keys (`.gitignore` already excludes them).

## Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**, choose your repo (`YOUR_USERNAME/Rolexidex2`), branch `main`, and set **Main file path** to `app.py`.
3. Click **Advanced settings** and add your secrets (root-level keys become environment variables):

   ```toml
   APIFY_API_KEY = "your_apify_token"
   OPENAI_API_KEY = "your_openai_key"
   ```

   Or add only the keys you use (e.g. Apify for scraping; OpenAI or Anthropic for AI ranking).
4. Deploy. Your app will be available at `https://YOUR_APP_NAME.streamlit.app`.

## Usage

1. **Apify API key** — Required for scraping. Enter in sidebar or set in secrets/env.
2. **Search** — Edit queries (e.g. Rolex, Omega, luxury automatic watch), set max products/pages, price range, listing type. Click **Run scrape**.
3. **Rule-based ranking** — Tab shows deal score table and formula.
4. **AI ranking** — Set OpenAI or Anthropic API key, click **Run AI analysis**. Tab shows quality, pricing, trend, overall score, and AI summary.
5. **Datasheets** — View top deals as cards; choose sort by rule-based or AI score.

## Project structure

| File | Purpose |
|------|--------|
| `app.py` | Streamlit UI: scrape, rule-based & AI tabs, datasheets, secrets/env for keys |
| `scraper.py` | Apify eBay scraper + rule-based deal scoring |
| `ai_ranking.py` | OpenAI/Anthropic AI analysis and scoring (quality, pricing, trends) |
| `requirements.txt` | streamlit, apify-client, pandas, openai, anthropic |
| `.gitignore` | Excludes venv, secrets, IDE/OS files |

## Notes

- **Apify** — Pay per product scraped; limit with “Max products per query” and “Max pages” to control cost.
- **AI** — Uses first N listings (configurable); OpenAI `gpt-4o-mini` or Anthropic Haiku are cost-effective.
- **Secrets** — On Streamlit Cloud, use Advanced settings → TOML; locally use `.streamlit/secrets.toml` or env vars. Never commit keys.
