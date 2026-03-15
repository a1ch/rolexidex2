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
- **Database** — Watches are stored in a DB (SQLite locally, or Postgres via `DATABASE_URL`) so the app loads fast. Use **Refresh data** in the sidebar to re-scrape and update the DB (e.g. once or twice a day).
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
   # DATABASE_URL = "sqlite:///watches.db"   # default; use Postgres on Cloud for persistence
   ```

   Or set environment variables. **Database:** by default the app uses a local SQLite file (`watches.db`). On Streamlit Cloud the filesystem is ephemeral, so set `DATABASE_URL` to a Postgres connection string (e.g. [Supabase](https://supabase.com) free tier) in secrets to persist data.

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

## Host the database on Supabase (free)

1. Go to [supabase.com](https://supabase.com) and sign in (or create an account).
2. Click **New project**. Pick an org, name the project (e.g. `rolexidex2`), set a **database password** (save it — you’ll need it for the connection string), choose a region, then **Create new project**.
3. When the project is ready, open **Project Settings** (gear icon in the sidebar) → **Database**.
4. Under **Connection string**, choose **URI** and copy the string. It will look like:
   `postgresql://postgres.[project-ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres`
   Replace `[YOUR-PASSWORD]` with the database password you set in step 2. If the copied URL uses `postgres://`, change it to `postgresql://`.
5. **Local app:** Put that URL in `.streamlit/secrets.toml`:
   ```toml
   DATABASE_URL = "postgresql://postgres.xxxx:YOUR_PASSWORD@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
   ```
6. **Streamlit Cloud:** In your app on [share.streamlit.io](https://share.streamlit.io), go to **Settings** → **Secrets** and add the same line (root-level key).
7. Restart the app (local or Cloud). The app will create the `listings` and `scrape_metadata` tables in Supabase on first run. Use **Run scrape** or **Refresh data** to populate the DB.

## Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**, choose your repo (`YOUR_USERNAME/Rolexidex2`), branch `main`, and set **Main file path** to `app.py`.
3. Click **Advanced settings** and add your secrets (include `DATABASE_URL` from Supabase — see below):

   ```toml
   APIFY_API_KEY = "your_apify_token"
   OPENAI_API_KEY = "your_openai_key"
   DATABASE_URL = "postgresql://postgres.[ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres"
   ```

4. Deploy. Your app will be available at `https://YOUR_APP_NAME.streamlit.app`.

## Usage

1. **Apify API key** — Required for scraping. Enter in sidebar or set in secrets/env.
2. **Load data** — On open, the app loads watches from the database (fast). If the DB is empty, click **Run scrape** to fetch from eBay and save to the DB.
3. **Refresh data** — Use **Refresh data (re-scrape & save)** in the sidebar to update the DB (e.g. once or twice a day) so normal loads stay fast.
4. **Search** — Edit queries (e.g. Rolex, Omega), max products/pages, price range, listing type before running a scrape or refresh.
5. **Rule-based ranking** — Tab shows deal score table and formula.
6. **AI ranking** — Set OpenAI or Anthropic API key, click **Run AI analysis**. Tab shows quality, pricing, trend, overall score, and AI summary.
7. **Datasheets** — View top deals as cards; choose sort by rule-based or AI score.

## Project structure

| File | Purpose |
|------|--------|
| `app.py` | Streamlit UI: load from DB, scrape/refresh, rule-based & AI tabs, datasheets |
| `scraper.py` | Apify eBay scraper + rule-based deal scoring |
| `ai_ranking.py` | OpenAI/Anthropic AI analysis (quality, pricing, trends) |
| `database.py` | SQLite/Postgres persistence; `get_listings`, `save_listings`, `get_last_scraped_at` |
| `requirements.txt` | streamlit, apify-client, pandas, openai, anthropic, sqlalchemy, psycopg2-binary |
| `.gitignore` | Excludes venv, secrets, watches.db, IDE/OS files |

## Notes

- **Apify** — Pay per product scraped; limit with “Max products per query” and “Max pages” to control cost.
- **AI** — Uses first N listings (configurable); OpenAI `gpt-4o-mini` or Anthropic Haiku are cost-effective.
- **Secrets** — On Streamlit Cloud, use Advanced settings → TOML; locally use `.streamlit/secrets.toml` or env vars. Never commit keys.
- **Database** — Local: SQLite file `watches.db` (created automatically). Cloud: set `DATABASE_URL` to a Postgres URL (e.g. Supabase) in secrets so data persists. Refresh 1–2x/day via the sidebar button to keep load times low.
