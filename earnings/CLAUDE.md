# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python script that fetches scheduled company earnings reports from NASDAQ, analyzes them using a local Ollama AI model with real market data from Yahoo Finance, and generates HTML reports with AI-powered recommendations. Reports are delivered via email (Gmail SMTP/API) or saved to disk.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run (fetches tomorrow's earnings by default)
python3 earnings.py

# Run for specific day (0=today, 1=tomorrow, etc.)
python3 earnings.py --days 0

# Gmail API authentication (run on machine with browser)
python3 authenticate_gmail.py

# Run tests (unit only)
python3 -m pytest test_ai_analysis.py -v -k "not real"

# Run real end-to-end test (requires Ollama + email config)
python3 -m pytest test_ai_analysis.py -v -k "real" -s

# View logs
tail -f logs/earnings_$(date +%Y%m%d).log
```

## Architecture

### Main Script Flow (earnings.py)
1. `validate_config()` - Check environment variables + Ollama connectivity
2. `calculate_target_date()` - Parse --days argument
3. `get_nasdaq_earnings_async()` - Fetch earnings from NASDAQ API
   - **Phase 1 (parallel)**: `get_comprehensive_yfinance_data_async()` fetches fundamentals, price history, ratios + Logo.dev logos — under `Semaphore(50)`
   - **Phase 2 (sequential)**: `get_ai_analysis_async()` sends market data to Ollama for AI recommendation — under `Semaphore(1)` (GPU-bound)
   - Applies filters: market cap ($1B+), major index membership, stock price ($5+)
   - Falls back to Finnhub analyst data if Ollama is unavailable
4. `generate_html_report()` - Create mobile-responsive HTML with AI reasoning tooltips
5. `send_email_gmail()` - Multi-tier delivery: SMTP 465 → SMTP 587 → Gmail API → file save

### AI Analysis Pipeline
- `get_comprehensive_yfinance_data()` - Fetches ~30 data points per ticker (price, P/E, margins, growth, 30-day history, etc.)
- `_build_ai_prompt_data()` - Formats market data into structured prompt
- `get_ai_analysis_async()` - POSTs to Ollama `/api/chat` with financial analyst system prompt
- `_parse_ai_response()` - 3-tier JSON extraction (direct parse → code block → regex)
- `_validate_ai_result()` - Normalizes recommendation/confidence values
- Fallback chain: Ollama AI → Finnhub analysts → hardcoded defaults

### External APIs & Services
- **NASDAQ Calendar API**: Earnings schedule (public)
- **Yahoo Finance (yfinance)**: Comprehensive market data — price, fundamentals, ratios, history (public)
- **Ollama (local)**: AI-powered stock analysis (`OLLAMA_MODEL` in .env, default `gpt-oss:20b`)
- **Finnhub API**: Analyst recommendations fallback (`FINNHUB_IO_API_KEY` in .env)
- **Logo.dev API**: Company logos (`LOGO_DEV_TOKEN` in .env, optional)
- **Wikipedia**: S&P 500/NASDAQ 100 constituents (scraped)
- **Gmail API**: Email delivery (OAuth2 via `token.pickle`)

### Key Configuration (earnings.py)
```python
FILTER_CONFIG = {
    'min_market_cap': 1_000_000_000,  # $1B
    'min_analysts': 0,  # AI is single-source; filter skipped when AI enabled
    'require_major_index': True,
    'min_stock_price': 5.0
}

OLLAMA_CONFIG = {
    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    'model': os.getenv('OLLAMA_MODEL', 'gpt-oss:20b'),
    'timeout': 120,
    'max_retries': 2,
    'enabled': True,  # Auto-disabled if Ollama unreachable
}
```

## Environment Variables (.env)
```bash
EMAIL_SERVICE=gmail
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=<16-char-app-password>
RECIPIENTS=email1@example.com,email2@example.com
FINNHUB_IO_API_KEY=your-api-key      # fallback when Ollama unavailable
LOGO_DEV_TOKEN=your-key              # optional
OLLAMA_MODEL=gpt-oss:20b             # optional, default shown
OLLAMA_BASE_URL=http://localhost:11434 # optional, default shown
```

## Key Implementation Details

- **AI-powered analysis**: Each ticker analyzed by local Ollama model with ~30 fundamental data points
- **Two-phase concurrency**: Market data fetched in parallel (50 concurrent), AI analysis sequential (GPU-bound)
- **Automatic fallback**: Ollama → Finnhub → hardcoded defaults
- **IPv4-only mode**: Socket monkey-patched to force IPv4, avoiding IPv6 timeout issues
- **Email fallback chain**: Automatic fallback through SMTP ports then Gmail API
- **Logging**: Daily rotating logs in `logs/` directory
- **Report truncation**: Email limited to 50 companies; full report saved as backup

## File Structure

- `earnings.py` - Main script (all core logic)
- `test_ai_analysis.py` - Tests (unit + real e2e integration test)
- `authenticate_gmail.py` - Gmail OAuth2 helper
- `requirements.txt` - Dependencies
- `OLLAMA_SETUP.md` - Ollama installation guide for new machines
- `.env` - Configuration (git-ignored)
- `credentials.json`, `token.pickle` - Gmail OAuth (git-ignored)
- `logs/` - Daily log files (git-ignored)
