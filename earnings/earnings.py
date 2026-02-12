#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Version
VERSION = "3.0"

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='requests')
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core._python_version_support')

# Fix for Python 3.9 compatibility with google-api-python-client
import sys
if sys.version_info < (3, 10):
    try:
        import importlib.metadata as importlib_metadata
        if not hasattr(importlib_metadata, 'packages_distributions'):
            # Monkey patch for Python 3.9
            def packages_distributions():
                pkg_to_dist = {}
                for dist in importlib_metadata.distributions():
                    try:
                        dist_name = dist.metadata.get('Name') or getattr(dist, '_name', 'unknown')
                        if dist.files:
                            for file in dist.files:
                                pkg = str(file).split('/')[0]
                                if pkg not in pkg_to_dist:
                                    pkg_to_dist[pkg] = []
                                pkg_to_dist[pkg].append(dist_name)
                    except Exception:
                        continue
                return pkg_to_dist
            importlib_metadata.packages_distributions = packages_distributions
    except Exception:
        pass

import os
import requests
from datetime import datetime, timedelta
import hashlib
import time
import random
import asyncio
import aiohttp
from dotenv import load_dotenv
import logging
import socket

# Load environment variables from .env file
load_dotenv()

# Force IPv4-only to avoid IPv6 timeout issues
# Store original getaddrinfo
_original_getaddrinfo = socket.getaddrinfo

def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Force all socket connections to use IPv4 only"""
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

# Monkey-patch socket.getaddrinfo to force IPv4
socket.getaddrinfo = _ipv4_only_getaddrinfo

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'earnings_{datetime.now().strftime("%Y%m%d")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also print to console for interactive runs
    ]
)
logger = logging.getLogger(__name__)

# Log IPv4-only mode
logger.info("IPv4-only mode enabled for all network connections")

# Suppress google auth file_cache warnings
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# Email configuration from environment variables
EMAIL_CONFIG = {
    'sender_email': os.getenv('SENDER_EMAIL'),
    'sender_password': os.getenv('SENDER_PASSWORD'),
    'recipients': os.getenv('RECIPIENTS', '').split(',') if os.getenv('RECIPIENTS') else [],
    'email_service': os.getenv('EMAIL_SERVICE', 'gmail')
}

# Gmail API timeout configuration (in seconds)
# Increased for slow networks - adjust if needed
GMAIL_API_TIMEOUT = 120

# Logo.dev API configuration
LOGO_DEV_TOKEN = os.getenv('LOGO_DEV_TOKEN', '')

# RECOMMENDATION SORTING WEIGHTS
recommendation_weights = {
    "Strong Buy": 1,
    "Buy": 2,
    "Hold": 3,
    "Sell": 4,
    "Strong Sell": 5,
}

# FILTERING THRESHOLDS
FILTER_CONFIG = {
    'min_market_cap': 1_000_000_000,  # $1 billion (small/mid-cap threshold)
    'min_analysts': 0,  # AI provides single-source analysis; no minimum needed
    'require_major_index': True,  # Must be in S&P 500, NASDAQ 100, or Russell 2000
    'min_stock_price': 5.0  # Minimum stock price (filters out penny stocks)
}

# Ollama AI Configuration
OLLAMA_CONFIG = {
    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    'model': os.getenv('OLLAMA_MODEL', 'llama3.1:8b'),
    'timeout': 120,  # seconds per request
    'max_retries': 2,
    'enabled': True,  # Auto-disabled at runtime if Ollama is unreachable
}

# Cache for major index constituents
_INDEX_CACHE = {
    'sp500': None,
    'nasdaq100': None,
    'russell2000': None,
    'last_updated': None
}

def get_recommendation_weight(company_data):
    """Extract recommendation weight for sorting"""
    analyst_data = company_data.get('analyst_data', {})
    recommendation = analyst_data.get('recommendation', '')
    return recommendation_weights.get(recommendation, float('inf'))

async def get_major_index_constituents_async(session):
    """Fetch lists of major index constituents (async with caching)"""
    global _INDEX_CACHE

    # Check if cache is still valid (refresh daily)
    if _INDEX_CACHE['last_updated']:
        cache_age = datetime.now() - _INDEX_CACHE['last_updated']
        if cache_age.total_seconds() < 86400:  # 24 hours
            return _INDEX_CACHE

    logger.info(" Fetching major index constituents...")

    sp500_symbols = set()
    nasdaq100_symbols = set()
    russell2000_symbols = set()

    # Use requests (not aiohttp) for Wikipedia — aiohttp gets 403'd by Wikipedia
    import re
    wiki_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html',
    }

    # Fetch S&P 500 from Wikipedia
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=wiki_headers, timeout=15)
        if resp.status_code == 200:
            patterns = [
                r'<td[^>]*><a[^>]*>([A-Z]{1,5})</a>',
                r'<td><a[^>]*>([A-Z]{1,5})</a></td>',
                r'<a[^>]*title="[^"]*">([A-Z]{2,5})</a>',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, resp.text)
                filtered = [m for m in matches if m not in ['NYSE', 'NASDAQ', 'GICS', 'SEC', 'PDF', 'CSV']]
                if len(filtered) >= 400:
                    sp500_symbols = set(filtered[:510])
                    logger.info(f" Found {len(sp500_symbols)} S&P 500 symbols")
                    break
            if not sp500_symbols:
                logger.warning(f" Could not parse S&P 500 symbols from Wikipedia")
        else:
            logger.warning(f" Wikipedia S&P 500 page returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f" Could not fetch S&P 500 list: {str(e)[:100]}")

    # Fetch NASDAQ 100 from Wikipedia
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        resp = requests.get(url, headers=wiki_headers, timeout=15)
        if resp.status_code == 200:
            patterns = [
                (r'<tr>.*?<td[^>]*>([A-Z]{1,5})</td>', re.DOTALL),
                (r'<td[^>]*><a[^>]*>([A-Z]{1,5})</a>', 0),
                (r'<td><a[^>]*>([A-Z]{1,5})</a></td>', 0),
            ]
            for pattern, flags in patterns:
                matches = re.findall(pattern, resp.text, flags)
                seen = set()
                filtered = []
                for m in matches:
                    if m not in ['NYSE', 'NASDAQ', 'GICS', 'SEC', 'PDF', 'CSV', 'WIKI', 'INDEX'] and m not in seen:
                        filtered.append(m)
                        seen.add(m)
                if len(filtered) >= 90:
                    nasdaq100_symbols = set(filtered[:110])
                    logger.info(f" Found {len(nasdaq100_symbols)} NASDAQ 100 symbols")
                    break
            if not nasdaq100_symbols:
                logger.warning(f" Could not parse NASDAQ 100 symbols from Wikipedia")
        else:
            logger.warning(f" Wikipedia NASDAQ 100 page returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f" Could not fetch NASDAQ 100 list: {str(e)[:100]}")

    # For Russell 2000, we'll use a simplified approach - any symbol not in S&P 500/NASDAQ 100
    # with market cap between $300M - $10B is likely Russell 2000
    # We won't pre-fetch Russell 2000 as it's too large and changes frequently

    _INDEX_CACHE = {
        'sp500': sp500_symbols,
        'nasdaq100': nasdaq100_symbols,
        'russell2000': russell2000_symbols,  # Empty for now
        'last_updated': datetime.now()
    }

    return _INDEX_CACHE

def is_in_major_index(symbol, market_cap, index_cache):
    """Check if symbol is in a major index"""
    symbol_upper = symbol.upper()

    # Check S&P 500
    if index_cache['sp500'] and symbol_upper in index_cache['sp500']:
        return True, 'S&P 500'

    # Check NASDAQ 100
    if index_cache['nasdaq100'] and symbol_upper in index_cache['nasdaq100']:
        return True, 'NASDAQ 100'

    return False, None

def apply_filters(earnings_data, index_cache):
    """Apply filtering criteria to earnings data"""
    filtered = []
    stats = {
        'total': len(earnings_data),
        'failed_market_cap': 0,
        'failed_analyst_coverage': 0,
        'failed_index': 0,
        'failed_stock_price': 0,
        'passed': 0
    }

    for company in earnings_data:
        symbol = company.get('symbol', '')
        market_cap = company.get('market_cap')
        analyst_count = company.get('analyst_data', {}).get('total_analysts', 0)
        stock_price = company.get('stock_price')

        # Filter 1: Market cap (must be mid-cap or larger: >= $1B)
        if not market_cap or market_cap < FILTER_CONFIG['min_market_cap']:
            stats['failed_market_cap'] += 1
            continue

        # Filter 2: Analyst coverage (skipped when AI analysis is enabled)
        if not OLLAMA_CONFIG['enabled'] and analyst_count < FILTER_CONFIG['min_analysts']:
            stats['failed_analyst_coverage'] += 1
            continue

        # Filter 3: Major index membership
        in_index, index_name = is_in_major_index(symbol, market_cap, index_cache)
        if FILTER_CONFIG['require_major_index'] and not in_index:
            stats['failed_index'] += 1
            continue

        # Filter 4: Stock price (filter out penny stocks < $5)
        if not stock_price or stock_price < FILTER_CONFIG['min_stock_price']:
            stats['failed_stock_price'] += 1
            continue

        # Passed all filters
        company['index'] = index_name
        filtered.append(company)
        stats['passed'] += 1

    return filtered, stats

async def get_company_logo_async(session, symbol):
    """Get company logo from Logo.dev API (async)"""
    if not LOGO_DEV_TOKEN:
        return None

    try:
        # Logo.dev ticker endpoint - defaults to NYSE/NASDAQ
        logo_url = f"https://img.logo.dev/ticker/{symbol}?token={LOGO_DEV_TOKEN}&format=png&size=128"

        # Logo.dev doesn't support HEAD requests, so we use GET but don't read the body
        # Just check the status code to verify the logo exists
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        async with session.get(logo_url, headers=headers, timeout=aiohttp.ClientTimeout(total=3)) as response:
            if response.status == 200:
                # Logo exists, return the URL without reading the image data
                return logo_url
            else:
                return None

    except asyncio.TimeoutError:
        return None
    except Exception as e:
        return None

async def get_stock_info_async(session, symbol):
    """Get stock price, industry, and market cap from Yahoo Finance (async)"""
    price = None
    industry = None
    market_cap = None

    try:
        # First try to get price from chart API
        chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        async with session.get(chart_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
            if response.status == 200:
                data = await response.json()

            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]

                # Get the regular market price (closing price)
                if 'meta' in result and 'regularMarketPrice' in result['meta']:
                    price = result['meta']['regularMarketPrice']

        # Try to get market cap and industry from quote summary
        try:
            quote_url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
            async with session.get(quote_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'quoteResponse' in data and 'result' in data['quoteResponse'] and data['quoteResponse']['result']:
                        quote = data['quoteResponse']['result'][0]
                        if 'marketCap' in quote and quote['marketCap']:
                            market_cap = quote['marketCap']
                        if 'sector' in quote and quote['sector']:
                            industry = quote['sector']
        except:
            pass

        # Try alternative approach - use search/lookup endpoint
        if not industry:
            try:
                search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}"
                async with session.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                    if 'quotes' in data and data['quotes']:
                        quote = data['quotes'][0]
                        if 'sector' in quote and quote['sector']:
                            industry = quote['sector']
                        elif 'industry' in quote and quote['industry']:
                            industry = quote['industry']
            except:
                pass

        # If that didn't work, try a basic lookup approach
        if not industry:
            # Simple fallback mapping for common cases
            industry_fallback = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
                'TSLA': 'Automotive', 'F': 'Automotive', 'GM': 'Automotive',
                'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
                'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
            }
            industry = industry_fallback.get(symbol.upper())

    except Exception as e:
        pass  # Silent fail for parallel processing

    return price, industry, market_cap

def get_news_link(symbol, company_name):
    """Generate smart news search links - this will ALWAYS work"""
    logger.info(f" Creating news link for {symbol}...")

    # Different link options for variety
    link_options = [
        f"Recent {symbol} earnings & stock news",
        f"Latest {symbol} financial reports",
        f"{symbol} stock analysis & updates",
        f"{symbol} investor news & insights"
    ]

    # Different URL options
    url_options = [
        f"https://www.google.com/search?q={symbol}+earnings+news&tbm=nws&tbs=qdr:w",
        f"https://finance.yahoo.com/quote/{symbol}/news/",
        f"https://www.marketwatch.com/investing/stock/{symbol.lower()}",
        f"https://seekingalpha.com/symbol/{symbol}/news"
    ]

    # Pick random combination
    summary = random.choice(link_options)
    url = random.choice(url_options)

    logger.info(f" Created news link for {symbol}: {summary[:30]}...")

    return {
        "summary": summary,
        "url": url,
        "title": summary
    }

def calculate_consensus_rating(strong_buy, buy, hold, sell, strong_sell):
    """Calculate consensus rating from analyst counts"""

    total_analysts = strong_buy + buy + hold + sell + strong_sell
    if total_analysts == 0:
        return "Hold", 0, "Low"

    # Calculate weighted score (1=Strong Buy, 5=Strong Sell)
    weighted_score = (
        (strong_buy * 1) +
        (buy * 2) +
        (hold * 3) +
        (sell * 4) +
        (strong_sell * 5)
    ) / total_analysts

    # Map score to rating with more realistic thresholds
    if weighted_score <= 1.75:
        consensus = "Strong Buy"
    elif weighted_score <= 2.5:
        consensus = "Buy"
    elif weighted_score <= 3.5:
        consensus = "Hold"
    elif weighted_score <= 4.25:
        consensus = "Sell"
    else:
        consensus = "Strong Sell"

    # Calculate confidence based on agreement
    max_count = max(strong_buy, buy, hold, sell, strong_sell)
    confidence_pct = (max_count / total_analysts) * 100

    if confidence_pct >= 60:
        confidence = "High"
    elif confidence_pct >= 40:
        confidence = "Medium"
    else:
        confidence = "Low"

    return consensus, total_analysts, confidence

def get_yahoo_target_price(symbol):
    """Get target price from Yahoo Finance using yfinance (synchronous helper)"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)

        # Try to get target price with timeout
        info = ticker.get_info()
        target_mean = info.get('targetMeanPrice')

        if target_mean and target_mean > 0:
            logger.info(f" Got Yahoo Finance target for {symbol}: ${target_mean:.2f}")
            return round(target_mean, 2), 'Yahoo Finance'
    except Exception as e:
        logger.warning(f" Yahoo Finance target fetch failed for {symbol}: {str(e)[:50]}")

    return None, 'N/A'

async def get_yahoo_target_price_async(symbol):
    """Get target price from Yahoo Finance using yfinance (async wrapper)"""
    # Run the synchronous yfinance call in a thread pool
    return await asyncio.to_thread(get_yahoo_target_price, symbol)

def get_comprehensive_yfinance_data(symbol):
    """Fetch comprehensive stock data from yfinance for AI analysis.

    Returns dict with fundamentals, price history, and key ratios.
    This data is used both for the report AND as context for the AI model.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()

        # Fetch 30-day price history
        hist = ticker.history(period='1mo')
        price_history = []
        if not hist.empty:
            for date, row in hist.iterrows():
                price_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'close': round(row['Close'], 2),
                    'volume': int(row['Volume']),
                })

        result = {
            'current_price': info.get('currentPrice'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'profit_margins': info.get('profitMargins'),
            'gross_margins': info.get('grossMargins'),
            'operating_margins': info.get('operatingMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
            'eps_trailing': info.get('epsTrailingTwelveMonths'),
            'eps_forward': info.get('epsForward'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'fifty_day_average': info.get('fiftyDayAverage'),
            'two_hundred_day_average': info.get('twoHundredDayAverage'),
            'beta': info.get('beta'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'free_cashflow': info.get('freeCashflow'),
            'dividend_yield': info.get('dividendYield'),
            'yahoo_target_mean': info.get('targetMeanPrice'),
            'price_history_30d': price_history,
        }

        logger.info(f"  Fetched comprehensive yfinance data for {symbol}")
        return result

    except Exception as e:
        logger.warning(f"  yfinance comprehensive fetch failed for {symbol}: {str(e)[:80]}")
        return None

async def get_comprehensive_yfinance_data_async(symbol):
    """Async wrapper for comprehensive yfinance data fetch."""
    return await asyncio.to_thread(get_comprehensive_yfinance_data, symbol)

def _build_ai_prompt_data(symbol, company_name, market_data, eps_forecast):
    """Build a formatted string of market data for the AI prompt."""

    def fmt_pct(val):
        if val is None:
            return "N/A"
        return f"{val * 100:.1f}%"

    def fmt_price(val):
        if val is None:
            return "N/A"
        return f"${val:,.2f}"

    def fmt_num(val):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    def fmt_market_cap(val):
        if val is None:
            return "N/A"
        if val >= 1_000_000_000_000:
            return f"${val/1_000_000_000_000:.1f}T"
        elif val >= 1_000_000_000:
            return f"${val/1_000_000_000:.1f}B"
        elif val >= 1_000_000:
            return f"${val/1_000_000:.0f}M"
        return f"${val:,.0f}"

    lines = [
        f"Symbol: {symbol}",
        f"Company: {company_name}",
        f"Sector: {market_data.get('sector', 'N/A')}",
        f"Industry: {market_data.get('industry', 'N/A')}",
        "",
        "PRICE & VALUATION:",
        f"  Current Price: {fmt_price(market_data.get('current_price'))}",
        f"  Market Cap: {fmt_market_cap(market_data.get('market_cap'))}",
        f"  Trailing P/E: {fmt_num(market_data.get('trailing_pe'))}",
        f"  Forward P/E: {fmt_num(market_data.get('forward_pe'))}",
        f"  Price/Book: {fmt_num(market_data.get('price_to_book'))}",
        f"  Price/Sales: {fmt_num(market_data.get('price_to_sales'))}",
        "",
        "PROFITABILITY:",
        f"  Profit Margin: {fmt_pct(market_data.get('profit_margins'))}",
        f"  Gross Margin: {fmt_pct(market_data.get('gross_margins'))}",
        f"  Operating Margin: {fmt_pct(market_data.get('operating_margins'))}",
        f"  Return on Equity: {fmt_pct(market_data.get('return_on_equity'))}",
        "",
        "GROWTH:",
        f"  Revenue Growth (YoY): {fmt_pct(market_data.get('revenue_growth'))}",
        f"  Earnings Growth (YoY): {fmt_pct(market_data.get('earnings_growth'))}",
        f"  Quarterly Earnings Growth: {fmt_pct(market_data.get('earnings_quarterly_growth'))}",
        "",
        "EARNINGS:",
        f"  EPS (TTM): {fmt_price(market_data.get('eps_trailing'))}",
        f"  EPS (Forward): {fmt_price(market_data.get('eps_forward'))}",
        f"  EPS Forecast (this quarter): {fmt_price(eps_forecast) if eps_forecast else 'N/A'}",
        "",
        "PRICE CONTEXT:",
        f"  52-Week High: {fmt_price(market_data.get('fifty_two_week_high'))}",
        f"  52-Week Low: {fmt_price(market_data.get('fifty_two_week_low'))}",
        f"  50-Day Avg: {fmt_price(market_data.get('fifty_day_average'))}",
        f"  200-Day Avg: {fmt_price(market_data.get('two_hundred_day_average'))}",
        f"  Beta: {fmt_num(market_data.get('beta'))}",
        "",
        "FINANCIAL HEALTH:",
        f"  Debt/Equity: {fmt_num(market_data.get('debt_to_equity'))}",
        f"  Current Ratio: {fmt_num(market_data.get('current_ratio'))}",
        f"  Dividend Yield: {fmt_pct(market_data.get('dividend_yield'))}",
    ]

    # Add price history summary
    price_history = market_data.get('price_history_30d', [])
    if price_history and len(price_history) >= 2:
        first_price = price_history[0]['close']
        last_price = price_history[-1]['close']
        change_pct = ((last_price - first_price) / first_price) * 100
        high_30d = max(p['close'] for p in price_history)
        low_30d = min(p['close'] for p in price_history)
        lines.extend([
            "",
            "30-DAY PRICE TREND:",
            f"  30-Day Change: {change_pct:+.1f}%",
            f"  30-Day High: {fmt_price(high_30d)}",
            f"  30-Day Low: {fmt_price(low_30d)}",
            f"  Trend: {'Upward' if change_pct > 2 else 'Downward' if change_pct < -2 else 'Sideways'}",
        ])

    return "\n".join(lines)


def _parse_ai_response(content, symbol):
    """Parse the AI model's JSON response into the standard analyst_data format."""
    import json
    import re

    if not content or not content.strip():
        return None

    # Try direct JSON parse first
    try:
        data = json.loads(content.strip())
        return _validate_ai_result(data, symbol)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            return _validate_ai_result(data, symbol)
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object with recommendation key
    json_match = re.search(r'\{[^{}]*"recommendation"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return _validate_ai_result(data, symbol)
        except json.JSONDecodeError:
            pass

    logger.warning(f"  Could not parse AI response for {symbol}: {content[:200]}")
    return None


def _validate_ai_result(data, symbol):
    """Validate and normalize the parsed AI response into the standard format."""

    valid_recommendations = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
    valid_confidences = {"High", "Medium", "Low"}

    recommendation = data.get('recommendation', 'Hold')
    if recommendation not in valid_recommendations:
        rec_lower = recommendation.lower().strip()
        rec_map = {
            'strong buy': 'Strong Buy', 'strongbuy': 'Strong Buy',
            'buy': 'Buy',
            'hold': 'Hold', 'neutral': 'Hold',
            'sell': 'Sell',
            'strong sell': 'Strong Sell', 'strongsell': 'Strong Sell',
        }
        recommendation = rec_map.get(rec_lower, 'Hold')

    confidence = data.get('confidence', 'Medium')
    if confidence not in valid_confidences:
        conf_map = {
            'high': 'High',
            'medium': 'Medium',
            'low': 'Low',
        }
        confidence = conf_map.get(str(confidence).lower().strip(), 'Medium')

    target_price = data.get('target_price')
    if target_price is not None:
        try:
            target_price = round(float(target_price), 2)
            if target_price <= 0:
                target_price = None
        except (ValueError, TypeError):
            target_price = None

    reasoning = data.get('reasoning', '')
    if isinstance(reasoning, list):
        reasoning = ' | '.join(str(r) for r in reasoning if r)
    elif not isinstance(reasoning, str):
        reasoning = str(reasoning) if reasoning else ''
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + '...'

    return {
        'recommendation': recommendation,
        'total_analysts': 1,
        'source': 'AI Analysis (Ollama)',
        'confidence': confidence,
        'target_price': target_price,
        'target_source': 'AI Analysis',
        'market_cap': None,  # Filled by caller
        'ai_reasoning': reasoning,
    }


def get_ai_fallback_data(symbol, market_data=None):
    """Fallback when AI analysis fails. Returns a conservative Hold recommendation."""
    logger.warning(f"  Using AI fallback for {symbol}")

    target_price = None
    if market_data and market_data.get('yahoo_target_mean'):
        target_price = round(market_data['yahoo_target_mean'], 2)

    return {
        'recommendation': 'Hold',
        'total_analysts': 0,
        'source': 'AI Fallback (No Analysis)',
        'confidence': 'Low',
        'target_price': target_price,
        'target_source': 'Yahoo Finance' if target_price else 'N/A',
        'market_cap': market_data.get('market_cap') if market_data else None,
        'ai_reasoning': 'AI analysis was unavailable. Defaulting to Hold.',
    }


async def get_ai_analysis_async(session, symbol, company_name, market_data, eps_forecast):
    """Get AI-powered stock analysis from Ollama local model."""
    if not OLLAMA_CONFIG['enabled'] or market_data is None:
        return get_ai_fallback_data(symbol, market_data)

    prompt_data = _build_ai_prompt_data(symbol, company_name, market_data, eps_forecast)

    system_prompt = """You are a cautious, skeptical senior equity research analyst. You protect investors from losses.
Your job is to find reasons NOT to buy a stock, then see if the bull case is strong enough to overcome them.
CRITICAL: Your entire response must be a single JSON object. Do NOT write any text before or after the JSON. Do NOT say "Here is" or explain anything. Start your response with { and end with }."""

    user_prompt = f"""Analyze {symbol} ({company_name}) ahead of their upcoming earnings report.

MARKET DATA:
{prompt_data}

INSTRUCTIONS - Follow these steps IN ORDER before giving your recommendation:

STEP 1 - FUNDAMENTAL ANALYSIS:
- Is the company profitable? Check profit margins and EPS. Negative EPS is a major red flag.
- Are margins improving or declining? Compare operating/gross margins to sector norms.
- Is revenue growing or shrinking? Declining revenue growth is bearish.
- Is the valuation reasonable? High P/E with negative growth = overvalued. No P/E with losses = speculative.
- Check debt/equity and current ratio for financial health risks.

STEP 2 - TECHNICAL ANALYSIS:
- Compare current price to 50-day and 200-day moving averages. Below both = bearish trend.
- Where is price relative to 52-week high/low? Near the low = downtrend. Near the high = momentum.
- What does the 30-day trend show? Downward trend into earnings is a warning sign.
- Check beta for volatility risk.

STEP 3 - BEAR CASE (list at least 2 specific risks based on the data above):
- What could go wrong? Be specific using the actual numbers provided.
- If EPS is negative, revenue is declining, or margins are shrinking, these MUST be listed as risks.

STEP 4 - BULL CASE (list positives, if any):
- What supports a buy? Use specific numbers.

STEP 5 - VERDICT:
- Weigh bear vs bull. Default to CAUTION — only recommend Buy/Strong Buy if bull case clearly outweighs bear case with strong numbers.
- If fundamentals are weak (negative EPS, declining revenue, poor margins), do NOT recommend Buy regardless of narrative.
- Set target_price based on fundamentals and technicals, not optimism. Use 52-week range and moving averages as anchors.

Your ENTIRE response must be this JSON and nothing else. Start with {{ end with }}:
{{"recommendation": "<Strong Buy|Buy|Hold|Sell|Strong Sell>", "confidence": "<High|Medium|Low>", "target_price": <number>, "reasoning": "<2-3 concise sentences summarizing your verdict, key risk, and key catalyst>"}}"""

    for attempt in range(OLLAMA_CONFIG['max_retries'] + 1):
        try:
            payload = {
                "model": OLLAMA_CONFIG['model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0}
            }

            url = f"{OLLAMA_CONFIG['base_url']}/api/chat"
            timeout = aiohttp.ClientTimeout(total=OLLAMA_CONFIG['timeout'])

            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('message', {}).get('content', '')

                    ai_result = _parse_ai_response(content, symbol)
                    if ai_result:
                        ai_result['market_cap'] = market_data.get('market_cap')
                        logger.info(f"  AI analysis for {symbol}: {ai_result['recommendation']} "
                                   f"(confidence: {ai_result['confidence']}, "
                                   f"target: ${ai_result.get('target_price', 'N/A')})")
                        return ai_result
                    else:
                        logger.warning(f"  Failed to parse AI response for {symbol}, attempt {attempt+1}")
                else:
                    logger.warning(f"  Ollama returned status {response.status} for {symbol}")

        except asyncio.TimeoutError:
            logger.warning(f"  Ollama timeout for {symbol}, attempt {attempt+1}")
        except aiohttp.ClientError as e:
            logger.warning(f"  Ollama connection error for {symbol}: {str(e)[:50]}")
        except Exception as e:
            logger.warning(f"  Ollama error for {symbol}: {str(e)[:80]}")

    logger.warning(f"  AI analysis failed for {symbol} after {OLLAMA_CONFIG['max_retries']+1} attempts, using fallback")
    return get_ai_fallback_data(symbol, market_data)

async def get_real_analyst_data_async(session, symbol):
    """Get real analyst ratings, price targets, and market cap from Finnhub API (async)"""

    # Get API key from environment
    finnhub_api_key = os.getenv('FINNHUB_IO_API_KEY')

    analyst_data = {
        'recommendation': 'Hold',
        'total_analysts': 0,
        'source': 'Finnhub Real Data',
        'confidence': 'Low',
        'target_price': None,
        'target_source': 'N/A',
        'market_cap': None
    }

    if not finnhub_api_key or finnhub_api_key == 'your_finnhub_api_key_here':
        return get_fallback_analyst_data(symbol)

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        # Get company profile (includes market cap)
        profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={finnhub_api_key}"
        async with session.get(profile_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                profile_data = await response.json()
                if profile_data and 'marketCapitalization' in profile_data:
                    # Finnhub returns market cap in millions
                    analyst_data['market_cap'] = profile_data['marketCapitalization'] * 1_000_000

        # Get analyst recommendations
        await asyncio.sleep(0.05)  # Small delay for rate limiting
        rec_url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={finnhub_api_key}"
        async with session.get(rec_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                rec_data = await response.json()
            if rec_data and len(rec_data) > 0:
                latest = rec_data[0]  # Most recent data

                strong_buy = latest.get('strongBuy', 0)
                buy = latest.get('buy', 0)
                hold = latest.get('hold', 0)
                sell = latest.get('sell', 0)
                strong_sell = latest.get('strongSell', 0)

                recommendation, total_analysts, confidence = calculate_consensus_rating(
                    strong_buy, buy, hold, sell, strong_sell
                )

                analyst_data.update({
                    'recommendation': recommendation,
                    'total_analysts': total_analysts,
                    'confidence': confidence,
                    'source': 'Finnhub Real Data'
                })

        # Try to get price target from Finnhub
        await asyncio.sleep(0.05)  # Small delay for rate limiting
        target_url = f"https://finnhub.io/api/v1/stock/price-target?symbol={symbol}&token={finnhub_api_key}"
        async with session.get(target_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                target_data = await response.json()
                if target_data and 'targetMean' in target_data and target_data['targetMean']:
                    analyst_data['target_price'] = round(target_data['targetMean'], 2)
                    analyst_data['target_source'] = 'Finnhub Real Data'

        # If Finnhub didn't provide a target price, try Yahoo Finance as fallback
        if analyst_data['target_price'] is None:
            try:
                yahoo_target, yahoo_source = await get_yahoo_target_price_async(symbol)
                if yahoo_target:
                    analyst_data['target_price'] = yahoo_target
                    analyst_data['target_source'] = yahoo_source
            except Exception as e:
                logger.warning(f" Yahoo Finance fallback failed for {symbol}: {str(e)[:50]}")

    except Exception as e:
        return get_fallback_analyst_data(symbol)

    # If we didn't get any analysts, use fallback
    if analyst_data['total_analysts'] == 0:
        return get_fallback_analyst_data(symbol)

    return analyst_data

def get_fallback_analyst_data(symbol, eps=None):
    """Fallback estimated analyst recommendation when real data unavailable"""

    logger.warning(f" Using estimated data for {symbol} (no real analyst data available)")

    # Simple mapping for common stocks
    known_ratings = {
        'AAPL': ('Buy', 25, 'Medium'),
        'MSFT': ('Buy', 28, 'Medium'),
        'GOOGL': ('Buy', 22, 'Medium'),
        'AMZN': ('Buy', 24, 'Medium'),
        'TSLA': ('Hold', 20, 'Low'),
        'META': ('Buy', 21, 'Medium'),
        'NVDA': ('Strong Buy', 30, 'High')
    }

    if symbol.upper() in known_ratings:
        rec, analysts, conf = known_ratings[symbol.upper()]
        return {
            'recommendation': rec,
            'total_analysts': analysts,
            'source': 'Estimated (Default)',
            'confidence': conf,
            'target_price': None,
            'target_source': 'N/A'
        }

    # Generic fallback for unknown stocks
    return {
        'recommendation': 'Hold',
        'total_analysts': 5,
        'source': 'Estimated (Default)',
        'confidence': 'Low',
        'target_price': None,
        'target_source': 'N/A'
    }

async def fetch_market_data(session, row, api_semaphore):
    """Phase 1: Fetch market data + logo in parallel. Fast, no AI."""
    symbol = row.get('symbol', 'N/A')
    company_name = row.get('name', 'N/A')

    # Parse EPS value
    eps_value = row.get('epsForecast', '')
    eps_parsed = None
    if eps_value and eps_value != '':
        try:
            eps_clean = eps_value.replace('$', '').replace('(', '-').replace(')', '')
            eps_parsed = float(eps_clean) if eps_clean else None
        except:
            eps_parsed = None

    async with api_semaphore:
        tasks = [
            get_comprehensive_yfinance_data_async(symbol),
            get_company_logo_async(session, symbol),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    market_data = results[0] if not isinstance(results[0], Exception) else None
    logo_url = results[1] if not isinstance(results[1], Exception) else None

    stock_price = market_data.get('current_price') if market_data else None
    industry = market_data.get('sector') if market_data else None
    yahoo_market_cap = market_data.get('market_cap') if market_data else None

    return {
        'symbol': symbol,
        'company': company_name,
        'time': row.get('time', 'time-not-supplied'),
        'eps': eps_parsed,
        'eps_raw': eps_value,
        'stock_price': stock_price,
        'industry': industry,
        'market_cap': yahoo_market_cap,
        'logo_url': logo_url,
        '_market_data': market_data,  # kept for AI phase
    }


async def run_ai_analysis(session, company, ollama_semaphore):
    """Phase 2: Run AI analysis for a single pre-filtered company (sequential)."""
    symbol = company['symbol']
    company_name = company['company']
    market_data = company.pop('_market_data', None)
    eps_parsed = company.get('eps')

    async with ollama_semaphore:
        if OLLAMA_CONFIG['enabled']:
            analyst_data = await get_ai_analysis_async(
                session, symbol, company_name, market_data, eps_parsed
            )
        else:
            analyst_data = await get_real_analyst_data_async(session, symbol)

    target_price = analyst_data.get('target_price')
    market_cap = analyst_data.get('market_cap') or company.get('market_cap')
    news_data = get_news_link(symbol, company_name)

    company['analyst_data'] = analyst_data
    company['news'] = news_data
    company['target_price'] = target_price
    company['market_cap'] = market_cap
    return company


async def get_nasdaq_earnings_async(target_date=None):
    """Get company earnings from NASDAQ with analyst recommendations and news (async)"""
    logger.info(" Fetching earnings from NASDAQ...")

    if target_date is None:
        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    logger.info(f"️  Checking earnings for: {target_date}")
    api_url = f"https://api.nasdaq.com/api/calendar/earnings?date={target_date}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*'
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Fetch major index constituents first
            index_cache = await get_major_index_constituents_async(session)

            async with session.get(api_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()

                    earnings_data = []
                    if 'data' in data and 'rows' in data['data'] and data['data']['rows'] is not None:
                        rows = data['data']['rows']
                        logger.info(f" Found {len(rows)} companies, fetching market data in parallel...")

                        # --- PHASE 1: Parallel market data fetch for ALL companies (fast) ---
                        api_semaphore = asyncio.Semaphore(50)
                        market_tasks = [fetch_market_data(session, row, api_semaphore) for row in rows]

                        start_time = time.time()
                        all_companies = await asyncio.gather(*market_tasks, return_exceptions=True)
                        all_companies = [c for c in all_companies if not isinstance(c, Exception)]

                        market_elapsed = time.time() - start_time
                        logger.info(f" Fetched market data for {len(all_companies)} companies in {market_elapsed:.1f}s")

                        # --- PRE-FILTER: Apply filters BEFORE expensive AI analysis ---
                        logger.info("\n🔍 Pre-filtering before AI analysis...")
                        logger.info(f"   - Market Cap: >= ${FILTER_CONFIG['min_market_cap']:,.0f} (Mid/Large-cap)")
                        logger.info(f"   - Major Index: S&P 500, NASDAQ 100, or Russell 2000")
                        logger.info(f"   - Stock Price: >= ${FILTER_CONFIG['min_stock_price']:.2f} (No penny stocks)")

                        filtered_data, filter_stats = apply_filters(all_companies, index_cache)

                        logger.info(f"\n📊 Filter Results:")
                        logger.info(f"   Total companies: {filter_stats['total']}")
                        logger.info(f"   ❌ Failed market cap filter: {filter_stats['failed_market_cap']}")
                        logger.info(f"   ❌ Failed analyst coverage filter: {filter_stats['failed_analyst_coverage']}")
                        logger.info(f"   ❌ Failed index membership filter: {filter_stats['failed_index']}")
                        logger.info(f"   ❌ Failed stock price filter: {filter_stats['failed_stock_price']}")
                        logger.info(f"   ✅ Passed all filters: {filter_stats['passed']}")

                        # --- PHASE 2: AI analysis ONLY for filtered companies (sequential, slow) ---
                        if OLLAMA_CONFIG['enabled']:
                            logger.info(f"\n🤖 Running AI analysis on {len(filtered_data)} filtered companies ({OLLAMA_CONFIG['model']})...")
                        else:
                            logger.info(f"\n📊 Fetching Finnhub analyst data for {len(filtered_data)} filtered companies...")

                        ollama_semaphore = asyncio.Semaphore(1)
                        ai_tasks = [run_ai_analysis(session, company, ollama_semaphore) for company in filtered_data]

                        ai_start = time.time()
                        earnings_data = await asyncio.gather(*ai_tasks, return_exceptions=True)
                        earnings_data = [e for e in earnings_data if not isinstance(e, Exception)]

                        ai_elapsed = time.time() - ai_start
                        total_elapsed = time.time() - start_time
                        logger.info(f" AI analysis completed in {ai_elapsed:.1f}s ({ai_elapsed/max(len(earnings_data),1):.1f}s/company)")
                        logger.info(f" Total pipeline: {total_elapsed:.1f}s (market: {market_elapsed:.1f}s + AI: {ai_elapsed:.1f}s)")

                        # SORT BY RECOMMENDATION PRIORITY
                        logger.info("\n🔄 Sorting companies by recommendation priority...")
                        earnings_data.sort(key=get_recommendation_weight)

                        # Print sorted order for verification
                        logger.info("\n📋 Final filtered & sorted list:")
                        for i, company in enumerate(earnings_data):
                            rec = company['analyst_data']['recommendation']
                            index = company.get('index', 'N/A')
                            market_cap_b = company.get('market_cap', 0) / 1_000_000_000
                            analysts = company['analyst_data']['total_analysts']
                            logger.info(f"  {i+1}. {company['symbol']:6} - {rec:12} | {index:25} | ${market_cap_b:6.1f}B | {analysts} analysts")
                    else:
                        logger.info("️  No companies found reporting earnings tomorrow")

            return earnings_data

    except Exception as e:
        logger.error(f" NASDAQ error: {e}")
        return []


def get_nasdaq_earnings(target_date=None):
    """Wrapper to run async function synchronously"""
    return asyncio.run(get_nasdaq_earnings_async(target_date))

def generate_html_report(earnings_data, target_date_display=None, is_full_report=True):
    """Generate HTML email report with analyst recommendations and news"""
    if target_date_display is None:
        target_date_display = (datetime.now() + timedelta(days=1)).strftime('%A, %B %d, %Y')
    tomorrow = target_date_display

    # Calculate statistics
    total_companies = len(earnings_data)
    profitable = len([e for e in earnings_data if e.get('eps') and e.get('eps') > 0])
    losses = len([e for e in earnings_data if e.get('eps') and e.get('eps') < 0])
    unknown = len([e for e in earnings_data if e.get('eps') is None])
    pre_market = len([e for e in earnings_data if e.get('time') == 'time-pre-market'])
    after_hours = len([e for e in earnings_data if e.get('time') == 'time-after-hours'])


    # Limit companies in email to prevent Gmail truncation (full report saved separately)
    email_company_limit = 50  # Adjust this based on your needs
    display_earnings = earnings_data if is_full_report else earnings_data[:email_company_limit]
    truncated_count = len(earnings_data) - len(display_earnings) if not is_full_report else 0

    # Generate company rows (already sorted by recommendation)
    company_rows = ""
    mobile_cards = ""
    for company in display_earnings:
        symbol = company.get('symbol', 'N/A')
        company_name = company.get('company', 'N/A')
        time_value = company.get('time', 'time-not-supplied')
        eps_value = company.get('eps')
        news_data = company.get('news', {})
        stock_price = company.get('stock_price')
        target_price = company.get('target_price')
        industry = company.get('industry')
        logo_url = company.get('logo_url')

        # Format stock price
        price_display = f"${stock_price:.2f}" if stock_price else "N/A"
        price_color = "#333"

        # Format target price and calculate upside/downside
        target_display = f"${target_price:.2f}" if target_price else "N/A"
        upside_pct = ""
        target_color = "#333"

        if stock_price and target_price:
            upside = ((target_price - stock_price) / stock_price) * 100
            if upside > 0:
                upside_pct = f" (+{upside:.1f}%)"
                target_color = "#28a745"  # Green for upside
            elif upside < 0:
                upside_pct = f" ({upside:.1f}%)"
                target_color = "#dc3545"  # Red for downside
            else:
                upside_pct = " (0.0%)"
                target_color = "#6c757d"  # Gray for neutral

        # Format EPS display
        eps_display = f"${eps_value:.2f}" if eps_value is not None else "N/A"
        eps_color = "#28a745" if (eps_value and eps_value > 0) else "#dc3545" if (eps_value and eps_value < 0) else "#6c757d"

        # Format timing
        time_display = {
            'time-pre-market': 'Pre',
            'time-after-hours': 'After',
            'time-not-supplied': 'TBD'
        }.get(time_value, 'TBD')

        time_color = {
            'time-pre-market': '#007bff',
            'time-after-hours': '#6f42c1',
            'time-not-supplied': '#6c757d'
        }.get(time_value, '#6c757d')

        # Get analyst recommendation data
        analyst_data = company.get('analyst_data', {})
        recommendation = analyst_data.get('recommendation', 'N/A')
        total_analysts = analyst_data.get('total_analysts', 0)
        source = analyst_data.get('source', '')
        confidence = analyst_data.get('confidence', '')
        ai_reasoning = analyst_data.get('ai_reasoning', '')

        # Map recommendations to icons and colors
        rec_icon = "❓"
        rec_color = "#6c757d"

        if recommendation == 'Strong Buy':
            rec_icon = "🚀"
            rec_color = "#28a745"
        elif recommendation == 'Buy':
            rec_icon = "💚"
            rec_color = "#28a745"
        elif recommendation == 'Hold':
            rec_icon = "🤝"
            rec_color = "#ffc107"
        elif recommendation == 'Sell':
            rec_icon = "📉"
            rec_color = "#dc3545"
        elif recommendation == 'Strong Sell':
            rec_icon = "💥"
            rec_color = "#dc3545"

        # Format news data - ensure we have valid values
        news_summary = news_data.get('summary', 'Search latest news')
        news_url = news_data.get('url', f"https://www.google.com/search?q={symbol}+stock+news&tbm=nws")

        # Format industry (longer for mobile since we have more space now)
        industry_display = industry[:25] + "..." if industry and len(industry) > 25 else (industry or "N/A")
        industry_mobile = industry[:20] + "..." if industry and len(industry) > 20 else (industry or "N/A")

        # Ensure all variables have safe values
        symbol_safe = symbol or 'N/A'
        company_safe = company_name[:16] + "..." if company_name and len(company_name) > 16 else (company_name or 'N/A')

        # Create logo HTML if available
        logo_html = ''
        if logo_url:
            logo_html = f'<img src="{logo_url}" alt="{symbol}" style="width: 24px; height: 24px; border-radius: 4px; margin-right: 8px; vertical-align: middle; object-fit: contain;">'

        # Combine logo and company name
        company_with_logo = f'<div style="display: flex; align-items: center;">{logo_html}<span>{company_safe}</span></div>'

        time_safe = time_display or 'TBD'
        price_safe = price_display or 'N/A'
        target_safe = f'<span style="color: {target_color}; font-weight: bold;">{target_display}{upside_pct}</span>' if target_price else 'N/A'
        eps_safe = eps_display or 'N/A'
        industry_safe = industry_display or 'N/A'
        rec_safe = f"{rec_icon} {recommendation}" if recommendation else 'N/A'
        conf_safe = confidence if confidence else 'N/A'
        news_safe = f'<a href="{news_url}" target="_blank" style="color: #007bff; text-decoration: none; font-size: 11px;">{news_summary[:20]}...</a>'

        company_rows += f"""
        <tr>
            <td style="padding: 8px 8px 4px; width: 7%;">{symbol_safe}</td>
            <td style="padding: 8px 8px 4px; width: 15%;">{company_with_logo}</td>
            <td style="padding: 8px 8px 4px; text-align: center; width: 7%;">{time_safe}</td>
            <td style="padding: 8px 8px 4px; text-align: center; width: 8%;">{price_safe}</td>
            <td style="padding: 8px 8px 4px; text-align: center; width: 11%;">{target_safe}</td>
            <td style="padding: 8px 8px 4px; text-align: center; width: 7%;">{eps_safe}</td>
            <td style="padding: 8px 8px 4px; width: 13%;">{industry_safe}</td>
            <td style="padding: 8px 8px 4px; width: 10%;">{rec_safe}</td>
            <td style="padding: 8px 8px 4px; width: 10%;">{conf_safe}</td>
            <td style="padding: 8px 8px 4px; width: 12%;">{news_safe}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e9ecef;">
            <td colspan="10" style="padding: 0 8px 10px 8px;">
                <div style="font-size: 11px; color: #555; line-height: 1.4; background: #f8f9fa; padding: 6px 10px; border-radius: 6px; word-wrap: break-word;">
                    <span style="font-weight: 600; color: #666; font-size: 10px; text-transform: uppercase;">AI Insight</span>&nbsp;&nbsp;{ai_reasoning if ai_reasoning else 'No AI insight available.'}
                </div>
            </td>
        </tr>
        """

        # Generate mobile card
        mobile_logo_html = ''
        if logo_url:
            mobile_logo_html = f'<img src="{logo_url}" alt="{symbol}" style="width: 32px; height: 32px; border-radius: 6px; margin-right: 12px; object-fit: contain;">'

        mobile_cards += f"""
        <div class="mobile-card">
            <div class="card-header">
                <div style="display: flex; align-items: center;">
                    {mobile_logo_html}
                    <div class="card-symbol">{symbol_safe}</div>
                </div>
                <div class="card-rating">{rec_safe}</div>
            </div>
            <div class="card-company">{company_name or 'N/A'}</div>
            <div class="card-details">
                <div class="card-item">
                    <span class="card-label">Price</span>
                    <span class="card-value">{price_safe}</span>
                </div>
                <div class="card-item">
                    <span class="card-label">Target</span>
                    <span class="card-value" style="color: {target_color}; font-weight: bold;">{target_display}{upside_pct}</span>
                </div>
                <div class="card-item">
                    <span class="card-label">EPS</span>
                    <span class="card-value">{eps_safe}</span>
                </div>
                <div class="card-item">
                    <span class="card-label">Industry</span>
                    <span class="card-value">{industry_mobile}</span>
                </div>
                <div class="card-item">
                    <span class="card-label">Time</span>
                    <span class="card-value">{time_safe}</span>
                </div>
                <div class="card-item">
                    <span class="card-label">Confidence</span>
                    <span class="card-value">{conf_safe}</span>
                </div>
            </div>
            {'<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e9ecef;"><span class="card-label" style="width: auto; display: block; margin-bottom: 4px;">AI INSIGHT</span><span style="font-size: 11px; color: #555; line-height: 1.4;">' + ai_reasoning + '</span></div>' if ai_reasoning else ''}
            <div class="card-news">
                <a href="{news_url}" target="_blank">📰 {news_summary[:25]}...</a>
            </div>
        </div>
        """


    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Earnings Report</title>
        <style>
            /* Mobile Cards View - Hidden by default */
            .mobile-cards {{
                display: none;
            }}

            .mobile-card {{
                background: white;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}

            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
                border-bottom: 1px solid #e9ecef;
                padding-bottom: 8px;
            }}

            .card-symbol {{
                font-weight: bold;
                font-size: 16px;
                color: #333;
            }}

            .card-rating {{
                font-size: 14px;
            }}

            .card-company {{
                font-size: 12px;
                color: #666;
                margin-bottom: 12px;
            }}

            .card-details {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }}

            .card-item {{
                display: flex;
                align-items: center;
                padding: 0;
                margin-bottom: 4px;
            }}

            .card-label {{
                font-size: 10px;
                color: #666;
                text-transform: uppercase;
                font-weight: 600;
                white-space: nowrap;
                margin-right: 6px;
                flex-shrink: 0;
                width: 50px;
            }}

            .card-value {{
                font-size: 12px;
                font-weight: bold;
                text-align: left;
                flex: 1;
            }}

            .card-news {{
                margin-top: 12px;
                padding-top: 8px;
                border-top: 1px solid #e9ecef;
            }}

            .card-news a {{
                color: #007bff;
                text-decoration: none;
                font-size: 12px;
                font-weight: 500;
            }}

            .card-news a:hover {{
                text-decoration: underline;
            }}

            /* Media Query - Switch to mobile cards on small screens */
            @media (max-width: 768px) {{
                .desktop-table {{
                    display: none;
                }}

                .mobile-cards {{
                    display: block;
                }}

                body {{
                    padding: 10px !important;
                }}

                .container {{
                    padding: 20px !important;
                }}
            }}
        </style>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1100px; margin: 0 auto; background-color: #f8f9fa;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 42px; font-weight: bold;">📊 EARNINGS CALENDAR</h1>
            <p style="margin: 15px 0 0 0; font-size: 20px; opacity: 0.9;">{tomorrow}</p>
            <div style="background: rgba(255,255,255,0.2); padding: 15px 25px; border-radius: 25px; display: inline-block; margin-top: 20px;">
                <span style="font-weight: bold; font-size: 20px;">🏢 {total_companies} Major Companies Reporting</span>
            </div>
        </div>

        <div style="padding: 40px; background: white;">
            <!-- Statistics -->
            <div style="background: #f8f9fa; padding: 16px 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                    <div style="flex: 1; text-align: center; padding: 0 10px;">
                        <div style="font-size: 20px; font-weight: bold; color: #28a745; margin-bottom: 2px;">📈 {profitable}</div>
                        <div style="font-size: 11px; color: #666; text-transform: uppercase; font-weight: 600;">Profitable</div>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 0 10px;">
                        <div style="font-size: 20px; font-weight: bold; color: #dc3545; margin-bottom: 2px;">📉 {losses}</div>
                        <div style="font-size: 11px; color: #666; text-transform: uppercase; font-weight: 600;">Losses</div>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 0 10px;">
                        <div style="font-size: 20px; font-weight: bold; color: #007bff; margin-bottom: 2px;">🌅 {pre_market}</div>
                        <div style="font-size: 11px; color: #666; text-transform: uppercase; font-weight: 600;">Pre-Market</div>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 0 10px;">
                        <div style="font-size: 20px; font-weight: bold; color: #6f42c1; margin-bottom: 2px;">🌙 {after_hours}</div>
                        <div style="font-size: 11px; color: #666; text-transform: uppercase; font-weight: 600;">After Hours</div>
                    </div>
                </div>
            </div>

            <!-- Companies Table -->
            <div class="desktop-table" style="background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 8px 25px rgba(0,0,0,0.1); border: 1px solid #e9ecef;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 25px;">
                    <h2 style="margin: 0; font-size: 28px; font-weight: bold;">📈 Companies Reporting Tomorrow{f' (Showing {len(display_earnings)} of {len(earnings_data)})' if truncated_count > 0 else ''}</h2>
                    {f'<p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">⚠️ Email shows top {len(display_earnings)} companies. {truncated_count} additional companies - see attached full report.</p>' if truncated_count > 0 else ''}
                </div>
                <table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 7%;">Symbol</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 15%;">Company</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 7%;">Time</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 8%;">Price</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 11%;">Target</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 7%;">EPS</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 13%;">Industry</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 10%;">Rating</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 10%;">Confidence</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 12%;">News</th>
                        </tr>
                    </thead>
                    <tbody>
                        {company_rows}
                    </tbody>
                </table>
            </div>

            <!-- Mobile Cards View -->
            <div class="mobile-cards">
                {mobile_cards}
            </div>

            <!-- Legend -->
            <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-top: 30px; border-left: 4px solid #667eea;">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px;">📋 Recommendation Guide</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 14px;">
                    <div><span style="font-size: 18px;">🚀</span> Strong Buy (Very Bullish)</div>
                    <div><span style="font-size: 18px;">💚</span> Buy (Bullish)</div>
                    <div><span style="font-size: 18px;">🤝</span> Hold (Neutral)</div>
                    <div><span style="font-size: 18px;">📉</span> Sell (Bearish)</div>
                    <div><span style="font-size: 18px;">💥</span> Strong Sell (Very Bearish)</div>
                    <div><span style="font-size: 18px;">❓</span> No Data Available</div>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                    * Recommendations from AI analysis (Ollama {OLLAMA_CONFIG['model']})<br>
                    * AI analysis based on fundamental data, valuation metrics, and price trends from Yahoo Finance<br>
                    * Target prices generated by AI model based on comprehensive market data<br>
                    * Companies are sorted by recommendation priority (Strong Buy first, Strong Sell last)
                </div>
            </div>

            <!-- Filter Criteria -->
            <div style="background: #e8f4f8; padding: 20px; border-radius: 12px; margin-top: 30px; border-left: 4px solid #007bff;">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px;">🔍 Filter Criteria</h3>
                <div style="font-size: 14px; color: #333;">
                    <strong>This report includes only high-quality companies that meet ALL of the following criteria:</strong>
                    <div style="margin-top: 10px; line-height: 1.8;">
                        ✓ <strong>Market Cap:</strong> Minimum ${FILTER_CONFIG['min_market_cap'] / 1_000_000_000:.1f} billion (mid/large-cap stocks)<br>
                        ✓ <strong>Analysis:</strong> AI-powered analysis using Ollama ({OLLAMA_CONFIG['model']})<br>
                        ✓ <strong>Index Membership:</strong> Must be in S&P 500, NASDAQ 100, or Russell 2000<br>
                        ✓ <strong>Stock Price:</strong> Minimum ${FILTER_CONFIG['min_stock_price']:.2f} per share (no penny stocks)<br>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #666; font-style: italic;">
                    These filters help focus on liquid, well-researched companies with institutional interest.
                </div>
            </div>

            <!-- Footer -->
            <div style="margin-top: 40px; padding-top: 25px; border-top: 3px solid #e9ecef; text-align: center; color: #666; font-size: 14px;">
                <strong>📊 Data Source: NASDAQ + Yahoo Finance + Ollama AI</strong> • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                EPS = Earnings Per Share (Analyst Estimates) • Pre-Market: Before 9:30 AM • After Hours: After 4:00 PM<br>
                <br>
                <strong>Earnings Report v{VERSION}</strong><br>
                <em>This is an automated financial report. To stop receiving these emails, contact the sender.</em>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content

def send_email_gmail_api(subject, html_content, recipients, attachment_html=None, attachment_filename=None):
    """Send email using Gmail API (uses HTTPS, works when SMTP ports are blocked)"""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        import base64
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        import pickle
        from datetime import datetime, timedelta
    except ImportError:
        logger.error(" Gmail API libraries not installed. Install with: pip install google-auth-oauthlib google-api-python-client")
        return False

    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    sender_email = EMAIL_CONFIG['sender_email']

    try:
        import socket
        socket.setdefaulttimeout(GMAIL_API_TIMEOUT)

        creds = None
        token_file = 'token.pickle'

        # Load existing credentials
        if os.path.exists(token_file):
            logger.info("Loading credentials from token.pickle")
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)

       # Check if token will expire soon (within 5 minutes) and refresh proactively
        if creds and creds.valid and creds.expiry:
            from datetime import datetime, timedelta
            if creds.expiry < datetime.utcnow() + timedelta(minutes=5):
                logger.info("Token expiring soon, refreshing proactively")
                try:
                    creds.refresh(Request())
                    with open(token_file, 'wb') as token:
                        pickle.dump(creds, token)
                    logger.info("Token refreshed proactively and saved")
                except Exception as e:
                    logger.warning(f"Proactive refresh failed: {e}")

        # Handle invalid or expired credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Token expiring soon, refreshing proactively")
                try:
                    creds.refresh(Request())
                    with open('token.pickle', 'wb') as token:
                        pickle.dump(creds, token)
                        logger.info("Credentials refreshed and saved successfully")
                except Exception as refresh_error:
                    logger.error(f"Failed to refresh token: {refresh_error}")
                    logger.error("Token may have been revoked. Please re-authenticate:")
                    logger.error("   1. Delete token.pickle")
                    logger.error("   2. Run 'python3 authenticate_gmail.py'")
                    logger.error("   3. Copy new token.pickle to this server")
                    return False
            else:
                logger.error(" No valid credentials found. Please authenticate:")
                logger.error("   1. Run 'python3 authenticate_gmail.py' on a machine with a browser")
                logger.error("   2. Copy token.pickle to this server")
                logger.error("   3. Place it in the same directory as earnings.py")
                return False

            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        # Build the Gmail service with proper timeout configuration
        logger.info(f"Building Gmail API service with {GMAIL_API_TIMEOUT}s timeout")
        # Create HTTP client with timeout
        import google.auth.transport.requests
        import google_auth_httplib2
        import httplib2

        # Configure HTTP client with custom timeout
        http = httplib2.Http(timeout=GMAIL_API_TIMEOUT)
        http = google_auth_httplib2.AuthorizedHttp(creds, http=http)

        # Build service with custom HTTP client
        service = build('gmail', 'v1', http=http, cache_discovery=False)

        # Send individual emails to each recipient to avoid timeout
        logger.info(f"Sending emails to {len(recipients)} recipients individually")
        import socket
        import time

        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(GMAIL_API_TIMEOUT)

        sent_count = 0
        failed_recipients = []

        try:
            for i, recipient in enumerate(recipients, 1):
                try:
                    # Create message for this recipient
                    logger.info(f"Creating email message for recipient {i}/{len(recipients)}: {recipient}")
                    start_time = time.time()

                    msg = MIMEMultipart('alternative')
                    msg['Subject'] = subject
                    msg['From'] = f"Earnings Alert System <{sender_email}>"
                    msg['To'] = recipient

                    # Attach HTML content
                    html_part = MIMEText(html_content, 'html')
                    msg.attach(html_part)

                    # Add attachment if provided
                    if attachment_html and attachment_filename:
                        attachment = MIMEBase('text', 'html')
                        attachment.set_payload(attachment_html.encode('utf-8'))
                        encoders.encode_base64(attachment)
                        attachment.add_header('Content-Disposition', f'attachment; filename={attachment_filename}')
                        msg.attach(attachment)

                    # Encode the message
                    msg_bytes = msg.as_bytes()
                    if i == 1:  # Log size only for first email
                        msg_size_kb = len(msg_bytes) / 1024
                        logger.info(f"Message size: {msg_size_kb:.1f} KB")

                    encode_start = time.time()
                    raw_message = base64.urlsafe_b64encode(msg_bytes).decode('utf-8')
                    message_body = {'raw': raw_message}
                    logger.info(f"Message encoding took {time.time() - encode_start:.1f}s")

                    # Send the message with detailed timing
                    api_start = time.time()
                    logger.info(f"Calling Gmail API send (timeout: {GMAIL_API_TIMEOUT}s)...")
                    service.users().messages().send(userId='me', body=message_body).execute()
                    api_time = time.time() - api_start

                    logger.info(f" Email {i}/{len(recipients)} sent successfully to {recipient} (API call: {api_time:.1f}s, total: {time.time() - start_time:.1f}s)")
                    sent_count += 1

                    # Add delay between sends to avoid rate limiting (except after last email)
                    if i < len(recipients):
                        logger.info(f"Waiting 5 seconds before next recipient...")
                        time.sleep(5)

                except Exception as e:
                    logger.error(f" Failed to send email to {recipient}: {type(e).__name__}: {e}")

                    # If attachment was included, try sending without it
                    if attachment_html and attachment_filename:
                        try:
                            logger.info(f"Retrying without attachment for {recipient}...")
                            retry_start = time.time()

                            # Create simpler message without attachment
                            msg_simple = MIMEMultipart('alternative')
                            msg_simple['Subject'] = subject + " (attachment too large - see full report file)"
                            msg_simple['From'] = f"Earnings Alert System <{sender_email}>"
                            msg_simple['To'] = recipient
                            html_part = MIMEText(html_content, 'html')
                            msg_simple.attach(html_part)

                            # Encode and send
                            msg_bytes_simple = msg_simple.as_bytes()
                            logger.info(f"Retry message size: {len(msg_bytes_simple) / 1024:.1f} KB (was {msg_size_kb:.1f} KB)")
                            raw_message_simple = base64.urlsafe_b64encode(msg_bytes_simple).decode('utf-8')
                            message_body_simple = {'raw': raw_message_simple}

                            service.users().messages().send(userId='me', body=message_body_simple).execute()
                            retry_time = time.time() - retry_start
                            logger.info(f" Email sent to {recipient} WITHOUT attachment (retry took {retry_time:.1f}s)")
                            sent_count += 1

                            if i < len(recipients):
                                logger.info(f"Waiting 5 seconds before next recipient...")
                                time.sleep(5)
                            continue

                        except Exception as retry_e:
                            logger.error(f" Retry also failed for {recipient}: {type(retry_e).__name__}: {retry_e}")

                    failed_recipients.append(recipient)
                    continue

        finally:
            socket.setdefaulttimeout(original_timeout)

        if sent_count > 0:
            logger.info(f" Successfully sent {sent_count}/{len(recipients)} emails via Gmail API")
            if failed_recipients:
                logger.warning(f" Failed recipients: {', '.join(failed_recipients)}")
            return True
        else:
            logger.error(" Failed to send any emails via Gmail API")
            return False

    except HttpError as e:
        if e.resp.status in [401, 403]:
            logger.error(f" Gmail API authentication failed (HTTP {e.resp.status})")
            logger.error(" Token may have been revoked or expired")
            logger.error(" To fix this:")
            logger.error("   1. Delete token.pickle file")
            logger.error("   2. Run 'python3 authenticate_gmail.py' on a machine with a browser")
            logger.error("   3. Copy the new token.pickle to this server")
        else:
            logger.error(f" Gmail API HTTP error {e.resp.status}: {e}")
        return False

    except socket.timeout as e:
        logger.error(f" Gmail API socket timeout: {e}")
        return False

    except Exception as e:
        logger.error(f" Failed to send email via Gmail API: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def send_email_gmail_smtp(subject, html_content, recipients, attachment_html=None, attachment_filename=None):
    """Send email using Gmail SMTP with optional HTML attachment"""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    sender_email = EMAIL_CONFIG['sender_email']
    sender_password = EMAIL_CONFIG['sender_password']

    if not sender_email or not sender_password:
        logger.error(" Gmail credentials not found in environment variables")
        return False

    # Try port 465 first, then 587
    ports = [
        (465, 'SMTP_SSL', 'smtplib.SMTP_SSL'),
        (587, 'STARTTLS', 'smtplib.SMTP')
    ]

    for port, method, _ in ports:
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"Earnings Alert System <{sender_email}>"
            msg['To'] = ', '.join(recipients)

            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Add attachment if provided
            if attachment_html and attachment_filename:
                attachment = MIMEBase('text', 'html')
                attachment.set_payload(attachment_html.encode('utf-8'))
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', f'attachment; filename={attachment_filename}')
                msg.attach(attachment)

            # Connect to Gmail SMTP server
            if port == 465:
                import smtplib
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10)
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipients, msg.as_string())
                server.quit()
            else:
                import smtplib
                server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipients, msg.as_string())
                server.quit()

            logger.info(f" Email sent successfully via Gmail SMTP (port {port}) to {len(recipients)} recipients")
            return True

        except Exception as e:
            logger.warning(f"  Failed to send via SMTP port {port} ({method}): {e}")
            if port == ports[-1][0]:  # Last port in the list
                logger.error(f" All SMTP ports failed")
                return False
            else:
                logger.info(f" Trying next port...")
                continue

    return False

def send_email_gmail(subject, html_content, recipients, attachment_html=None, attachment_filename=None):
    """Send email using Gmail with automatic fallback: SMTP -> Gmail API -> Save to file"""

    # Try SMTP first (faster if available)
    logger.info(" Attempting to send email via Gmail SMTP...")
    if send_email_gmail_smtp(subject, html_content, recipients, attachment_html, attachment_filename):
        return True

    # Fall back to Gmail API (works when SMTP ports are blocked)
    logger.info(" SMTP failed, trying Gmail API...")
    if send_email_gmail_api(subject, html_content, recipients, attachment_html, attachment_filename):
        return True

    # Both methods failed
    logger.error(" Both Gmail SMTP and API failed")
    return False

def save_to_file(subject, html_content):
    """Save report to file as fallback"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"earnings_report_{timestamp}.html"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f" Report saved to file: {filename}")
        logger.info(f" Open {filename} in your browser to view the report")
        return True

    except Exception as e:
        logger.error(f" Failed to save file: {e}")
        return False

def validate_config():
    """Validate environment configuration"""
    errors = []

    if not EMAIL_CONFIG['sender_email']:
        errors.append("SENDER_EMAIL is required")

    if not EMAIL_CONFIG['recipients'] or EMAIL_CONFIG['recipients'] == ['']:
        errors.append("RECIPIENTS is required")

    service = EMAIL_CONFIG['email_service']
    if service == 'gmail' and not EMAIL_CONFIG['sender_password']:
        errors.append("SENDER_PASSWORD is required when using Gmail service")

    # Validate Ollama connectivity if enabled
    if OLLAMA_CONFIG['enabled']:
        try:
            resp = requests.get(f"{OLLAMA_CONFIG['base_url']}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if OLLAMA_CONFIG['model'] not in models:
                    logger.warning(f"  Ollama model '{OLLAMA_CONFIG['model']}' not found. "
                                  f"Available: {models}. Falling back to Finnhub.")
                    OLLAMA_CONFIG['enabled'] = False
                else:
                    logger.info(f"  Ollama connected: {OLLAMA_CONFIG['model']} ready")
            else:
                logger.warning(f"  Ollama returned status {resp.status_code}. Falling back to Finnhub.")
                OLLAMA_CONFIG['enabled'] = False
        except Exception as e:
            logger.warning(f"  Cannot connect to Ollama at {OLLAMA_CONFIG['base_url']}: {e}. "
                          f"Falling back to Finnhub.")
            OLLAMA_CONFIG['enabled'] = False

    if errors:
        logger.error(" Configuration errors:")
        for error in errors:
            logger.info(f"   • {error}")
        logger.info("\n💡 Check your .env file and make sure all required variables are set")
        return False

    return True

def main():
    """Main function for cron job"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch earnings reports for a specific date')
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days from today to fetch earnings (default: 1 for tomorrow)'
    )
    args = parser.parse_args()

    if not validate_config():
        return

    # Calculate target date based on days parameter
    target_date = (datetime.now() + timedelta(days=args.days)).strftime('%Y-%m-%d')
    target_date_display = (datetime.now() + timedelta(days=args.days)).strftime('%A, %B %d, %Y')

    logger.debug(f" Fetching earnings for: {target_date_display}")

    earnings_data = get_nasdaq_earnings(target_date=target_date)

    if not earnings_data:
        return

    # Generate full report
    full_html_report = generate_html_report(earnings_data, target_date_display, is_full_report=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Always save full report to file as backup
    full_report_filename = f"earnings_report_full_{timestamp}.html"
    try:
        with open(full_report_filename, 'w', encoding='utf-8') as f:
            f.write(full_html_report)
        logger.info(f" Full report saved to: {full_report_filename}")
    except Exception as e:
        logger.warning(f" Could not save full report: {e}")

    # Determine if we need to limit the email content
    needs_truncation = len(earnings_data) > 50

    # Generate email content (limited if needed)
    email_html_report = generate_html_report(earnings_data, target_date_display, is_full_report=not needs_truncation)
    subject = f"Earnings Report of Major Companies - {target_date}"

    service = EMAIL_CONFIG['email_service']
    success = False

    if service == 'gmail':
        # Attach full report if content was truncated
        attachment_html = full_html_report if needs_truncation else None
        attachment_name = f"earnings_full_report_{target_date}.html" if needs_truncation else None
        success = send_email_gmail(subject, email_html_report, EMAIL_CONFIG['recipients'],
                                   attachment_html, attachment_name)
    else:
        success = save_to_file(subject, email_html_report)

    if not success and service != 'file':
        save_to_file(subject, email_html_report)

if __name__ == "__main__":
    main()
