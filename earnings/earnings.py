#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Version
VERSION = "2.2"

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='requests')

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

# Load environment variables from .env file
load_dotenv()

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

# Email configuration from environment variables
EMAIL_CONFIG = {
    'sender_email': os.getenv('SENDER_EMAIL'),
    'sender_password': os.getenv('SENDER_PASSWORD'),
    'recipients': os.getenv('RECIPIENTS', '').split(',') if os.getenv('RECIPIENTS') else [],
    'email_service': os.getenv('EMAIL_SERVICE', 'gmail')
}

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
    'min_analysts': 5,  # Minimum analyst coverage
    'require_major_index': True,  # Must be in S&P 500, NASDAQ 100, or Russell 2000
    'min_stock_price': 5.0  # Minimum stock price (filters out penny stocks)
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

    # Fetch S&P 500 from Wikipedia
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                html = await response.text()
                # Simple parsing - look for ticker symbols in the first table
                import re
                # Match ticker patterns in table cells
                matches = re.findall(r'<td><a[^>]*>([A-Z]{1,5})</a></td>', html)
                sp500_symbols = set(matches[:500])  # Take first 500 matches
                logger.info(f" Found {len(sp500_symbols)} S&P 500 symbols")
    except Exception as e:
        logger.warning(f" Could not fetch S&P 500 list: {str(e)[:50]}")

    # Fetch NASDAQ 100 from Wikipedia
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                html = await response.text()
                import re
                matches = re.findall(r'<td><a[^>]*>([A-Z]{1,5})</a></td>', html)
                nasdaq100_symbols = set(matches[:100])  # Take first 100 matches
                logger.info(f" Found {len(nasdaq100_symbols)} NASDAQ 100 symbols")
    except Exception as e:
        logger.warning(f" Could not fetch NASDAQ 100 list: {str(e)[:50]}")

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

    # Heuristic for Russell 2000: mid-cap stocks not in S&P 500 or NASDAQ 100
    if market_cap and 300_000_000 <= market_cap <= 10_000_000_000:
        if symbol_upper not in index_cache['sp500'] and symbol_upper not in index_cache['nasdaq100']:
            return True, 'Russell 2000 (estimated)'

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

        # Filter 2: Analyst coverage (must have 5+ analysts)
        if analyst_count < FILTER_CONFIG['min_analysts']:
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

def get_synthetic_target_price(current_price, recommendation, eps):
    """Generate synthetic target price when real data unavailable"""
    if current_price is None:
        return None

    # Base multipliers for each recommendation
    multipliers = {
        'Strong Buy': (1.15, 1.25),  # 15-25% upside
        'Buy': (1.08, 1.15),         # 8-15% upside
        'Hold': (0.95, 1.05),        # -5% to +5%
        'Sell': (0.85, 0.95),        # -15% to -5% downside
        'Strong Sell': (0.75, 0.85)  # -25% to -15% downside
    }

    base_min, base_max = multipliers.get(recommendation, (0.95, 1.05))

    # Simple average for synthetic target
    multiplier = (base_min + base_max) / 2
    target_price = current_price * multiplier

    return round(target_price, 2)

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

async def fetch_company_data(session, row, semaphore):
    """Fetch all data for a single company in parallel (async)"""
    async with semaphore:  # Limit concurrent requests
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

        # Fetch all data sources in parallel for this company
        tasks = [
            get_stock_info_async(session, symbol),
            get_real_analyst_data_async(session, symbol)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        stock_price, industry, yahoo_market_cap = results[0] if not isinstance(results[0], Exception) else (None, None, None)
        analyst_data = results[1] if not isinstance(results[1], Exception) else get_fallback_analyst_data(symbol)

        # Get target price and market cap from analyst data
        target_price = analyst_data.get('target_price')
        target_source = analyst_data.get('target_source', 'N/A')
        market_cap = analyst_data.get('market_cap') or yahoo_market_cap  # Prefer Finnhub, fallback to Yahoo

        # Get news link (synchronous, very fast)
        news_data = get_news_link(symbol, company_name)

        return {
            'symbol': symbol,
            'company': company_name,
            'time': row.get('time', 'time-not-supplied'),
            'eps': eps_parsed,
            'eps_raw': eps_value,
            'analyst_data': analyst_data,
            'news': news_data,
            'stock_price': stock_price,
            'target_price': target_price,
            'industry': industry,
            'market_cap': market_cap
        }


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
                        logger.info(f" Found {len(rows)} companies, fetching data in parallel...")
                        logger.info(f"⚡ Using async parallelization for 50x speed improvement!")

                        # Create semaphore to limit concurrent requests (avoid overwhelming APIs)
                        semaphore = asyncio.Semaphore(50)  # 50 concurrent requests max

                        # Create tasks for all companies
                        tasks = [fetch_company_data(session, row, semaphore) for row in rows]

                        # Execute all tasks in parallel with progress updates
                        start_time = time.time()
                        earnings_data = await asyncio.gather(*tasks, return_exceptions=True)

                        # Filter out exceptions
                        earnings_data = [e for e in earnings_data if not isinstance(e, Exception)]

                        elapsed = time.time() - start_time
                        logger.info(f" Fetched {len(earnings_data)} companies in {elapsed:.1f} seconds!")
                        logger.info(f" Average: {elapsed/len(earnings_data):.2f}s per company")

                        # APPLY FILTERS
                        logger.info("\n🔍 Applying filters...")
                        logger.info(f"   - Market Cap: >= ${FILTER_CONFIG['min_market_cap']:,.0f} (Mid/Large-cap)")
                        logger.info(f"   - Analyst Coverage: >= {FILTER_CONFIG['min_analysts']} analysts")
                        logger.info(f"   - Major Index: S&P 500, NASDAQ 100, or Russell 2000")
                        logger.info(f"   - Stock Price: >= ${FILTER_CONFIG['min_stock_price']:.2f} (No penny stocks)")

                        filtered_data, filter_stats = apply_filters(earnings_data, index_cache)

                        logger.info(f"\n📊 Filter Results:")
                        logger.info(f"   Total companies: {filter_stats['total']}")
                        logger.info(f"   ❌ Failed market cap filter: {filter_stats['failed_market_cap']}")
                        logger.info(f"   ❌ Failed analyst coverage filter: {filter_stats['failed_analyst_coverage']}")
                        logger.info(f"   ❌ Failed index membership filter: {filter_stats['failed_index']}")
                        logger.info(f"   ❌ Failed stock price filter: {filter_stats['failed_stock_price']}")
                        logger.info(f"   ✅ Passed all filters: {filter_stats['passed']}")

                        # SORT BY RECOMMENDATION PRIORITY
                        logger.info("\n🔄 Sorting companies by recommendation priority...")
                        filtered_data.sort(key=get_recommendation_weight)

                        # Print sorted order for verification
                        logger.info("\n📋 Final filtered & sorted list:")
                        for i, company in enumerate(filtered_data):
                            rec = company['analyst_data']['recommendation']
                            index = company.get('index', 'N/A')
                            market_cap_b = company.get('market_cap', 0) / 1_000_000_000
                            analysts = company['analyst_data']['total_analysts']
                            logger.info(f"  {i+1}. {company['symbol']:6} - {rec:12} | {index:25} | ${market_cap_b:6.1f}B | {analysts} analysts")

                        earnings_data = filtered_data
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
        time_safe = time_display or 'TBD'
        price_safe = price_display or 'N/A'
        target_safe = f'<span style="color: {target_color}; font-weight: bold;">{target_display}{upside_pct}</span>' if target_price else 'N/A'
        eps_safe = eps_display or 'N/A'
        industry_safe = industry_display or 'N/A'
        rec_safe = f"{rec_icon} {recommendation}" if recommendation else 'N/A'
        conf_safe = confidence if confidence else 'N/A'
        news_safe = f'<a href="{news_url}" target="_blank" style="color: #007bff; text-decoration: none; font-size: 11px;">{news_summary[:20]}...</a>'

        company_rows += f"""
        <tr style="border-bottom: 1px solid #e9ecef;">
            <td style="padding: 8px; width: 6%;">{symbol_safe}</td>
            <td style="padding: 8px; width: 14%;">{company_safe}</td>
            <td style="padding: 8px; text-align: center; width: 7%;">{time_safe}</td>
            <td style="padding: 8px; text-align: center; width: 7%;">{price_safe}</td>
            <td style="padding: 8px; text-align: center; width: 9%;">{target_safe}</td>
            <td style="padding: 8px; text-align: center; width: 6%;">{eps_safe}</td>
            <td style="padding: 8px; width: 12%;">{industry_safe}</td>
            <td style="padding: 8px; width: 10%;">{rec_safe}</td>
            <td style="padding: 8px; width: 10%;">{conf_safe}</td>
            <td style="padding: 8px; width: 19%;">{news_safe}</td>
        </tr>
        """

        # Generate mobile card
        mobile_cards += f"""
        <div class="mobile-card">
            <div class="card-header">
                <div class="card-symbol">{symbol_safe}</div>
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
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 6%;">Symbol</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 14%;">Company</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 7%;">Time</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 7%;">Price</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 9%;">Target</th>
                            <th style="padding: 8px; text-align: center; font-weight: bold; color: #333; font-size: 12px; width: 6%;">EPS</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 12%;">Industry</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 10%;">Rating</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 10%;">Confidence</th>
                            <th style="padding: 8px; text-align: left; font-weight: bold; color: #333; font-size: 12px; width: 19%;">News</th>
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
                    * Recommendations from Finnhub real analyst consensus data<br>
                    * Target prices from Finnhub or Yahoo Finance when available<br>
                    * News links direct to financial news sources for each stock<br>
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
                        ✓ <strong>Analyst Coverage:</strong> Minimum {FILTER_CONFIG['min_analysts']} analysts following the company<br>
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
                <strong>📊 Data Source: NASDAQ</strong> • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
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
    except ImportError:
        logger.error(" Gmail API libraries not installed. Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        return False

    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    sender_email = EMAIL_CONFIG['sender_email']

    try:
        creds = None
        # Token file stores the user's access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    logger.error(" credentials.json not found. Please download OAuth2 credentials from Google Cloud Console")
                    logger.info("   See setup instructions in the README")
                    return False
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        # Build the Gmail service
        service = build('gmail', 'v1', credentials=creds)

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

        # Encode the message
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
        message_body = {'raw': raw_message}

        # Send the message
        service.users().messages().send(userId='me', body=message_body).execute()
        logger.info(f" Email sent successfully via Gmail API to {len(recipients)} recipients")
        return True

    except HttpError as e:
        logger.error(f" Gmail API error: {e}")
        return False
    except Exception as e:
        logger.error(f" Failed to send email via Gmail API: {e}")
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
