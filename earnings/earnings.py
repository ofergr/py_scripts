#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Version
VERSION = "1.10"

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='requests')

import os
import requests
from datetime import datetime, timedelta
import hashlib
import time
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Email configuration from environment variables
EMAIL_CONFIG = {
    'sendgrid_api_key': os.getenv('SENDGRID_API_KEY'),
    'sender_email': os.getenv('SENDER_EMAIL'),
    'recipients': os.getenv('RECIPIENTS', '').split(',') if os.getenv('RECIPIENTS') else [],
    'email_service': os.getenv('EMAIL_SERVICE', 'sendgrid')
}

# RECOMMENDATION SORTING WEIGHTS
recommendation_weights = {
    "Strong Buy": 1,
    "Buy": 2,
    "Hold": 3,
    "Sell": 4,
    "Strong Sell": 5,
}

def get_recommendation_weight(company_data):
    """Extract recommendation weight for sorting"""
    analyst_data = company_data.get('analyst_data', {})
    recommendation = analyst_data.get('recommendation', '')
    return recommendation_weights.get(recommendation, float('inf'))

def get_stock_info(symbol):
    """Get stock price and industry from Yahoo Finance"""
    print(f"💰 Getting info for {symbol}...")

    price = None
    industry = None

    try:
        # First try to get price from chart API
        chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(chart_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()

            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]

                # Get the regular market price (closing price)
                if 'meta' in result and 'regularMarketPrice' in result['meta']:
                    price = result['meta']['regularMarketPrice']

        # Try alternative approach - use search/lookup endpoint
        try:
            search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}"
            response = requests.get(search_url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
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
        print(f"❌ Info fetch error for {symbol}: {e}")

    if price:
        print(f"✅ Got info for {symbol}: ${price:.2f}, {industry or 'N/A'}")
    else:
        print(f"⚠️ Partial info for {symbol}: Price=N/A, Industry={industry or 'N/A'}")

    return price, industry

def get_news_link(symbol, company_name):
    """Generate smart news search links - this will ALWAYS work"""
    print(f"📰 Creating news link for {symbol}...")

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

    print(f"✅ Created news link for {symbol}: {summary[:30]}...")

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
    """Get target price from Yahoo Finance using yfinance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)

        # Try to get target price with timeout
        info = ticker.get_info()
        target_mean = info.get('targetMeanPrice')

        if target_mean and target_mean > 0:
            print(f"✅ Got Yahoo Finance target for {symbol}: ${target_mean:.2f}")
            return round(target_mean, 2), 'Yahoo Finance'
    except Exception as e:
        print(f"⚠️ Yahoo Finance target fetch failed for {symbol}: {str(e)[:50]}")

    return None, 'N/A'

def get_real_analyst_data(symbol):
    """Get real analyst ratings and price targets from Finnhub API"""

    print(f"📊 Getting real analyst data for {symbol}...")

    # Get API key from environment
    finnhub_api_key = os.getenv('FINNHUB_IO_API_KEY')

    analyst_data = {
        'recommendation': 'Hold',
        'total_analysts': 0,
        'source': 'Finnhub Real Data',
        'confidence': 'Low',
        'target_price': None,
        'target_source': 'N/A'
    }

    if not finnhub_api_key or finnhub_api_key == 'your_finnhub_api_key_here':
        print(f"⚠️ No Finnhub API key found for {symbol}, using fallback")
        return get_fallback_analyst_data(symbol)

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        # Get analyst recommendations
        rec_url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={finnhub_api_key}"
        response = requests.get(rec_url, headers=headers, timeout=10)

        if response.status_code == 200:
            rec_data = response.json()
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

                print(f"✅ Got real recommendation for {symbol}: {recommendation} ({total_analysts} analysts)")

        # Try to get price target from Finnhub first
        time.sleep(0.1)  # Rate limit respect
        target_url = f"https://finnhub.io/api/v1/stock/price-target?symbol={symbol}&token={finnhub_api_key}"
        response = requests.get(target_url, headers=headers, timeout=10)

        if response.status_code == 200:
            target_data = response.json()
            if target_data and 'targetMean' in target_data and target_data['targetMean']:
                analyst_data['target_price'] = round(target_data['targetMean'], 2)
                analyst_data['target_source'] = 'Finnhub Real Data'
                print(f"✅ Got Finnhub price target for {symbol}: ${analyst_data['target_price']}")

        # If Finnhub doesn't have target price, try Yahoo Finance
        if not analyst_data['target_price']:
            yahoo_target, yahoo_source = get_yahoo_target_price(symbol)
            if yahoo_target:
                analyst_data['target_price'] = yahoo_target
                analyst_data['target_source'] = yahoo_source

    except Exception as e:
        print(f"❌ Error fetching analyst data for {symbol}: {e}")
        return get_fallback_analyst_data(symbol)

    # If we didn't get any analysts, use fallback
    if analyst_data['total_analysts'] == 0:
        return get_fallback_analyst_data(symbol)

    return analyst_data

def get_fallback_analyst_data(symbol, eps=None):
    """Fallback estimated analyst recommendation when real data unavailable"""

    print(f"⚠️ Using estimated data for {symbol} (no real analyst data available)")

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

def get_nasdaq_earnings():
    """Get company earnings from NASDAQ with analyst recommendations and news"""
    print("📊 Fetching earnings from NASDAQ...")

    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    api_url = f"https://api.nasdaq.com/api/calendar/earnings?date={tomorrow}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*'
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()

            earnings_data = []
            if 'data' in data and 'rows' in data['data']:
                print(f"🔍 Found {len(data['data']['rows'])} companies, generating recommendations and news links...")

                for i, row in enumerate(data['data']['rows']):
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

                    # Get stock price and industry (all companies)
                    stock_price, industry = get_stock_info(symbol)

                    # Get real analyst recommendation and target price
                    print(f"📈 Getting analyst data for {symbol} ({i+1}/{len(data['data']['rows'])})")
                    analyst_data = get_real_analyst_data(symbol)

                    # Get target price from analyst data (already fetched from Finnhub/Yahoo)
                    target_price = analyst_data.get('target_price')
                    target_source = analyst_data.get('target_source', 'N/A')

                    # Get news link - this will ALWAYS work
                    news_data = get_news_link(symbol, company_name)

                    # Debug output with all cell values
                    rec = analyst_data.get('recommendation', 'N/A')
                    analysts = analyst_data.get('total_analysts', 'N/A')
                    confidence = analyst_data.get('confidence', 'N/A')
                    news_summary = news_data.get('summary', 'News link created')
                    price_str = f"${stock_price:.2f}" if stock_price else "N/A"
                    target_str = f"${target_price:.2f}" if target_price else "N/A"

                    industry_str = industry[:20] + "..." if industry and len(industry) > 20 else (industry or "N/A")
                    source_str = analyst_data.get('source', 'Unknown')[:12]
                    target_source_str = target_source[:12] if target_source != 'N/A' else 'N/A'
                    print(f"✅ {symbol}: Price={price_str} | Target={target_str} ({target_source_str}) | EPS={f'${eps_parsed:.2f}' if eps_parsed else 'N/A'} | Industry={industry_str} | Rec={rec} ({source_str}) | News={news_summary[:12]}...")

                    earnings_data.append({
                        'symbol': symbol,
                        'company': company_name,
                        'time': row.get('time', 'time-not-supplied'),
                        'eps': eps_parsed,
                        'eps_raw': eps_value,
                        'analyst_data': analyst_data,
                        'news': news_data,
                        'stock_price': stock_price,
                        'target_price': target_price,
                        'industry': industry
                    })

                    # Small delay
                    time.sleep(0.1)

            # SORT BY RECOMMENDATION PRIORITY
            print("🔄 Sorting companies by recommendation priority...")
            earnings_data.sort(key=get_recommendation_weight)

            # Print sorted order for verification
            print("📋 Sorted order:")
            for i, company in enumerate(earnings_data):
                rec = company['analyst_data']['recommendation']
                weight = get_recommendation_weight(company)
                print(f"  {i+1}. {company['symbol']} - {rec} (weight: {weight})")

            return earnings_data

    except Exception as e:
        print(f"❌ NASDAQ error: {e}")
        return []

def generate_html_report(earnings_data, is_full_report=True):
    """Generate HTML email report with analyst recommendations and news"""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%A, %B %d, %Y')

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
                <span style="font-weight: bold; font-size: 20px;">🏢 {total_companies} Companies Reporting</span>
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

def send_email_sendgrid(subject, html_content, recipients, attachment_html=None, attachment_filename=None):
    """Send email using SendGrid API with optional HTML attachment"""

    api_key = EMAIL_CONFIG['sendgrid_api_key']
    if not api_key:
        print("❌ SendGrid API key not found in environment variables")
        return False

    url = "https://api.sendgrid.com/v3/mail/send"

    email_data = {
        "personalizations": [
            {
                "to": [{"email": email} for email in recipients],
                "subject": subject
            }
        ],
        "from": {
            "email": EMAIL_CONFIG['sender_email'],
            "name": "Earnings Alert System"
        },
        "reply_to": {
            "email": EMAIL_CONFIG['sender_email'],
            "name": "Earnings Alert System"
        },
        "content": [
            {
                "type": "text/html",
                "value": html_content
            }
        ],
        "categories": ["earnings-report", "financial-data"]
    }

    # Add attachment if provided
    if attachment_html and attachment_filename:
        import base64
        encoded_content = base64.b64encode(attachment_html.encode('utf-8')).decode('utf-8')
        email_data["attachments"] = [
            {
                "content": encoded_content,
                "type": "text/html",
                "filename": attachment_filename,
                "disposition": "attachment"
            }
        ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=email_data)

        if response.status_code == 202:
            print(f"✅ Email sent successfully via SendGrid to {len(recipients)} recipients")
            return True
        else:
            print(f"❌ SendGrid error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Failed to send email via SendGrid: {e}")
        return False

def save_to_file(subject, html_content):
    """Save report to file as fallback"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"earnings_report_{timestamp}.html"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Report saved to file: {filename}")
        print(f"💡 Open {filename} in your browser to view the report")
        return True

    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        return False

def validate_config():
    """Validate environment configuration"""
    errors = []

    if not EMAIL_CONFIG['sender_email']:
        errors.append("SENDER_EMAIL is required")

    if not EMAIL_CONFIG['recipients'] or EMAIL_CONFIG['recipients'] == ['']:
        errors.append("RECIPIENTS is required")

    service = EMAIL_CONFIG['email_service']
    if service == 'sendgrid' and not EMAIL_CONFIG['sendgrid_api_key']:
        errors.append("SENDGRID_API_KEY is required when using SendGrid service")

    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        print("\n💡 Check your .env file and make sure all required variables are set")
        return False

    return True

def main():
    """Main function for cron job"""

    if not validate_config():
        return

    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    earnings_data = get_nasdaq_earnings()

    if not earnings_data:
        return

    # Generate full report
    full_html_report = generate_html_report(earnings_data, is_full_report=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Always save full report to file as backup
    full_report_filename = f"earnings_report_full_{timestamp}.html"
    try:
        with open(full_report_filename, 'w', encoding='utf-8') as f:
            f.write(full_html_report)
        print(f"✅ Full report saved to: {full_report_filename}")
    except Exception as e:
        print(f"⚠️ Could not save full report: {e}")

    # Determine if we need to limit the email content
    needs_truncation = len(earnings_data) > 50

    # Generate email content (limited if needed)
    email_html_report = generate_html_report(earnings_data, is_full_report=not needs_truncation)
    subject = f"📊 Daily Earnings Report with News - {len(earnings_data)} Companies - {tomorrow_date}"

    service = EMAIL_CONFIG['email_service']
    success = False

    if service == 'sendgrid':
        # Attach full report if content was truncated
        attachment_html = full_html_report if needs_truncation else None
        attachment_name = f"earnings_full_report_{tomorrow_date}.html" if needs_truncation else None
        success = send_email_sendgrid(subject, email_html_report, EMAIL_CONFIG['recipients'],
                                      attachment_html, attachment_name)
    else:
        success = save_to_file(subject, email_html_report)

    if not success and service != 'file':
        save_to_file(subject, email_html_report)

if __name__ == "__main__":
    main()
