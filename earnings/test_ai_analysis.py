#!/usr/bin/env python3
"""Tests for AI analysis integration in earnings.py.

Unit tests use mocking — no real Ollama or network calls needed.
Integration test (test_real_*) hits real Ollama + yfinance + sends a test email.

Run all:        python3 -m pytest test_ai_analysis.py -v
Run unit only:  python3 -m pytest test_ai_analysis.py -v -k "not real"
Run e2e only:   python3 -m pytest test_ai_analysis.py -v -k "real"
"""

import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import the functions under test
from earnings import (
    _build_ai_prompt_data,
    _parse_ai_response,
    _validate_ai_result,
    get_ai_fallback_data,
    get_ai_analysis_async,
    get_comprehensive_yfinance_data,
    apply_filters,
    fetch_market_data,
    run_ai_analysis,
    generate_html_report,
    send_email_gmail,
    EMAIL_CONFIG,
    OLLAMA_CONFIG,
    FILTER_CONFIG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_market_data():
    """Full market data dict as returned by get_comprehensive_yfinance_data."""
    return {
        'current_price': 150.0,
        'market_cap': 2_500_000_000_000,
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'trailing_pe': 28.5,
        'forward_pe': 25.0,
        'price_to_book': 45.0,
        'price_to_sales': 7.5,
        'profit_margins': 0.26,
        'gross_margins': 0.46,
        'operating_margins': 0.30,
        'return_on_equity': 1.47,
        'revenue_growth': 0.08,
        'earnings_growth': 0.12,
        'earnings_quarterly_growth': 0.15,
        'eps_trailing': 6.50,
        'eps_forward': 7.20,
        'fifty_two_week_high': 180.0,
        'fifty_two_week_low': 120.0,
        'fifty_day_average': 148.0,
        'two_hundred_day_average': 140.0,
        'beta': 1.2,
        'debt_to_equity': 150.0,
        'current_ratio': 1.1,
        'free_cashflow': 100_000_000_000,
        'dividend_yield': 0.005,
        'yahoo_target_mean': 175.0,
        'price_history_30d': [
            {'date': '2026-01-12', 'close': 140.0, 'volume': 50_000_000},
            {'date': '2026-01-26', 'close': 145.0, 'volume': 55_000_000},
            {'date': '2026-02-09', 'close': 150.0, 'volume': 60_000_000},
        ],
    }


@pytest.fixture
def minimal_market_data():
    """Market data with many None values."""
    return {
        'current_price': 50.0,
        'market_cap': 5_000_000_000,
        'sector': None,
        'industry': None,
        'trailing_pe': None,
        'forward_pe': None,
        'price_to_book': None,
        'price_to_sales': None,
        'profit_margins': None,
        'gross_margins': None,
        'operating_margins': None,
        'return_on_equity': None,
        'revenue_growth': None,
        'earnings_growth': None,
        'earnings_quarterly_growth': None,
        'eps_trailing': None,
        'eps_forward': None,
        'fifty_two_week_high': None,
        'fifty_two_week_low': None,
        'fifty_day_average': None,
        'two_hundred_day_average': None,
        'beta': None,
        'debt_to_equity': None,
        'current_ratio': None,
        'free_cashflow': None,
        'dividend_yield': None,
        'yahoo_target_mean': None,
        'price_history_30d': [],
    }


# ---------------------------------------------------------------------------
# Tests: _parse_ai_response
# ---------------------------------------------------------------------------

class TestParseAiResponse:

    def test_valid_json(self):
        content = json.dumps({
            "recommendation": "Buy",
            "confidence": "High",
            "target_price": 180.0,
            "reasoning": "Strong growth metrics."
        })
        result = _parse_ai_response(content, "AAPL")
        assert result is not None
        assert result['recommendation'] == 'Buy'
        assert result['confidence'] == 'High'
        assert result['target_price'] == 180.0
        assert result['ai_reasoning'] == 'Strong growth metrics.'

    def test_json_in_markdown_code_block(self):
        content = '```json\n{"recommendation": "Strong Buy", "confidence": "Medium", "target_price": 200.0, "reasoning": "Excellent fundamentals."}\n```'
        result = _parse_ai_response(content, "MSFT")
        assert result is not None
        assert result['recommendation'] == 'Strong Buy'
        assert result['confidence'] == 'Medium'

    def test_json_embedded_in_text(self):
        content = 'Here is my analysis:\n{"recommendation": "Hold", "confidence": "Low", "target_price": 50.0, "reasoning": "Mixed signals."}\nThank you.'
        result = _parse_ai_response(content, "XYZ")
        assert result is not None
        assert result['recommendation'] == 'Hold'

    def test_empty_input(self):
        assert _parse_ai_response("", "AAPL") is None
        assert _parse_ai_response(None, "AAPL") is None
        assert _parse_ai_response("   ", "AAPL") is None

    def test_garbage_input(self):
        assert _parse_ai_response("this is not json at all", "AAPL") is None

    def test_json_missing_recommendation_key(self):
        content = json.dumps({"confidence": "High", "target_price": 100.0})
        result = _parse_ai_response(content, "AAPL")
        assert result is not None
        assert result['recommendation'] == 'Hold'


# ---------------------------------------------------------------------------
# Tests: _validate_ai_result
# ---------------------------------------------------------------------------

class TestValidateAiResult:

    def test_valid_values_pass_through(self):
        data = {
            "recommendation": "Strong Buy",
            "confidence": "High",
            "target_price": 200.0,
            "reasoning": "Great company."
        }
        result = _validate_ai_result(data, "AAPL")
        assert result['recommendation'] == 'Strong Buy'
        assert result['confidence'] == 'High'
        assert result['target_price'] == 200.0
        assert result['source'] == 'AI Analysis (Ollama)'

    def test_case_normalization(self):
        data = {"recommendation": "strong buy", "confidence": "high"}
        result = _validate_ai_result(data, "AAPL")
        assert result['recommendation'] == 'Strong Buy'
        assert result['confidence'] == 'High'

    def test_confidence_case_normalization(self):
        """Lowercase confidence values from the AI model should be normalized."""
        for raw, expected in [("high", "High"), ("medium", "Medium"), ("low", "Low"),
                              ("HIGH", "High"), ("Low", "Low"), (" High ", "High")]:
            data = {"recommendation": "Buy", "confidence": raw}
            result = _validate_ai_result(data, "TEST")
            assert result['confidence'] == expected, f"confidence '{raw}' should become '{expected}', got '{result['confidence']}'"

    def test_invalid_confidence_defaults_to_medium(self):
        """Truly invalid confidence values should still default to Medium."""
        for bad in ["Very High", "Super Low", "maybe", "", 123]:
            data = {"recommendation": "Buy", "confidence": bad}
            result = _validate_ai_result(data, "TEST")
            assert result['confidence'] == 'Medium', f"confidence '{bad}' should default to 'Medium', got '{result['confidence']}'"

    def test_neutral_maps_to_hold(self):
        data = {"recommendation": "neutral"}
        result = _validate_ai_result(data, "AAPL")
        assert result['recommendation'] == 'Hold'

    def test_invalid_recommendation_defaults_to_hold(self):
        data = {"recommendation": "maybe buy"}
        result = _validate_ai_result(data, "AAPL")
        assert result['recommendation'] == 'Hold'

    def test_negative_target_price_set_to_none(self):
        data = {"recommendation": "Buy", "target_price": -50.0}
        result = _validate_ai_result(data, "AAPL")
        assert result['target_price'] is None

    def test_zero_target_price_set_to_none(self):
        data = {"recommendation": "Buy", "target_price": 0}
        result = _validate_ai_result(data, "AAPL")
        assert result['target_price'] is None

    def test_non_numeric_target_price_set_to_none(self):
        data = {"recommendation": "Buy", "target_price": "not a number"}
        result = _validate_ai_result(data, "AAPL")
        assert result['target_price'] is None

    def test_total_analysts_is_one(self):
        data = {"recommendation": "Buy"}
        result = _validate_ai_result(data, "AAPL")
        assert result['total_analysts'] == 1

    def test_all_valid_recommendations(self):
        for rec in ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]:
            data = {"recommendation": rec}
            result = _validate_ai_result(data, "TEST")
            assert result['recommendation'] == rec

    def test_reasoning_string_short_passes_through(self):
        data = {"recommendation": "Buy", "reasoning": "Strong growth and margins."}
        result = _validate_ai_result(data, "TEST")
        assert result['ai_reasoning'] == "Strong growth and margins."

    def test_reasoning_string_truncated_at_300_chars(self):
        long_reasoning = "A" * 400
        data = {"recommendation": "Buy", "reasoning": long_reasoning}
        result = _validate_ai_result(data, "TEST")
        assert len(result['ai_reasoning']) == 300
        assert result['ai_reasoning'].endswith('...')

    def test_reasoning_list_joined_and_truncated(self):
        reasoning_list = ["Point one about fundamentals." * 5, "Point two about technicals." * 5]
        data = {"recommendation": "Buy", "reasoning": reasoning_list}
        result = _validate_ai_result(data, "TEST")
        assert ' | ' in result['ai_reasoning'] or len(result['ai_reasoning']) <= 300
        assert len(result['ai_reasoning']) <= 300

    def test_reasoning_exactly_300_chars_not_truncated(self):
        reasoning = "A" * 300
        data = {"recommendation": "Buy", "reasoning": reasoning}
        result = _validate_ai_result(data, "TEST")
        assert result['ai_reasoning'] == reasoning
        assert len(result['ai_reasoning']) == 300


# ---------------------------------------------------------------------------
# Tests: _build_ai_prompt_data
# ---------------------------------------------------------------------------

class TestBuildAiPromptData:

    def test_contains_key_fields(self, sample_market_data):
        prompt = _build_ai_prompt_data("AAPL", "Apple Inc.", sample_market_data, 1.50)
        assert "AAPL" in prompt
        assert "Apple Inc." in prompt
        assert "Technology" in prompt
        assert "$150.00" in prompt
        assert "Trailing P/E" in prompt
        assert "EPS Forecast" in prompt
        assert "$1.50" in prompt

    def test_handles_none_values(self, minimal_market_data):
        prompt = _build_ai_prompt_data("XYZ", "Unknown Corp", minimal_market_data, None)
        assert "XYZ" in prompt
        assert "N/A" in prompt

    def test_includes_price_trend(self, sample_market_data):
        prompt = _build_ai_prompt_data("AAPL", "Apple Inc.", sample_market_data, None)
        assert "30-DAY PRICE TREND" in prompt
        assert "30-Day Change" in prompt
        assert "Upward" in prompt

    def test_no_price_trend_without_history(self, minimal_market_data):
        prompt = _build_ai_prompt_data("XYZ", "Unknown Corp", minimal_market_data, None)
        assert "30-DAY PRICE TREND" not in prompt

    def test_sideways_trend(self, sample_market_data):
        sample_market_data['price_history_30d'] = [
            {'date': '2026-01-12', 'close': 150.0, 'volume': 50_000_000},
            {'date': '2026-02-09', 'close': 151.0, 'volume': 60_000_000},
        ]
        prompt = _build_ai_prompt_data("AAPL", "Apple Inc.", sample_market_data, None)
        assert "Sideways" in prompt


# ---------------------------------------------------------------------------
# Tests: get_ai_fallback_data
# ---------------------------------------------------------------------------

class TestGetAiFallbackData:

    def test_returns_hold_with_low_confidence(self):
        result = get_ai_fallback_data("AAPL")
        assert result['recommendation'] == 'Hold'
        assert result['confidence'] == 'Low'
        assert result['total_analysts'] == 0
        assert 'AI Fallback' in result['source']

    def test_uses_yahoo_target_from_market_data(self, sample_market_data):
        result = get_ai_fallback_data("AAPL", sample_market_data)
        assert result['target_price'] == 175.0
        assert result['target_source'] == 'Yahoo Finance'

    def test_handles_none_market_data(self):
        result = get_ai_fallback_data("AAPL", None)
        assert result['target_price'] is None
        assert result['target_source'] == 'N/A'

    def test_has_ai_reasoning(self):
        result = get_ai_fallback_data("AAPL")
        assert result['ai_reasoning'] != ''


# ---------------------------------------------------------------------------
# Tests: get_ai_analysis_async (mocked Ollama)
# ---------------------------------------------------------------------------

class TestGetAiAnalysisAsync:

    @pytest.mark.asyncio
    async def test_successful_ollama_response(self, sample_market_data):
        ai_response = json.dumps({
            "recommendation": "Buy",
            "confidence": "High",
            "target_price": 180.0,
            "reasoning": "Strong fundamentals and growth."
        })

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'message': {'content': ai_response}
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.dict(OLLAMA_CONFIG, {'enabled': True, 'max_retries': 0, 'timeout': 10, 'base_url': 'http://localhost:11434', 'model': 'test-model'}):
            result = await get_ai_analysis_async(
                mock_session, "AAPL", "Apple Inc.", sample_market_data, 1.50
            )

        assert result['recommendation'] == 'Buy'
        assert result['confidence'] == 'High'
        assert result['target_price'] == 180.0
        assert result['market_cap'] == sample_market_data['market_cap']

    @pytest.mark.asyncio
    async def test_ollama_disabled_returns_fallback(self, sample_market_data):
        mock_session = MagicMock()

        with patch.dict(OLLAMA_CONFIG, {'enabled': False}):
            result = await get_ai_analysis_async(
                mock_session, "AAPL", "Apple Inc.", sample_market_data, 1.50
            )

        assert result['recommendation'] == 'Hold'
        assert 'Fallback' in result['source']

    @pytest.mark.asyncio
    async def test_none_market_data_returns_fallback(self):
        mock_session = MagicMock()

        with patch.dict(OLLAMA_CONFIG, {'enabled': True}):
            result = await get_ai_analysis_async(
                mock_session, "AAPL", "Apple Inc.", None, 1.50
            )

        assert result['recommendation'] == 'Hold'
        assert 'Fallback' in result['source']

    @pytest.mark.asyncio
    async def test_ollama_timeout_falls_back(self, sample_market_data):
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.dict(OLLAMA_CONFIG, {'enabled': True, 'max_retries': 1, 'timeout': 1, 'base_url': 'http://localhost:11434', 'model': 'test-model'}):
            result = await get_ai_analysis_async(
                mock_session, "AAPL", "Apple Inc.", sample_market_data, 1.50
            )

        assert result['recommendation'] == 'Hold'
        assert 'Fallback' in result['source']

    @pytest.mark.asyncio
    async def test_ollama_invalid_json_falls_back(self, sample_market_data):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'message': {'content': 'This is not valid JSON at all'}
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.dict(OLLAMA_CONFIG, {'enabled': True, 'max_retries': 0, 'timeout': 10, 'base_url': 'http://localhost:11434', 'model': 'test-model'}):
            result = await get_ai_analysis_async(
                mock_session, "AAPL", "Apple Inc.", sample_market_data, 1.50
            )

        assert result['recommendation'] == 'Hold'
        assert 'Fallback' in result['source']

    @pytest.mark.asyncio
    async def test_ollama_non_200_status_falls_back(self, sample_market_data):
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.dict(OLLAMA_CONFIG, {'enabled': True, 'max_retries': 0, 'timeout': 10, 'base_url': 'http://localhost:11434', 'model': 'test-model'}):
            result = await get_ai_analysis_async(
                mock_session, "AAPL", "Apple Inc.", sample_market_data, 1.50
            )

        assert result['recommendation'] == 'Hold'
        assert 'Fallback' in result['source']


# ---------------------------------------------------------------------------
# Tests: apply_filters with AI mode
# ---------------------------------------------------------------------------

class TestApplyFiltersAiMode:

    def _make_company(self, symbol, market_cap, analyst_count, stock_price):
        return {
            'symbol': symbol,
            'market_cap': market_cap,
            'stock_price': stock_price,
            'analyst_data': {
                'recommendation': 'Buy',
                'total_analysts': analyst_count,
            },
        }

    def test_ai_enabled_skips_analyst_filter(self):
        index_cache = {
            'sp500': {'AAPL'},
            'nasdaq100': set(),
            'russell2000': set(),
            'last_updated': None,
        }
        companies = [
            self._make_company('AAPL', 2_000_000_000_000, 0, 150.0),
        ]

        with patch.dict(OLLAMA_CONFIG, {'enabled': True}):
            filtered, stats = apply_filters(companies, index_cache)

        assert len(filtered) == 1
        assert stats['failed_analyst_coverage'] == 0

    def test_ai_disabled_applies_analyst_filter(self):
        index_cache = {
            'sp500': {'AAPL'},
            'nasdaq100': set(),
            'russell2000': set(),
            'last_updated': None,
        }
        companies = [
            self._make_company('AAPL', 2_000_000_000_000, 0, 150.0),
        ]

        with patch.dict(OLLAMA_CONFIG, {'enabled': False}), \
             patch.dict(FILTER_CONFIG, {'min_analysts': 5}):
            filtered, stats = apply_filters(companies, index_cache)

        assert len(filtered) == 0
        assert stats['failed_analyst_coverage'] == 1


# ---------------------------------------------------------------------------
# Tests: HTML report — text must not overflow its container
# ---------------------------------------------------------------------------

class TestHtmlReportOverflow:

    def _make_company(self, ai_reasoning):
        return {
            'symbol': 'TEST',
            'company': 'Test Corp',
            'time': 'time-pre-market',
            'eps': 1.50,
            'eps_raw': '$1.50',
            'analyst_data': {
                'recommendation': 'Buy',
                'total_analysts': 1,
                'source': 'AI Analysis (Ollama)',
                'confidence': 'High',
                'target_price': 200.0,
                'target_source': 'AI Analysis',
                'market_cap': 5_000_000_000,
                'ai_reasoning': ai_reasoning,
            },
            'news': {
                'summary': 'Test news',
                'url': 'https://example.com',
                'title': 'Test',
            },
            'stock_price': 150.0,
            'target_price': 200.0,
            'industry': 'Technology',
            'market_cap': 5_000_000_000,
            'logo_url': None,
        }

    def test_table_uses_fixed_layout(self):
        """Table must use table-layout:fixed so columns respect width percentages."""
        company = self._make_company("Short reasoning.")
        html = generate_html_report([company], "Monday, Feb 10, 2026", is_full_report=True)
        assert 'table-layout: fixed' in html, "Table must use table-layout: fixed to prevent content overflow"

    def test_ai_insight_wraps_long_text(self):
        """AI insight div must have word-wrap so long text stays within the table."""
        long_reasoning = "A" * 500 + " " + "B" * 500  # very long AI reasoning
        company = self._make_company(long_reasoning)
        html = generate_html_report([company], "Monday, Feb 10, 2026", is_full_report=True)
        assert 'word-wrap: break-word' in html, "AI insight must use word-wrap: break-word"
        assert long_reasoning in html, "Full AI reasoning should appear in the report"


# ---------------------------------------------------------------------------
# Tests: fetch_market_data + run_ai_analysis two-phase architecture (mocked)
# ---------------------------------------------------------------------------

class TestTwoPhaseArchitecture:

    @pytest.mark.asyncio
    async def test_fetch_market_data_returns_basic_info(self):
        row = {'symbol': 'AAPL', 'name': 'Apple Inc.', 'epsForecast': '$1.50', 'time': 'time-pre-market'}
        api_sem = asyncio.Semaphore(50)

        mock_market_data = {
            'current_price': 150.0,
            'market_cap': 2_500_000_000_000,
            'sector': 'Technology',
            'yahoo_target_mean': 175.0,
            'price_history_30d': [],
        }

        mock_session = MagicMock()

        with patch('earnings.get_comprehensive_yfinance_data_async', new_callable=AsyncMock, return_value=mock_market_data), \
             patch('earnings.get_company_logo_async', new_callable=AsyncMock, return_value=None):

            result = await fetch_market_data(mock_session, row, api_sem)

        assert result['symbol'] == 'AAPL'
        assert result['stock_price'] == 150.0
        assert result['market_cap'] == 2_500_000_000_000
        assert result['_market_data'] is mock_market_data
        assert 'analyst_data' not in result  # no AI yet

    @pytest.mark.asyncio
    async def test_run_ai_analysis_adds_analyst_data(self):
        company = {
            'symbol': 'AAPL',
            'company': 'Apple Inc.',
            'eps': 1.50,
            'market_cap': 2_500_000_000_000,
            '_market_data': {'current_price': 150.0, 'market_cap': 2_500_000_000_000},
        }
        ollama_sem = asyncio.Semaphore(1)

        ai_result = {
            'recommendation': 'Buy',
            'total_analysts': 1,
            'source': 'AI Analysis (Ollama)',
            'confidence': 'High',
            'target_price': 180.0,
            'target_source': 'AI Analysis',
            'market_cap': 2_500_000_000_000,
            'ai_reasoning': 'Strong fundamentals.',
        }

        mock_session = MagicMock()

        with patch.dict(OLLAMA_CONFIG, {'enabled': True}), \
             patch('earnings.get_ai_analysis_async', new_callable=AsyncMock, return_value=ai_result), \
             patch('earnings.get_news_link', return_value={'summary': 'test', 'url': 'http://test.com', 'title': 'test'}):

            result = await run_ai_analysis(mock_session, company, ollama_sem)

        assert result['analyst_data']['recommendation'] == 'Buy'
        assert result['analyst_data']['source'] == 'AI Analysis (Ollama)'

    @pytest.mark.asyncio
    async def test_run_ai_falls_back_to_finnhub_when_disabled(self):
        company = {
            'symbol': 'AAPL',
            'company': 'Apple Inc.',
            'eps': None,
            'market_cap': 2_500_000_000_000,
            '_market_data': {'current_price': 150.0, 'market_cap': 2_500_000_000_000},
        }
        ollama_sem = asyncio.Semaphore(1)

        finnhub_result = {
            'recommendation': 'Strong Buy',
            'total_analysts': 25,
            'source': 'Finnhub Real Data',
            'confidence': 'High',
            'target_price': 190.0,
            'target_source': 'Finnhub Real Data',
            'market_cap': 2_500_000_000_000,
        }

        mock_session = MagicMock()

        with patch.dict(OLLAMA_CONFIG, {'enabled': False}), \
             patch('earnings.get_real_analyst_data_async', new_callable=AsyncMock, return_value=finnhub_result), \
             patch('earnings.get_news_link', return_value={'summary': 'test', 'url': 'http://test.com', 'title': 'test'}):

            result = await run_ai_analysis(mock_session, company, ollama_sem)

        assert result['analyst_data']['recommendation'] == 'Strong Buy'
        assert result['analyst_data']['source'] == 'Finnhub Real Data'


# ===========================================================================
# REAL FILTER TEST — fetches real data, applies filters, no AI
#
# Hits real services: NASDAQ API, yfinance, Wikipedia
# Shows exactly how many companies survive filtering
#
# Run:  python3 -m pytest test_ai_analysis.py -v -k "real_filter" -s
# ===========================================================================

class TestRealFilter:

    @pytest.mark.asyncio
    async def test_real_filter_count(self):
        """Fetch today's earnings, market data, and apply filters — no AI."""
        import aiohttp
        from datetime import datetime
        from earnings import (
            get_major_index_constituents_async,
            fetch_market_data,
            apply_filters,
        )

        target_date = datetime.now().strftime('%Y-%m-%d')
        api_url = f"https://api.nasdaq.com/api/calendar/earnings?date={target_date}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*'
        }

        async with aiohttp.ClientSession() as session:
            # Fetch index constituents
            print(f"\n[FILTER] Fetching index constituents...")
            index_cache = await get_major_index_constituents_async(session)
            sp500_count = len(index_cache['sp500']) if index_cache['sp500'] else 0
            nasdaq100_count = len(index_cache['nasdaq100']) if index_cache['nasdaq100'] else 0
            print(f"[FILTER]   S&P 500: {sp500_count} symbols")
            print(f"[FILTER]   NASDAQ 100: {nasdaq100_count} symbols")

            # Fetch earnings calendar
            print(f"[FILTER] Fetching NASDAQ earnings for {target_date}...")
            async with session.get(api_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                assert response.status == 200, f"NASDAQ API returned {response.status}"
                data = await response.json()

            rows = data.get('data', {}).get('rows') or []
            print(f"[FILTER] Found {len(rows)} companies reporting earnings")
            assert len(rows) > 0, "No earnings found for today"

            # Fetch market data in parallel (no AI)
            print(f"[FILTER] Fetching market data for all {len(rows)} companies...")
            api_semaphore = asyncio.Semaphore(50)
            tasks = [fetch_market_data(session, row, api_semaphore) for row in rows]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            companies = [r for r in results if not isinstance(r, Exception)]
            print(f"[FILTER] Got market data for {len(companies)} companies")

            # Apply filters
            filtered, stats = apply_filters(companies, index_cache)

        print(f"\n[FILTER] === FILTER RESULTS ===")
        print(f"[FILTER]   Total: {stats['total']}")
        print(f"[FILTER]   Failed market cap (< $1B): {stats['failed_market_cap']}")
        print(f"[FILTER]   Failed analyst coverage: {stats['failed_analyst_coverage']}")
        print(f"[FILTER]   Failed index membership: {stats['failed_index']}")
        print(f"[FILTER]   Failed stock price (< $5): {stats['failed_stock_price']}")
        print(f"[FILTER]   >>> PASSED ALL FILTERS: {stats['passed']} <<<")
        print(f"\n[FILTER] Survivors:")
        for c in filtered:
            cap_b = (c.get('market_cap') or 0) / 1e9
            print(f"[FILTER]   {c['symbol']:6} | {c.get('index',''):25} | ${cap_b:6.1f}B | ${c.get('stock_price', 0):>8.2f}")

        assert stats['passed'] > 0, "No companies passed filters"
        # Sanity check: should be roughly 20-50, not 100+
        print(f"\n[FILTER] AI would analyze {stats['passed']} companies instead of {stats['total']}")


# ===========================================================================
# REAL END-TO-END INTEGRATION TEST
#
# This test hits real services:
#   1. yfinance  — fetches real AAPL market data
#   2. Ollama    — sends data to the local AI model for analysis
#   3. Gmail     — sends a test email with the generated report
#
# Requirements:
#   - Ollama running locally with the configured model
#   - .env configured with email credentials
#
# Run:  python3 -m pytest test_ai_analysis.py -v -k "real"
# ===========================================================================

class TestRealEndToEnd:
    """Real integration test — no mocks. Requires Ollama + email config."""

    @pytest.mark.asyncio
    async def test_real_aapl_analysis_and_email(self):
        """Full pipeline: yfinance -> Ollama AI -> HTML report -> send email."""
        import aiohttp
        from datetime import datetime

        # --- Step 1: Fetch real market data from yfinance ---
        print("\n[E2E] Step 1: Fetching real AAPL data from yfinance...")
        market_data = get_comprehensive_yfinance_data("AAPL")
        assert market_data is not None, "yfinance failed to fetch AAPL data"
        assert market_data['current_price'] is not None, "No current price for AAPL"
        assert market_data['market_cap'] is not None, "No market cap for AAPL"
        print(f"[E2E]   Price: ${market_data['current_price']:.2f}")
        print(f"[E2E]   Market Cap: ${market_data['market_cap']:,.0f}")
        print(f"[E2E]   Sector: {market_data.get('sector')}")
        print(f"[E2E]   30d history points: {len(market_data.get('price_history_30d', []))}")

        # --- Step 2: Send to real Ollama for AI analysis ---
        print("\n[E2E] Step 2: Sending to Ollama for AI analysis...")
        assert OLLAMA_CONFIG['enabled'], (
            "Ollama is not enabled. Make sure Ollama is running: ollama serve"
        )

        async with aiohttp.ClientSession() as session:
            analyst_data = await get_ai_analysis_async(
                session, "AAPL", "Apple Inc.", market_data, 1.50
            )

        assert analyst_data is not None
        assert analyst_data['recommendation'] in {
            'Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'
        }, f"Invalid recommendation: {analyst_data['recommendation']}"
        assert analyst_data['confidence'] in {'High', 'Medium', 'Low'}
        assert analyst_data.get('ai_reasoning'), "AI should provide reasoning"
        print(f"[E2E]   Recommendation: {analyst_data['recommendation']}")
        print(f"[E2E]   Confidence: {analyst_data['confidence']}")
        print(f"[E2E]   Target Price: ${analyst_data.get('target_price', 'N/A')}")
        print(f"[E2E]   Reasoning: {analyst_data.get('ai_reasoning', '')[:150]}")

        # --- Step 3: Build a one-company earnings report ---
        print("\n[E2E] Step 3: Generating HTML report...")
        company_data = {
            'symbol': 'AAPL',
            'company': 'Apple Inc.',
            'time': 'time-pre-market',
            'eps': 1.50,
            'eps_raw': '$1.50',
            'analyst_data': analyst_data,
            'news': {
                'summary': 'Latest AAPL earnings news',
                'url': 'https://finance.yahoo.com/quote/AAPL/news/',
                'title': 'AAPL News',
            },
            'stock_price': market_data['current_price'],
            'target_price': analyst_data.get('target_price'),
            'industry': market_data.get('sector'),
            'market_cap': market_data['market_cap'],
            'logo_url': None,
        }

        today = datetime.now().strftime('%A, %B %d, %Y')
        html_report = generate_html_report([company_data], today, is_full_report=True)
        assert len(html_report) > 500, "HTML report is too short"
        assert 'AAPL' in html_report
        assert analyst_data['recommendation'] in html_report
        print(f"[E2E]   Report size: {len(html_report):,} chars")

        # --- Step 4: Send the real test email ---
        print("\n[E2E] Step 4: Sending test email...")
        assert EMAIL_CONFIG['sender_email'], "SENDER_EMAIL not configured in .env"
        assert EMAIL_CONFIG['recipients'], "RECIPIENTS not configured in .env"

        subject = f"[TEST] AI Earnings Analysis - AAPL - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        success = send_email_gmail(subject, html_report, EMAIL_CONFIG['recipients'])
        assert success, "Email sending failed"
        print(f"[E2E]   Email sent to: {', '.join(EMAIL_CONFIG['recipients'])}")
        print(f"[E2E]   Subject: {subject}")

        print("\n[E2E] End-to-end test PASSED!")
