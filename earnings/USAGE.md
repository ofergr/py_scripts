# Earnings Script Usage

## Overview
The earnings script fetches scheduled earnings reports from NASDAQ for a specified date, along with analyst recommendations, stock prices, and news links.

## Command Line Usage

### Basic Syntax
```bash
python3 earnings.py [--days DAYS]
```

### Parameters
- `--days`: Number of days from today to fetch earnings (default: 1 for tomorrow)
  - Use `0` for today
  - Use `1` for tomorrow (default)
  - Use `3` for 3 days from now (e.g., Monday if today is Friday)
  - Any positive integer is valid

### Examples

#### Fetch earnings for tomorrow (default)
```bash
python3 earnings.py
# or explicitly:
python3 earnings.py --days 1
```

#### Fetch earnings for today
```bash
python3 earnings.py --days 0
```

#### Fetch earnings for Monday (if today is Friday)
```bash
python3 earnings.py --days 3
```

#### Fetch earnings for next week
```bash
python3 earnings.py --days 7
```

#### Show help
```bash
python3 earnings.py --help
```

## Output
The script will:
1. Display the target date being queried
2. Fetch all companies reporting earnings on that date
3. Retrieve analyst recommendations from Finnhub
4. Get current stock prices from Yahoo Finance
5. Generate news links for each company
6. Sort companies by recommendation priority (Strong Buy → Buy → Hold → Sell → Strong Sell)
7. Generate an HTML report (saved to file)
8. Send the report via email (if configured)

## Features
- ⚡ Fast async parallel processing (~0.01s per company)
- 📊 Real analyst recommendations from Finnhub
- 💰 Current stock prices from Yahoo Finance
- 📰 Curated news links for each stock
- 📧 Email delivery via SendGrid
- 📱 Mobile-responsive HTML reports
- 🎯 Smart sorting by analyst recommendation
