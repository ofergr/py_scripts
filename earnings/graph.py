import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import random

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Chart Configuration
CHART_WIDTH = 15
CHART_HEIGHT = 12
CHART_DPI = 300
CHART_BACKGROUND = 'white'

# Data Generation
DEFAULT_DATA_DAYS = 80
BASE_PRICES = {'AAPL': 180.0, 'MSFT': 350.0, 'GOOGL': 140.0, 'TSLA': 250.0, 'IBM': 235.0}
DEFAULT_BASE_PRICE = 100.0
VOLUME_MIN = 50000000
VOLUME_MAX = 150000000

# Price Movement Parameters
WEEKLY_VOLATILITY = 0.04
DAILY_VOLATILITY = 0.015
UPTREND_BIAS = 0.002
DOWNTREND_BIAS = -0.001
RECOVERY_BIAS = 0.001
MIN_PRICE_RATIO = 0.5
DAILY_RANGE_MIN = 0.01
DAILY_RANGE_MAX = 0.03
OPEN_POSITION_MIN = 0.3
OPEN_POSITION_MAX = 0.7

# Moving Average Periods
MA_SHORT_PERIOD = 20
MA_LONG_PERIOD = 50

# RSI Configuration
RSI_PERIOD = 7
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL = 50
RSI_MIN_DAYS_REQUIRED = 15

# Default Analysis Parameters
DEFAULT_DISPLAY_DAYS = 20
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "IBM"]

# Chart Layout
PRICE_VOLUME_RSI_RATIOS = [3, 1, 1]
Y_MARGIN_RATIO = 0.05

# Data Periods and Calculations
TREND_PERIOD_DIVISOR = 3
WEEKLY_VOLATILITY_CYCLE = 7

# Line Styling
PRICE_LINE_WIDTH = 3
MA_LINE_WIDTH = 2.5
RSI_LINE_WIDTH = 2.5
REFERENCE_LINE_WIDTH = 1

# Colors
PRICE_COLOR = 'black'
MA_SHORT_COLOR = 'blue'
MA_LONG_COLOR = 'red'
RSI_COLOR = 'purple'
VOLUME_UP_COLOR = 'green'
VOLUME_DOWN_COLOR = 'red'
OVERBOUGHT_COLOR = 'r'
OVERSOLD_COLOR = 'g'
REFERENCE_LINE_COLOR = 'gray'
NEUTRAL_ZONE_COLOR = 'yellow'

# Transparency/Alpha Values
MA_ALPHA = 0.8
VOLUME_ALPHA = 0.7
REFERENCE_LINE_ALPHA = 0.7
NEUTRAL_ZONE_ALPHA = 0.15
GRID_ALPHA = 0.3

# Text and Fonts
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 12
WARNING_FONT_SIZE = 11
TITLE_PAD = 20

# Date Formatting
DATE_FORMAT = '%m/%d'
DATE_INTERVAL_DIVISOR = 8
ROTATION_ANGLE = 45

# Volume Bar Configuration
VOLUME_BAR_WIDTH_HOURS = 15

# Print Formatting
SUMMARY_SEPARATOR_LENGTH = 60
ANALYSIS_SEPARATOR_LENGTH = 80

# =============================================================================
# FUNCTIONS
# =============================================================================

def create_sample_data(symbol, days=DEFAULT_DATA_DAYS):
    """Create sample stock data with more realistic price movements for RSI"""
    random.seed(hash(symbol) % 2**32)
    
    start_price = BASE_PRICES.get(symbol.upper(), DEFAULT_BASE_PRICE)
    current_price = start_price
    stock_data = []
    start_date = datetime.now() - timedelta(days=days-1)
    
    for i in range(days):
        # Create more varied price movements for better RSI calculation
        if i % WEEKLY_VOLATILITY_CYCLE == 0:  # Weekly volatility spikes
            volatility = WEEKLY_VOLATILITY
        else:
            volatility = DAILY_VOLATILITY
            
        # Add some trending periods and reversals
        if i < days // TREND_PERIOD_DIVISOR:
            trend_bias = UPTREND_BIAS  # Uptrend
        elif i < 2 * days // TREND_PERIOD_DIVISOR:
            trend_bias = DOWNTREND_BIAS  # Downtrend
        else:
            trend_bias = RECOVERY_BIAS  # Recovery
            
        noise = random.gauss(0, volatility)
        change = trend_bias + noise
        current_price *= (1 + change)
        
        # Ensure we don't go negative
        if current_price < start_price * MIN_PRICE_RATIO:
            current_price = start_price * MIN_PRICE_RATIO
        
        daily_volatility = current_price * random.uniform(DAILY_RANGE_MIN, DAILY_RANGE_MAX)
        high = current_price + random.uniform(0, daily_volatility)
        low = current_price - random.uniform(0, daily_volatility)
        open_price = low + (high - low) * random.uniform(OPEN_POSITION_MIN, OPEN_POSITION_MAX)
        volume = random.randint(VOLUME_MIN, VOLUME_MAX)
        
        stock_data.append({
            'date': start_date + timedelta(days=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    return stock_data

def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    sma = []
    for i in range(len(prices)):
        if i < window - 1:
            sma.append(None)
        else:
            avg = sum(prices[i - window + 1:i + 1]) / window
            sma.append(avg)
    return sma

def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate RSI - simple working version"""
    if len(prices) < period + 1:
        return [None] * len(prices)
    
    rsi_values = [None] * period
    gains = []
    losses = []
    
    # Initial period
    for i in range(1, period + 1):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    # Calculate RSI for remaining periods
    for i in range(period, len(prices)):
        change = prices[i] - prices[i-1]
        gain = max(change, 0)
        loss = max(-change, 0)
        
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
    
    return rsi_values

def plot_stock_analysis(symbol="AAPL", display_days=DEFAULT_DISPLAY_DAYS):
    """Create comprehensive line chart with technical indicators"""
    
    # Get data
    all_data = create_sample_data(symbol, DEFAULT_DATA_DAYS)
    all_closes = [d['close'] for d in all_data]
    
    # Calculate MAs on full dataset
    ma20_full = calculate_sma(all_closes, MA_SHORT_PERIOD)
    ma50_full = calculate_sma(all_closes, MA_LONG_PERIOD)
    
    # Take last display_days for plotting
    display_data = all_data[-display_days:]
    closes = [d['close'] for d in display_data]
    opens = [d['open'] for d in display_data]
    highs = [d['high'] for d in display_data]
    lows = [d['low'] for d in display_data]
    dates = [d['date'] for d in display_data]
    volumes = [d['volume'] for d in display_data]
    
    ma20 = ma20_full[-display_days:]
    ma50 = ma50_full[-display_days:]
    rsi = calculate_rsi(closes, RSI_PERIOD)
    
    # Create the plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(CHART_WIDTH, CHART_HEIGHT), 
                                       gridspec_kw={'height_ratios': PRICE_VOLUME_RSI_RATIOS})
    
    # Plot 1: Price and Moving Averages
    ax1.plot(dates, closes, label='Close Price', linewidth=PRICE_LINE_WIDTH, color=PRICE_COLOR, zorder=3)
    
    # Plot MA20
    ma20_dates = []
    ma20_values = []
    for i, val in enumerate(ma20):
        if val is not None:
            ma20_dates.append(dates[i])
            ma20_values.append(val)
    
    if ma20_values:
        ax1.plot(ma20_dates, ma20_values, label=f'MA{MA_SHORT_PERIOD}', 
                linewidth=MA_LINE_WIDTH, color=MA_SHORT_COLOR, alpha=MA_ALPHA)
    
    # Plot MA50
    ma50_dates = []
    ma50_values = []
    for i, val in enumerate(ma50):
        if val is not None:
            ma50_dates.append(dates[i])
            ma50_values.append(val)
    
    if ma50_values:
        ax1.plot(ma50_dates, ma50_values, label=f'MA{MA_LONG_PERIOD}', 
                linewidth=MA_LINE_WIDTH, color=MA_LONG_COLOR, alpha=MA_ALPHA)
    
    # Set Y-axis limits to show everything properly
    all_y_values = closes[:]
    all_y_values.extend(ma20_values)
    all_y_values.extend(ma50_values)
    
    y_min = min(all_y_values)
    y_max = max(all_y_values)
    y_range = y_max - y_min
    ax1.set_ylim(y_min - y_range * Y_MARGIN_RATIO, y_max + y_range * Y_MARGIN_RATIO)
    
    ax1.set_title(f'{symbol} - Stock Analysis (Last {display_days} Days)', 
                 fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=TITLE_PAD)
    ax1.set_ylabel('Price ($)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax1.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
    ax1.grid(True, alpha=GRID_ALPHA)
    
    # Plot 2: Volume
    colors = [VOLUME_UP_COLOR if closes[i] >= opens[i] else VOLUME_DOWN_COLOR 
              for i in range(len(closes))]
    ax2.bar(dates, volumes, alpha=VOLUME_ALPHA, color=colors, 
           width=timedelta(hours=VOLUME_BAR_WIDTH_HOURS))
    ax2.set_ylabel('Volume', fontsize=AXIS_LABEL_FONT_SIZE)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2.grid(True, alpha=GRID_ALPHA)
    
    # Plot 3: RSI
    rsi_dates = []
    rsi_values = []
    for i, val in enumerate(rsi):
        if val is not None:
            rsi_dates.append(dates[i])
            rsi_values.append(val)
    
    # Always show the RSI framework
    ax3.axhline(y=RSI_OVERBOUGHT, color=OVERBOUGHT_COLOR, linestyle='--', 
               alpha=REFERENCE_LINE_ALPHA, label=f'Overbought ({RSI_OVERBOUGHT})')
    ax3.axhline(y=RSI_OVERSOLD, color=OVERSOLD_COLOR, linestyle='--', 
               alpha=REFERENCE_LINE_ALPHA, label=f'Oversold ({RSI_OVERSOLD})')
    ax3.axhline(y=RSI_NEUTRAL, color=REFERENCE_LINE_COLOR, linestyle='-', 
               alpha=REFERENCE_LINE_ALPHA, linewidth=REFERENCE_LINE_WIDTH)
    
    if len(rsi_values) > 1:
        ax3.plot(rsi_dates, rsi_values, color=RSI_COLOR, linewidth=RSI_LINE_WIDTH, label='RSI')
        ax3.fill_between(rsi_dates, RSI_OVERSOLD, RSI_OVERBOUGHT, 
                        alpha=NEUTRAL_ZONE_ALPHA, color=NEUTRAL_ZONE_COLOR)
        print(f"RSI plotted successfully with {len(rsi_values)} points")
    elif display_days < RSI_MIN_DAYS_REQUIRED:
        ax3.fill_between([dates[0], dates[-1]], RSI_OVERSOLD, RSI_OVERBOUGHT, 
                        alpha=NEUTRAL_ZONE_ALPHA, color=NEUTRAL_ZONE_COLOR)
        ax3.text(dates[len(dates)//2], RSI_NEUTRAL, 
                f'Need {RSI_MIN_DAYS_REQUIRED}+ days for RSI\n(Currently {display_days} days)', 
                ha='center', va='center', fontsize=WARNING_FONT_SIZE, color='red', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        print(f"Warning: Need at least {RSI_MIN_DAYS_REQUIRED} days for RSI calculation, got {display_days}")
    else:
        ax3.fill_between([dates[0], dates[-1]], RSI_OVERSOLD, RSI_OVERBOUGHT, 
                        alpha=NEUTRAL_ZONE_ALPHA, color=NEUTRAL_ZONE_COLOR)
        ax3.text(dates[len(dates)//2], RSI_NEUTRAL, 'RSI calculation failed', 
                ha='center', va='center', fontsize=WARNING_FONT_SIZE, color='red')
        print("RSI calculation failed despite having enough data")
    
    ax3.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)
    ax3.set_ylabel('RSI', fontsize=AXIS_LABEL_FONT_SIZE)
    ax3.set_xlabel('Date', fontsize=AXIS_LABEL_FONT_SIZE)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=GRID_ALPHA)
    
    # Format dates for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//DATE_INTERVAL_DIVISOR)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=ROTATION_ANGLE)
    
    plt.tight_layout()
    
    filename = f"{symbol}_technical_analysis.png"
    plt.savefig(filename, dpi=CHART_DPI, bbox_inches='tight', facecolor=CHART_BACKGROUND)
    print(f"Chart saved as: {filename}")
    
    # Print analysis summary
    current_price = closes[-1]
    start_price = closes[0]
    price_change = current_price - start_price
    percent_change = (price_change / start_price) * 100
    
    print(f"\n{'='*SUMMARY_SEPARATOR_LENGTH}")
    print(f"{symbol} TECHNICAL ANALYSIS SUMMARY")
    print('='*SUMMARY_SEPARATOR_LENGTH)
    print(f"Current Price: ${current_price:.2f}")
    print(f"Period Start: ${start_price:.2f}")
    print(f"Price Change: ${price_change:.2f} ({percent_change:.2f}%)")
    print(f"Period High: ${max(highs):.2f}")
    print(f"Period Low: ${min(lows):.2f}")
    print(f"Average Volume: {sum(volumes)/len(volumes):,.0f}")
    
    # Moving Average Analysis
    print(f"\nMoving Average Analysis:")
    if ma20_values and ma20[-1] is not None:
        ma20_current = ma20[-1]
        ma20_signal = "BULLISH" if current_price > ma20_current else "BEARISH"
        print(f"  MA{MA_SHORT_PERIOD}: ${ma20_current:.2f} - Price vs MA{MA_SHORT_PERIOD}: {ma20_signal}")
    
    if ma50_values and ma50[-1] is not None:
        ma50_current = ma50[-1]
        ma50_signal = "BULLISH" if current_price > ma50_current else "BEARISH"
        print(f"  MA{MA_LONG_PERIOD}: ${ma50_current:.2f} - Price vs MA{MA_LONG_PERIOD}: {ma50_signal}")
        
        # Golden/Death Cross
        if ma20_values and ma20[-1] is not None:
            if ma20_current > ma50_current:
                print(f"  GOLDEN CROSS: MA{MA_SHORT_PERIOD} > MA{MA_LONG_PERIOD} - BULLISH SIGNAL")
            else:
                print(f"  DEATH CROSS: MA{MA_SHORT_PERIOD} < MA{MA_LONG_PERIOD} - BEARISH SIGNAL")
    
    # RSI Analysis
    if rsi_values:
        rsi_current = rsi_values[-1]
        if rsi_current > RSI_OVERBOUGHT:
            rsi_signal = "OVERBOUGHT - Consider selling"
        elif rsi_current < RSI_OVERSOLD:
            rsi_signal = "OVERSOLD - Consider buying"
        else:
            rsi_signal = "NEUTRAL - No extreme conditions"
        print(f"\nRSI Analysis:")
        print(f"  Current RSI: {rsi_current:.2f} - {rsi_signal}")
    
    return filename

if __name__ == "__main__":
    # Analyze multiple symbols
    for symbol in DEFAULT_SYMBOLS:
        print(f"\n{'='*ANALYSIS_SEPARATOR_LENGTH}")
        print(f"ANALYZING {symbol}")
        print('='*ANALYSIS_SEPARATOR_LENGTH)
        plot_stock_analysis(symbol, DEFAULT_DISPLAY_DAYS)
        print()  # Add spacing between analyses