import asyncio
import sys
import ccxt.async_support as ccxt
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

# تنظیم SelectorEventLoop برای ویندوز
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# توکن ربات تلگرام
TOKEN: str = '7264155681:AAFnVkC4cGb9Z9chzFOiZi_IjiJAeAw1OTM'

# لیست صرافی‌ها
EXCHANGES: Dict[str, ccxt.Exchange] = {
    'binance': ccxt.binance({'enableRateLimit': True}),
    'kucoin': ccxt.kucoin({'enableRateLimit': True}),
    'bybit': ccxt.bybit({'enableRateLimit': True}),
    'okx': ccxt.okx({'enableRateLimit': True})
}

# مراحل مکالمه
LANGUAGE, EXCHANGE, PAIR, TIMEFRAME, INDICATOR, RESTART = range(6)

# جفت‌ارزهای اصلی
MAIN_PAIRS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT']

# تایم‌فریم‌های رایج
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

# اندیکاتورها
INDICATORS = [
    "All Indicators", "SMA", "RSI", "MACD", "Bollinger Bands",
    "Stochastic", "ATR", "ADX", "VWAP", "Ichimoku", "Fibonacci"
]

# پیام خوش‌آمدگویی و توضیحات
WELCOME_MESSAGE_EN = """
Welcome to the Crypto Analysis Bot!
This bot provides detailed technical analysis for cryptocurrency pairs across multiple timeframes. 
Select an exchange, pair, timeframe, and indicators to get in-depth insights and predictions.
Developed by: [Reza Roostaei]
"""
WELCOME_MESSAGE_FA = """
به ربات تحلیل کریپتو خوش آمدید!
این ربات تحلیل تکنیکال مفصلی برای جفت‌ارزهای کریپتو در تایم‌فریم‌های مختلف ارائه می‌دهد.
صرافی، جفت‌ارز، تایم‌فریم و اندیکاتورها را انتخاب کنید تا تحلیل و پیش‌بینی دقیق دریافت کنید.
توسعه‌دهنده: [رضا روستائی]
"""

# تابع گرفتن داده‌ها
async def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 50) -> pd.DataFrame:
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        raise Exception(f"Error fetching data from {exchange.name}: {str(e)}")
    finally:
        await exchange.close()

# توابع محاسبه اندیکاتورها
def calculate_sma(df: pd.DataFrame, period: int) -> float:
    return df['close'].rolling(window=period).mean().iloc[-1]

def calculate_rsi(df: pd.DataFrame, period: int) -> float:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> tuple[float, float, float]:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], macd_histogram.iloc[-1]

def calculate_bollinger_bands(df: pd.DataFrame, period: int, deviation: float) -> tuple[float, float, float]:
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (std * deviation)
    lower_band = sma - (std * deviation)
    return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]

def calculate_stochastic(df: pd.DataFrame, k_period: int, k_slowing: int, d_period: int) -> tuple[float, float]:
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k_fast = 100 * (df['close'] - low_min) / (high_max - low_min)
    k_slow = k_fast.rolling(window=k_slowing).mean()
    d = k_slow.rolling(window=d_period).mean()
    return k_slow.iloc[-1], d.iloc[-1]

def calculate_atr(df: pd.DataFrame, period: int) -> float:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean().iloc[-1]

def calculate_adx(df: pd.DataFrame, period: int) -> float:
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.abs().where((low_diff > high_diff) & (low_diff < 0), 0)
    tr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.rolling(window=period).mean().iloc[-1]

def calculate_vwap(df: pd.DataFrame, period: int) -> float:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return vwap.iloc[-1]

def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int, kijun_period: int, senkou_period: int) -> tuple[float, float, float, float, Optional[float]]:
    tenkan_sen = (df['high'].rolling(window=tenkan_period).max() + df['low'].rolling(window=tenkan_period).min()) / 2
    kijun_sen = (df['high'].rolling(window=kijun_period).max() + df['low'].rolling(window=kijun_period).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (df['high'].rolling(window=senkou_period).max() + df['low'].rolling(window=senkou_period).min()) / 2
    chikou_span = df['close'].shift(-26)
    return (tenkan_sen.iloc[-1], kijun_sen.iloc[-1], senkou_span_a.iloc[-1], senkou_span_b.iloc[-1], chikou_span.iloc[-26] if len(df) > 26 else None)

def calculate_fibonacci(df: pd.DataFrame, period: int) -> Dict[str, float]:
    high = df['high'].rolling(window=period).max().iloc[-1]
    low = df['low'].rolling(window=period).min().iloc[-1]
    diff = high - low
    return {
        '23.6%': high - (diff * 0.236), '38.2%': high - (diff * 0.382),
        '50.0%': high - (diff * 0.5), '61.8%': high - (diff * 0.618),
        '78.6%': high - (diff * 0.786)
    }

# تابع تبدیل اعداد به فارسی
def to_persian_num(num: str) -> str:
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    return ''.join(persian_digits[int(d)] if d.isdigit() else d for d in num)

# تابع نتیجه خلاصه
def get_summary_prediction(bullish_signals: int, bearish_signals: int, lang: str = 'fa') -> str:
    if lang == 'en':
        if bullish_signals > bearish_signals and bullish_signals >= 4:
            return f"Summary: Bullish - {bullish_signals} positive signals vs {bearish_signals} negative signals.\n"
        elif bearish_signals > bullish_signals and bearish_signals >= 4:
            return f"Summary: Bearish - {bearish_signals} negative signals vs {bullish_signals} positive signals.\n"
        else:
            return f"Summary: Neutral - {bullish_signals} bullish signals and {bearish_signals} bearish signals indicate no clear trend.\n"
    else:
        if bullish_signals > bearish_signals and bullish_signals >= 4:
            return f"خلاصه: صعودی - {to_persian_num(str(bullish_signals))} سیگنال مثبت در مقابل {to_persian_num(str(bearish_signals))} سیگنال منفی.\n"
        elif bearish_signals > bullish_signals and bearish_signals >= 4:
            return f"خلاصه: نزولی - {to_persian_num(str(bearish_signals))} سیگنال منفی در مقابل {to_persian_num(str(bullish_signals))} سیگنال مثبت.\n"
        else:
            return f"خلاصه: خنثی - {to_persian_num(str(bullish_signals))} سیگنال صعودی و {to_persian_num(str(bearish_signals))} سیگنال نزولی نشان‌دهنده عدم وجود روند مشخص.\n"

# تابع توضیحات مفصل اندیکاتورها
def get_indicator_details(last_price: float, sma: float, rsi: float, macd_line: float, signal_line: float, histogram: float, bb_upper: float, bb_middle: float, bb_lower: float, sto_k: float, sto_d: float, atr: float, adx: float, vwap: float, ich_tenkan: float, ich_kijun: float, ich_senkou_a: float, ich_senkou_b: float, ich_chikou: float, fib_levels: Dict[str, float], lang: str = 'fa') -> str:
    if lang == 'en':
        details = "Indicator Analysis:\n"
        details += f"- SMA (14): {sma:.2f}. The Simple Moving Average shows the average price over 14 periods. "
        details += f"Current price ({last_price:.2f}) is {'above' if last_price > sma else 'below' if last_price < sma else 'at'} the SMA, indicating {'potential upward momentum' if last_price > sma else 'downward pressure' if last_price < sma else 'a neutral stance'}.\n"
        
        details += f"- RSI (14): {rsi:.2f}. The Relative Strength Index measures momentum on a scale of 0-100. "
        details += f"{'Overbought (>70)' if rsi > 70 else 'Oversold (<30)' if rsi < 30 else 'Neutral (30-70)'}. "
        details += f"This suggests {'a potential downward correction' if rsi > 70 else 'an upward bounce' if rsi < 30 else 'balanced momentum'}.\n"
        
        details += f"- MACD (12, 26, 9): Line: {macd_line:.2f}, Signal: {signal_line:.2f}, Histogram: {histogram:.2f}. "
        details += f"MACD shows trend direction and momentum. {'Bullish crossover' if macd_line > signal_line else 'Bearish crossover'}. "
        details += f"The histogram ({'positive' if histogram > 0 else 'negative'}) indicates {'increasing bullish momentum' if histogram > 0 else 'increasing bearish momentum'}.\n"
        
        details += f"- Bollinger Bands (20, 2): Upper: {bb_upper:.2f}, Middle: {bb_middle:.2f}, Lower: {bb_lower:.2f}. "
        details += f"Price is {'above upper band' if last_price > bb_upper else 'below lower band' if last_price < bb_lower else 'within bands'}. "
        details += f"This indicates {'strong buying pressure' if last_price > bb_upper else 'strong selling pressure' if last_price < bb_lower else 'normal volatility'}.\n"
        
        details += f"- Stochastic (%K: {sto_k:.2f}, %D: {sto_d:.2f}): {'Overbought (>80)' if sto_k > 80 else 'Oversold (<20)' if sto_k < 20 else 'Neutral'}. "
        details += f"{'Bullish crossover' if sto_k > sto_d and sto_k > 50 else 'Bearish crossover' if sto_k < sto_d and sto_k < 50 else 'No clear crossover'}. "
        details += f"This suggests {'a potential reversal down' if sto_k > 80 else 'a potential reversal up' if sto_k < 20 else 'balanced conditions'}.\n"
        
        details += f"- ATR (14): {atr:.2f}. The Average True Range measures volatility. "
        details += f"{'High volatility' if atr > df['close'].std() else 'Low volatility'}, indicating {'strong price movements' if atr > df['close'].std() else 'market consolidation'}.\n"
        
        details += f"- ADX (14): {adx:.2f}. The Average Directional Index measures trend strength. "
        details += f"{'Strong trend (>25)' if adx > 25 else 'Weak trend (<20)' if adx < 20 else 'Moderate trend'}. "
        details += f"This suggests {'a reliable trend' if adx > 25 else 'possible consolidation'}.\n"
        
        details += f"- VWAP (14): {vwap:.2f}. Volume Weighted Average Price shows the average price weighted by volume. "
        details += f"Price is {'above' if last_price > vwap else 'below'} VWAP, indicating {'bullish sentiment' if last_price > vwap else 'bearish sentiment'}.\n"
        
        details += f"- Ichimoku (9, 26, 52): Tenkan: {ich_tenkan:.2f}, Kijun: {ich_kijun:.2f}, Senkou A: {ich_senkou_a:.2f}, Senkou B: {ich_senkou_b:.2f}, Chikou: {ich_chikou:.2f if ich_chikou else 'N/A'}. "
        details += f"Price is {'above' if last_price > max(ich_senkou_a, ich_senkou_b) else 'below' if last_price < min(ich_senkou_a, ich_senkou_b) else 'within'} the cloud, suggesting {'bullish trend' if last_price > max(ich_senkou_a, ich_senkou_b) else 'bearish trend' if last_price < min(ich_senkou_a, ich_senkou_b) else 'neutral conditions'}. "
        details += f"Tenkan/Kijun: {'Bullish crossover' if ich_tenkan > ich_kijun else 'Bearish crossover' if ich_tenkan < ich_kijun else 'Neutral'}.\n"
        
        details += f"- Fibonacci (14): 23.6%: {fib_levels['23.6%']:.2f}, 38.2%: {fib_levels['38.2%']:.2f}, 50.0%: {fib_levels['50.0%']:.2f}, 61.8%: {fib_levels['61.8%']:.2f}, 78.6%: {fib_levels['78.6%']:.2f}. "
        details += f"Price is {'near support' if last_price < fib_levels['38.2%'] else 'near resistance' if last_price > fib_levels['61.8%'] else 'between levels'}, suggesting {'potential bounce up' if last_price < fib_levels['38.2%'] else 'potential pullback' if last_price > fib_levels['61.8%'] else 'no strong bias'}.\n"
    else:
        details = "تحلیل اندیکاتورها:\n"
        details += f"- میانگین متحرک ساده (SMA 14): {to_persian_num(f'{sma:.2f}')}. میانگین متحرک ساده قیمت میانگین ۱۴ دوره را نشان می‌دهد. "
        details += f"قیمت فعلی ({to_persian_num(f'{last_price:.2f}')}) {'بالاتر از' if last_price > sma else 'پایین‌تر از' if last_price < sma else 'در سطح'} SMA است، که نشان‌دهنده {'حرکت صعودی بالقوه' if last_price > sma else 'فشار نزولی' if last_price < sma else 'وضعیت خنثی'} است.\n"
        
        details += f"- قدرت نسبی (RSI 14): {to_persian_num(f'{rsi:.2f}')}. شاخص قدرت نسبی شتاب را در مقیاس ۰-۱۰۰ اندازه‌گیری می‌کند. "
        details += f"{'بیش‌خرید (>۷۰)' if rsi > 70 else 'بیش‌فروش (<۳۰)' if rsi < 30 else 'خنثی (۳۰-۷۰)'}. "
        details += f"این نشان‌دهنده {'اصلاح احتمالی به سمت پایین' if rsi > 70 else 'جهش به سمت بالا' if rsi < 30 else 'شتاب متعادل'} است.\n"
        
        details += f"- مکدی (MACD 12, 26, 9): خط: {to_persian_num(f'{macd_line:.2f}')}, سیگنال: {to_persian_num(f'{signal_line:.2f}')}, هیستوگرام: {to_persian_num(f'{histogram:.2f}')}. "
        details += f"مکدی جهت روند و شتاب را نشان می‌دهد. {'تقاطع صعودی' if macd_line > signal_line else 'تقاطع نزولی'}. "
        details += f"هیستوگرام ({'مثبت' if histogram > 0 else 'منفی'}) نشان‌دهنده {'شتاب صعودی رو به افزایش' if histogram > 0 else 'شتاب نزولی رو به افزایش'} است.\n"
        
        details += f"- باندهای بولینگر (BB 20, 2): بالا: {to_persian_num(f'{bb_upper:.2f}')}, وسط: {to_persian_num(f'{bb_middle:.2f}')}, پایین: {to_persian_num(f'{bb_lower:.2f}')}. "
        details += f"قیمت {'بالاتر از باند بالایی' if last_price > bb_upper else 'پایین‌تر از باند پایینی' if last_price < bb_lower else 'در محدوده باندها'} است. "
        details += f"این نشان‌دهنده {'فشار خرید قوی' if last_price > bb_upper else 'فشار فروش قوی' if last_price < bb_lower else 'نوسانات عادی'} است.\n"
        
        details += f"- استوکاستیک (%K: {to_persian_num(f'{sto_k:.2f}')}, %D: {to_persian_num(f'{sto_d:.2f}')}): {'بیش‌خرید (>۸۰)' if sto_k > 80 else 'بیش‌فروش (<۲۰)' if sto_k < 20 else 'خنثی'}. "
        details += f"{'تقاطع صعودی' if sto_k > sto_d and sto_k > 50 else 'تقاطع نزولی' if sto_k < sto_d and sto_k < 50 else 'بدون تقاطع مشخص'}. "
        details += f"این نشان‌دهنده {'بازگشت احتمالی به پایین' if sto_k > 80 else 'بازگشت احتمالی به بالا' if sto_k < 20 else 'شرایط متعادل'} است.\n"
        
        details += f"- دامنه واقعی (ATR 14): {to_persian_num(f'{atr:.2f}')}. دامنه واقعی میانگین نوسانات را اندازه‌گیری می‌کند. "
        details += f"{'نوسان بالا' if atr > df['close'].std() else 'نوسان پایین'}، که نشان‌دهنده {'حرکات قیمتی قوی' if atr > df['close'].std() else 'تثبیت بازار'} است.\n"
        
        details += f"- قدرت روند (ADX 14): {to_persian_num(f'{adx:.2f}')}. شاخص جهت‌دار میانگین قدرت روند را اندازه‌گیری می‌کند. "
        details += f"{'روند قوی (>۲۵)' if adx > 25 else 'روند ضعیف (<۲۰)' if adx < 20 else 'روند متوسط'}. "
        details += f"این نشان‌دهنده {'روند قابل اعتماد' if adx > 25 else 'تثبیت احتمالی'} است.\n"
        
        details += f"- قیمت میانگین وزنی (VWAP 14): {to_persian_num(f'{vwap:.2f}')}. VWAP میانگین قیمت وزن‌شده با حجم را نشان می‌دهد. "
        details += f"قیمت {'بالاتر از' if last_price > vwap else 'پایین‌تر از'} VWAP است، که نشان‌دهنده {'احساسات صعودی' if last_price > vwap else 'احساسات نزولی'} است.\n"
        
        details += f"- ایچیموکو (Ichimoku 9, 26, 52): تنکان: {to_persian_num(f'{ich_tenkan:.2f}')}, کیجون: {to_persian_num(f'{ich_kijun:.2f}')}, سنکو A: {to_persian_num(f'{ich_senkou_a:.2f}')}, سنکو B: {to_persian_num(f'{ich_senkou_b:.2f}')}, چیکو: {to_persian_num(f'{ich_chikou:.2f}') if ich_chikou else 'نامشخص'}. "
        details += f"قیمت {'بالای ابر' if last_price > max(ich_senkou_a, ich_senkou_b) else 'زیر ابر' if last_price < min(ich_senkou_a, ich_senkou_b) else 'درون ابر'} است، که نشان‌دهنده {'روند صعودی' if last_price > max(ich_senkou_a, ich_senkou_b) else 'روند نزولی' if last_price < min(ich_senkou_a, ich_senkou_b) else 'شرایط خنثی'} است. "
        details += f"تنکان/کیجون: {'تقاطع صعودی' if ich_tenkan > ich_kijun else 'تقاطع نزولی' if ich_tenkan < ich_kijun else 'خنثی'}.\n"
        
        details += f"- فیبوناچی (Fibonacci 14): ۲۳.۶٪: {to_persian_num(f'{fib_levels['23.6%']:.2f}')}, ۳۸.۲٪: {to_persian_num(f'{fib_levels['38.2%']:.2f}')}, ۵۰.۰٪: {to_persian_num(f'{fib_levels['50.0%']:.2f}')}, ۶۱.۸٪: {to_persian_num(f'{fib_levels['61.8%']:.2f}')}, ۷۸.۶٪: {to_persian_num(f'{fib_levels['78.6%']:.2f}')}. "
        details += f"قیمت {'نزدیک سطح حمایت' if last_price < fib_levels['38.2%'] else 'نزدیک سطح مقاومت' if last_price > fib_levels['61.8%'] else 'بین سطوح'} است، که نشان‌دهنده {'جهش احتمالی به بالا' if last_price < fib_levels['38.2%'] else 'بازگشت احتمالی به پایین' if last_price > fib_levels['61.8%'] else 'عدم تعصب قوی'} است.\n"
    return details

# تابع پیش‌بینی مفصل‌تر
def get_detailed_prediction(last_price: float, sma: float, rsi: float, macd_line: float, signal_line: float, adx: float, bb_upper: float, bb_middle: float, bb_lower: float, sto_k: float, sto_d: float, atr: float, vwap: float, ich_tenkan: float, ich_kijun: float, ich_senkou_a: float, ich_senkou_b: float, lang: str = 'fa') -> str:
    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    macd_status = "Bullish" if macd_line > signal_line else "Bearish"
    adx_status = "Strong" if adx > 25 else "Weak" if adx < 20 else "Moderate"
    bb_status = "Near Upper (Buy Pressure)" if last_price > bb_upper else "Near Lower (Sell Pressure)" if last_price < bb_lower else "Neutral"
    sto_status = "Overbought" if sto_k > 80 else "Oversold" if sto_k < 20 else "Neutral"
    sto_cross = "Bullish Cross" if sto_k > sto_d and sto_k > 50 else "Bearish Cross" if sto_k < sto_d and sto_k < 50 else "No Clear Cross"
    ich_cloud = "Above Cloud (Bullish)" if last_price > max(ich_senkou_a, ich_senkou_b) else "Below Cloud (Bearish)" if last_price < min(ich_senkou_a, ich_senkou_b) else "In Cloud (Neutral)"
    ich_cross = "Bullish" if ich_tenkan > ich_kijun else "Bearish" if ich_tenkan < ich_kijun else "Neutral"
    sma_status = "Bullish" if last_price > sma else "Bearish" if last_price < sma else "Neutral"
    vwap_status = "Bullish" if last_price > vwap else "Bearish" if last_price < vwap else "Neutral"
    atr_status = "High Volatility" if atr > bb_middle.std() else "Low Volatility"

    bullish_signals = sum([macd_status == "Bullish", adx_status == "Strong", bb_status == "Near Upper (Buy Pressure)", sto_status == "Neutral" and sto_cross == "Bullish Cross", ich_cloud == "Above Cloud (Bullish)", ich_cross == "Bullish", sma_status == "Bullish", vwap_status == "Bullish"])
    bearish_signals = sum([macd_status == "Bearish", adx_status == "Strong", bb_status == "Near Lower (Sell Pressure)", sto_status == "Neutral" and sto_cross == "Bearish Cross", ich_cloud == "Below Cloud (Bearish)", ich_cross == "Bearish", sma_status == "Bearish", vwap_status == "Bearish"])

    if lang == 'en':
        prediction = "Comprehensive Prediction:\n"
        prediction += f"Based on {bullish_signals} bullish signals and {bearish_signals} bearish signals, here’s the detailed outlook:\n"
        
        if bullish_signals > bearish_signals and bullish_signals >= 4:
            prediction += "Bullish Outlook:\n"
            prediction += f"- The market shows strong upward momentum with {bullish_signals} positive signals. "
            prediction += "Key indicators include:\n"
            if sma_status == "Bullish": prediction += "  - Price above SMA: Indicates consistent buying interest.\n"
            if rsi_status == "Neutral": prediction += "  - RSI in neutral zone: Suggests room for further upside without overbought conditions.\n"
            if macd_status == "Bullish": prediction += "  - MACD bullish crossover: Confirms increasing upward momentum.\n"
            if bb_status == "Near Upper (Buy Pressure)": prediction += "  - Price near upper Bollinger Band: Strong buying pressure is present.\n"
            if sto_cross == "Bullish Cross": prediction += "  - Stochastic bullish crossover: Signals potential continuation of the uptrend.\n"
            if adx_status == "Strong": prediction += "  - ADX above 25: Confirms a strong bullish trend.\n"
            if vwap_status == "Bullish": prediction += "  - Price above VWAP: Buyers are in control.\n"
            if ich_cloud == "Above Cloud (Bullish)": prediction += "  - Price above Ichimoku cloud: Long-term bullish trend is supported.\n"
            prediction += "Recommendation: Consider a long position. Watch for RSI approaching 70 or a pullback to VWAP/SMA as potential entry points.\n"
        
        elif bearish_signals > bullish_signals and bearish_signals >= 4:
            prediction += "Bearish Outlook:\n"
            prediction += f"- The market indicates downward pressure with {bearish_signals} negative signals. "
            prediction += "Key indicators include:\n"
            if sma_status == "Bearish": prediction += "  - Price below SMA: Suggests sustained selling pressure.\n"
            if rsi_status == "Neutral": prediction += "  - RSI in neutral zone: Allows room for further downside without oversold conditions.\n"
            if macd_status == "Bearish": prediction += "  - MACD bearish crossover: Confirms increasing downward momentum.\n"
            if bb_status == "Near Lower (Sell Pressure)": prediction += "  - Price near lower Bollinger Band: Strong selling pressure is evident.\n"
            if sto_cross == "Bearish Cross": prediction += "  - Stochastic bearish crossover: Signals potential continuation of the downtrend.\n"
            if adx_status == "Strong": prediction += "  - ADX above 25: Confirms a strong bearish trend.\n"
            if vwap_status == "Bearish": prediction += "  - Price below VWAP: Sellers dominate the market.\n"
            if ich_cloud == "Below Cloud (Bearish)": prediction += "  - Price below Ichimoku cloud: Long-term bearish trend is supported.\n"
            prediction += "Recommendation: Consider a short position. Monitor RSI nearing 30 or a bounce to VWAP/SMA as potential entry points.\n"
        
        else:
            prediction += "Neutral Outlook:\n"
            prediction += "- The market lacks a clear direction with balanced signals. "
            prediction += "Key observations:\n"
            if rsi_status in ["Overbought", "Oversold"]: prediction += f"  - RSI {rsi_status}: Watch for a reversal soon.\n"
            if adx_status == "Weak": prediction += "  - ADX below 20: Indicates consolidation; wait for a breakout.\n"
            if bb_status == "Neutral": prediction += "  - Price within Bollinger Bands: Suggests normal volatility with no strong trend.\n"
            if ich_cloud == "In Cloud (Neutral)": prediction += "  - Price in Ichimoku cloud: Trend direction is unclear.\n"
            prediction += "Recommendation: Stay cautious. Look for a breakout above VWAP/SMA or below BB lower band to confirm direction.\n"
        
        prediction += f"Volatility Note: {atr_status}. {'Expect significant price swings' if atr_status == 'High Volatility' else 'Market may remain range-bound'}.\n"
    else:
        prediction = "پیش‌بینی جامع:\n"
        prediction += f"بر اساس {to_persian_num(str(bullish_signals))} سیگنال صعودی و {to_persian_num(str(bearish_signals))} سیگنال نزولی، تحلیل زیر ارائه می‌شود:\n"
        
        if bullish_signals > bearish_signals and bullish_signals >= 4:
            prediction += "چشم‌انداز صعودی:\n"
            prediction += f"- بازار شتاب صعودی قوی با {to_persian_num(str(bullish_signals))} سیگنال مثبت نشان می‌دهد. "
            prediction += "اندیکاتورهای کلیدی:\n"
            if sma_status == "Bullish": prediction += "  - قیمت بالای SMA: نشان‌دهنده علاقه مداوم به خرید است.\n"
            if rsi_status == "Neutral": prediction += "  - RSI در محدوده خنثی: فضا برای صعود بیشتر بدون شرایط بیش‌خرید.\n"
            if macd_status == "Bullish": prediction += "  - تقاطع صعودی MACD: شتاب صعودی رو به افزایش را تأیید می‌کند.\n"
            if bb_status == "Near Upper (Buy Pressure)": prediction += "  - قیمت نزدیک باند بالایی: فشار خرید قوی وجود دارد.\n"
            if sto_cross == "Bullish Cross": prediction += "  - تقاطع صعودی استوکاستیک: ادامه روند صعودی را نشان می‌دهد.\n"
            if adx_status == "Strong": prediction += "  - ADX بالای ۲۵: روند صعودی قوی را تأیید می‌کند.\n"
            if vwap_status == "Bullish": prediction += "  - قیمت بالای VWAP: خریداران کنترل بازار را در دست دارند.\n"
            if ich_cloud == "Above Cloud (Bullish)": prediction += "  - قیمت بالای ابر ایچیموکو: روند صعودی بلندمدت پشتیبانی می‌شود.\n"
            prediction += "توصیه: موقعیت خرید را در نظر بگیرید. به RSI نزدیک ۷۰ یا بازگشت به VWAP/SMA به عنوان نقاط ورود توجه کنید.\n"
        
        elif bearish_signals > bullish_signals and bearish_signals >= 4:
            prediction += "چشم‌انداز نزولی:\n"
            prediction += f"- بازار فشار نزولی با {to_persian_num(str(bearish_signals))} سیگنال منفی نشان می‌دهد. "
            prediction += "اندیکاتورهای کلیدی:\n"
            if sma_status == "Bearish": prediction += "  - قیمت زیر SMA: فشار فروش مداوم را نشان می‌دهد.\n"
            if rsi_status == "Neutral": prediction += "  - RSI در محدوده خنثی: فضا برای نزول بیشتر بدون شرایط بیش‌فروش.\n"
            if macd_status == "Bearish": prediction += "  - تقاطع نزولی MACD: شتاب نزولی رو به افزایش را تأیید می‌کند.\n"
            if bb_status == "Near Lower (Sell Pressure)": prediction += "  - قیمت نزدیک باند پایینی: فشار فروش قوی مشهود است.\n"
            if sto_cross == "Bearish Cross": prediction += "  - تقاطع نزولی استوکاستیک: ادامه روند نزولی را نشان می‌دهد.\n"
            if adx_status == "Strong": prediction += "  - ADX بالای ۲۵: روند نزولی قوی را تأیید می‌کند.\n"
            if vwap_status == "Bearish": prediction += "  - قیمت زیر VWAP: فروشندگان بازار را کنترل می‌کنند.\n"
            if ich_cloud == "Below Cloud (Bearish)": prediction += "  - قیمت زیر ابر ایچیموکو: روند نزولی بلندمدت پشتیبانی می‌شود.\n"
            prediction += "توصیه: موقعیت فروش را در نظر بگیرید. به RSI نزدیک ۳۰ یا جهش به VWAP/SMA به عنوان نقاط ورود توجه کنید.\n"
        
        else:
            prediction += "چشم‌انداز خنثی:\n"
            prediction += "- بازار جهت مشخصی ندارد و سیگنال‌ها متعادل هستند. "
            prediction += "مشاهدات کلیدی:\n"
            if rsi_status in ["Overbought", "Oversold"]: prediction += f"  - RSI {rsi_status}: به زودی منتظر بازگشت باشید.\n"
            if adx_status == "Weak": prediction += "  - ADX زیر ۲۰: تثبیت را نشان می‌دهد؛ منتظر شکست باشید.\n"
            if bb_status == "Neutral": prediction += "  - قیمت در باندهای بولینگر: نوسانات عادی و بدون روند قوی.\n"
            if ich_cloud == "In Cloud (Neutral)": prediction += "  - قیمت در ابر ایچیموکو: جهت روند نامشخص است.\n"
            prediction += "توصیه: احتیاط کنید. به شکست بالای VWAP/SMA یا زیر باند پایینی BB برای تأیید جهت توجه کنید.\n"
        
        prediction += f"یادداشت نوسانات: {'نوسان بالا' if atr_status == 'High Volatility' else 'نوسان پایین'}. {'انتظار حرکات قیمتی قابل توجه' if atr_status == 'High Volatility' else 'بازار ممکن است در محدوده بماند'}.\n"
    return prediction

# شروع مکالمه
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(WELCOME_MESSAGE_EN + "\n" + WELCOME_MESSAGE_FA)
    lang_buttons = [[KeyboardButton("فارسی"), KeyboardButton("English")]]
    reply_markup = ReplyKeyboardMarkup(lang_buttons, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "Please select a language:\nلطفاً یک زبان انتخاب کنید:",
        reply_markup=reply_markup
    )
    return LANGUAGE

# انتخاب زبان
async def select_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = update.message.text
    if lang == "فارسی":
        context.user_data['lang'] = 'fa'
    elif lang == "English":
        context.user_data['lang'] = 'en'
    else:
        await update.message.reply_text("Invalid choice! Please select 'فارسی' or 'English'.")
        return LANGUAGE
    lang = context.user_data['lang']
    exchange_buttons = [[KeyboardButton(name)] for name in EXCHANGES.keys()]
    exchange_buttons.append([KeyboardButton("Back" if lang == 'en' else "بازگشت")])
    reply_markup = ReplyKeyboardMarkup(exchange_buttons, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "Please select an exchange:" if lang == 'en' else "لطفاً یه صرافی انتخاب کن:",
        reply_markup=reply_markup
    )
    return EXCHANGE

# انتخاب صرافی
async def select_exchange(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = context.user_data.get('lang', 'fa')
    selection = update.message.text.lower()
    if selection == ("back" if lang == 'en' else "بازگشت"):
        lang_buttons = [[KeyboardButton("فارسی"), KeyboardButton("English")]]
        reply_markup = ReplyKeyboardMarkup(lang_buttons, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text(
            "Please select a language:\nلطفاً یک زبان انتخاب کنید:",
            reply_markup=reply_markup
        )
        return LANGUAGE
    if selection not in EXCHANGES:
        await update.message.reply_text(
            "Invalid exchange! Please select from the list." if lang == 'en' else "صرافی نامعتبر! لطفاً از لیست انتخاب کن."
        )
        return EXCHANGE
    context.user_data['exchange'] = selection
    pair_buttons = [[KeyboardButton(pair)] for pair in MAIN_PAIRS]
    pair_buttons.append([KeyboardButton("Enter Custom Pair" if lang == 'en' else "وارد کردن جفت‌ارز دستی")])
    pair_buttons.append([KeyboardButton("Back" if lang == 'en' else "بازگشت")])
    reply_markup = ReplyKeyboardMarkup(pair_buttons, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        f"Exchange set to {selection}. Now select a pair:" if lang == 'en' else f"صرافی به {selection} تنظیم شد. حالا یه جفت‌ارز انتخاب کن:",
        reply_markup=reply_markup
    )
    return PAIR

# انتخاب جفت‌ارز
async def select_pair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = context.user_data.get('lang', 'fa')
    selection = update.message.text
    if selection == ("back" if lang == 'en' else "بازگشت"):
        exchange_buttons = [[KeyboardButton(name)] for name in EXCHANGES.keys()]
        exchange_buttons.append([KeyboardButton("Back" if lang == 'en' else "بازگشت")])
        reply_markup = ReplyKeyboardMarkup(exchange_buttons, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text(
            "Please select an exchange:" if lang == 'en' else "لطفاً یه صرافی انتخاب کن:",
            reply_markup=reply_markup
        )
        return EXCHANGE
    if selection == ("Enter Custom Pair" if lang == 'en' else "وارد کردن جفت‌ارز دستی"):
        await update.message.reply_text(
            "Please enter the custom pair (e.g., BTC/USDT):" if lang == 'en' else "لطفاً جفت‌ارز دلخواه رو وارد کن (مثال: BTC/USDT):",
            reply_markup=ReplyKeyboardRemove()
        )
        return PAIR
    context.user_data['pair'] = selection.upper()
    timeframe_buttons = [[KeyboardButton(tf)] for tf in TIMEFRAMES]
    timeframe_buttons.append([KeyboardButton("Confirm" if lang == 'en' else "تأیید")])
    timeframe_buttons.append([KeyboardButton("Back" if lang == 'en' else "بازگشت")])
    reply_markup = ReplyKeyboardMarkup(timeframe_buttons, one_time_keyboard=False, resize_keyboard=True)
    await update.message.reply_text(
        "Select one or more timeframes (press 'Confirm' when done):" if lang == 'en' else "یک یا چند تایم‌فریم انتخاب کن (وقتی تموم شد 'تأیید' رو بزن):",
        reply_markup=reply_markup
    )
    context.user_data['timeframes'] = []
    return TIMEFRAME

# انتخاب تایم‌فریم‌ها
async def select_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = context.user_data.get('lang', 'fa')
    selection = update.message.text
    if selection == ("back" if lang == 'en' else "بازگشت"):
        pair_buttons = [[KeyboardButton(pair)] for pair in MAIN_PAIRS]
        pair_buttons.append([KeyboardButton("Enter Custom Pair" if lang == 'en' else "وارد کردن جفت‌ارز دستی")])
        pair_buttons.append([KeyboardButton("Back" if lang == 'en' else "بازگشت")])
        reply_markup = ReplyKeyboardMarkup(pair_buttons, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text(
            f"Exchange set to {context.user_data['exchange']}. Now select a pair:" if lang == 'en' else f"صرافی به {context.user_data['exchange']} تنظیم شد. حالا یه جفت‌ارز انتخاب کن:",
            reply_markup=reply_markup
        )
        return PAIR
    if selection == ("Confirm" if lang == 'en' else "تأیید"):
        if not context.user_data['timeframes']:
            await update.message.reply_text(
                "Please select at least one timeframe before confirming!" if lang == 'en' else "لطفاً حداقل یه تایم‌فریم انتخاب کن قبل از تأیید!"
            )
            return TIMEFRAME
        indicator_buttons = [[KeyboardButton(ind)] for ind in INDICATORS]
        indicator_buttons.append([KeyboardButton("Confirm" if lang == 'en' else "تأیید")])
        indicator_buttons.append([KeyboardButton("Back" if lang == 'en' else "بازگشت")])
        reply_markup = ReplyKeyboardMarkup(indicator_buttons, one_time_keyboard=False, resize_keyboard=True)
        await update.message.reply_text(
            "Select one or more indicators (press 'Confirm' when done):" if lang == 'en' else "یک یا چند اندیکاتور انتخاب کن (وقتی تموم شد 'تأیید' رو بزن):",
            reply_markup=reply_markup
        )
        context.user_data['indicators'] = []
        return INDICATOR
    if selection in TIMEFRAMES:
        context.user_data['timeframes'].append(selection)
        await update.message.reply_text(
            f"Added {selection}. Select more or press 'Confirm':" if lang == 'en' else f"{selection} اضافه شد. بیشتر انتخاب کن یا 'تأیید' رو بزن:"
        )
        return TIMEFRAME
    await update.message.reply_text(
        "Invalid timeframe! Please select from the list." if lang == 'en' else "تایم‌فریم نامعتبر! لطفاً از لیست انتخاب کن."
    )
    return TIMEFRAME

# انتخاب اندیکاتورها و نمایش نتیجه
async def select_indicator(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = context.user_data.get('lang', 'fa')
    selection = update.message.text
    LTR = '\u202A'
    POP = '\u202C'
    
    if selection == ("back" if lang == 'en' else "بازگشت"):
        timeframe_buttons = [[KeyboardButton(tf)] for tf in TIMEFRAMES]
        timeframe_buttons.append([KeyboardButton("Confirm" if lang == 'en' else "تأیید")])
        timeframe_buttons.append([KeyboardButton("Back" if lang == 'en' else "بازگشت")])
        reply_markup = ReplyKeyboardMarkup(timeframe_buttons, one_time_keyboard=False, resize_keyboard=True)
        await update.message.reply_text(
            "Select one or more timeframes (press 'Confirm' when done):" if lang == 'en' else "یک یا چند تایم‌فریم انتخاب کن (وقتی تموم شد 'تأیید' رو بزن):",
            reply_markup=reply_markup
        )
        context.user_data['timeframes'] = []
        return TIMEFRAME
    
    if selection == ("Confirm" if lang == 'en' else "تأیید"):
        if not context.user_data['indicators']:
            await update.message.reply_text(
                "Please select at least one indicator before confirming!" if lang == 'en' else "لطفاً حداقل یه اندیکاتور انتخاب کن قبل از تأیید!"
            )
            return INDICATOR
        exchange_name = context.user_data['exchange']
        pair = context.user_data['pair']
        timeframes = context.user_data['timeframes']
        indicators = context.user_data['indicators']
        exchange = EXCHANGES[exchange_name]
        response = ""
        
        for timeframe in timeframes:
            try:
                df = await fetch_ohlcv(exchange, pair, timeframe, limit=200)
                last_price = df['close'].iloc[-1]
                sma = calculate_sma(df, 14)
                rsi = calculate_rsi(df, 14)
                macd_line, signal_line, histogram = calculate_macd(df, 12, 26, 9)
                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, 20, 2)
                sto_k, sto_d = calculate_stochastic(df, 14, 3, 3)
                atr = calculate_atr(df, 14)
                adx = calculate_adx(df, 14)
                vwap = calculate_vwap(df, 14)
                ich_tenkan_sen, ich_kijun_sen, ich_senkou_a, ich_senkou_b, ich_chikou = calculate_ichimoku(df, 9, 26, 52)
                fib_levels = calculate_fibonacci(df, 14)
                
                # محاسبه سیگنال‌ها برای خلاصه
                bullish_signals = sum([
                    last_price > sma, rsi < 70 and rsi > 30, macd_line > signal_line, 
                    last_price > bb_upper, sto_k > sto_d and sto_k > 50, adx > 25, 
                    last_price > vwap, last_price > max(ich_senkou_a, ich_senkou_b)
                ])
                bearish_signals = sum([
                    last_price < sma, rsi > 70 or rsi < 30, macd_line < signal_line, 
                    last_price < bb_lower, sto_k < sto_d and sto_k < 50, adx > 25, 
                    last_price < vwap, last_price < min(ich_senkou_a, ich_senkou_b)
                ])
                
                if lang == 'en':
                    response += f"\n--- Analysis for Timeframe: {timeframe} ---\n"
                    response += f"Exchange: {exchange_name} | Pair: {pair} | Last Price: {last_price:.2f}\n"
                else:
                    response += f"\n--- تحلیل برای تایم‌فریم: {LTR}{timeframe}{POP} ---\n"
                    response += f"صرافی: <code>{exchange_name}</code> | جفت‌ارز: <code>{pair}</code> | آخرین قیمت: {to_persian_num(f'{last_price:.2f}')}\n"
                
                # اضافه کردن نتیجه خلاصه
                response += get_summary_prediction(bullish_signals, bearish_signals, lang)
                response += "-"*30 + "\n"
                
                # توضیحات مفصل اندیکاتورها
                if "All Indicators" in indicators:
                    response += get_indicator_details(last_price, sma, rsi, macd_line, signal_line, histogram, bb_upper, bb_middle, bb_lower, sto_k, sto_d, atr, adx, vwap, ich_tenkan_sen, ich_kijun_sen, ich_senkou_a, ich_senkou_b, ich_chikou, fib_levels, lang)
                    response += "\n"
                
                # پیش‌بینی مفصل
                response += get_detailed_prediction(last_price, sma, rsi, macd_line, signal_line, adx, bb_upper, bb_middle, bb_lower, sto_k, sto_d, atr, vwap, ich_tenkan_sen, ich_kijun_sen, ich_senkou_a, ich_senkou_b, lang)
                response += "\n" + "-"*50 + "\n"
                
            except Exception as e:
                response += f"Error for {timeframe}: {str(e)}\n"
        
        if not response:
            response = "No data available." if lang == 'en' else "داده‌ای در دسترس نیست."
        
        restart_button = [[KeyboardButton("Restart" if lang == 'en' else "شروع مجدد")]]
        reply_markup = ReplyKeyboardMarkup(restart_button, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text(response, parse_mode='HTML' if lang == 'fa' else None, reply_markup=reply_markup)
        return RESTART
    
    if selection in INDICATORS:
        context.user_data['indicators'].append(selection)
        await update.message.reply_text(
            f"Added {selection}. Select more or press 'Confirm':" if lang == 'en' else f"{selection} اضافه شد. بیشتر انتخاب کن یا 'تأیید' رو بزن:"
        )
        return INDICATOR
    
    await update.message.reply_text(
        "Invalid indicator! Please select from the list." if lang == 'en' else "اندیکاتور نامعتبر! لطفاً از لیست انتخاب کن."
    )
    return INDICATOR

# تابع مدیریت شروع مجدد
async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = context.user_data.get('lang', 'fa')
    if update.message.text == ("Restart" if lang == 'en' else "شروع مجدد"):
        context.user_data.clear()
        await update.message.reply_text(WELCOME_MESSAGE_EN + "\n" + WELCOME_MESSAGE_FA)
        lang_buttons = [[KeyboardButton("فارسی"), KeyboardButton("English")]]
        reply_markup = ReplyKeyboardMarkup(lang_buttons, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text(
            "Please select a language:\nلطفاً یک زبان انتخاب کنید:",
            reply_markup=reply_markup
        )
        return LANGUAGE
    return RESTART

# لغو مکالمه
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Operation cancelled." if context.user_data.get('lang', 'fa') == 'en' else "عملیات لغو شد.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

# تابع اصلی برای راه‌اندازی ربات
def main() -> None:
    application = Application.builder().token(TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            LANGUAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_language)],
            EXCHANGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_exchange)],
            PAIR: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_pair)],
            TIMEFRAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_timeframe)],
            INDICATOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_indicator)],
            RESTART: [MessageHandler(filters.TEXT & ~filters.COMMAND, restart)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == '__main__':
    main()