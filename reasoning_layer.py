from datetime import datetime
import numpy as np

def reasoning(symbol, df, checked, reasons, order_book, heatmap, news_headlines=None, context={}):
    """
    Market-Structure-Aware, Multi-Timeframe Advanced Reasoning Layer for AI Signal Explanations
    """
    now_str = datetime.utcnow().strftime("%H:%M:%S")
    last = df.iloc[-1]
    base = symbol.split('/')[0].lower()

    # --- News & Sentiment context overlay ---
    headline_info = ""
    if news_headlines:
        recent_news = "; ".join(news_headlines[:2])
        headline_info = f"\n[!] Market news: {recent_news}"

    # --- Heatmap context (trendiness/risk-on-off) ---
    trending_coins = [coin['item']['symbol'].lower() for coin in heatmap.get('coins', [])] if heatmap else []
    trending_status = "Trending on heatmap" if base in trending_coins else "Not trending (risk-off)"

    # --- Helper for missing values ---
    def get_col(d, name, default=np.nan):
        return float(d.get(name, default))

    # Price, indicator snapshot (multi-TF / volatility regime aware)
    price = get_col(last, "close")
    ema21 = get_col(last, "ema21")
    ema200 = get_col(last, "ema200")
    rsi = get_col(last, "rsi")
    bb_upper = get_col(last, "bb_upper")
    bb_lower = get_col(last, "bb_lower")
    atr = get_col(last, "atr")
    obv = get_col(last, "obv")

    ema21_1h = get_col(last, "ema21_1h")
    ema200_1h = get_col(last, "ema200_1h")
    rsi_1h = get_col(last, "rsi_1h")
    ema21_15m = get_col(last, "ema21_15m")
    ema200_15m = get_col(last, "ema200_15m")
    rsi_15m = get_col(last, "rsi_15m")
    ema21_1m = get_col(last, "ema21_1m")
    ema200_1m = get_col(last, "ema200_1m")
    rsi_1m = get_col(last, "rsi_1m")

    # --- Market Structure: Trend/Regime Assessment (multi-TF) ---
    def trend_state(e21, e200):
        if np.isnan(e21) or np.isnan(e200):
            return "Indeterminate"
        if e21 > e200: return "Bullish"
        if e21 < e200: return "Bearish"
        return "Neutral"

    trend_mtf = {
        "1m": trend_state(ema21_1m, ema200_1m),
        "5m": trend_state(ema21, ema200),
        "15m": trend_state(ema21_15m, ema200_15m),
        "1h": trend_state(ema21_1h, ema200_1h)
    }

    # --- Cross-Timeframe Trend Alignment ---
    tf_trends = [v for v in trend_mtf.values() if v != "Indeterminate"]
    all_bull = all(t == "Bullish" for t in tf_trends)
    all_bear = all(t == "Bearish" for t in tf_trends)
    regime = "UPTREND" if all_bull else "DOWNTREND" if all_bear else "MIXED/UNCERTAIN"

    # --- Advanced Momentum (multi-TF) ---
    def mom_state(val):
        if np.isnan(val): return "Unknown"
        if val > 70: return "Overbought"
        if val < 30: return "Oversold"
        return "Balanced"
    mom_mtf = {
        "1m": mom_state(rsi_1m),
        "5m": mom_state(rsi),
        "15m": mom_state(rsi_15m),
        "1h": mom_state(rsi_1h)
    }

    # --- Volatility Context ---
    high_vol = False
    low_vol = False
    if not np.isnan(bb_upper) and not np.isnan(bb_lower) and not np.isnan(atr):
        bb_width = bb_upper - bb_lower
        vol_ratio = bb_width / (atr + 1e-8)
        if vol_ratio > 2.5: high_vol = True
        elif vol_ratio < 1.0: low_vol = True
    volatility_note = (
        "High volatility breakout (watch for whipsaw risk)" if high_vol
        else "Low volatility consolidation (prime for breakout wake-up)"
        if low_vol else "Medium volatility"
    )

    # --- Volume/Liquidity Profile ---
    if not np.isnan(obv):
        volume_state = "Strong accumulation" if obv > 0 else "Weak/flat volume"
    else:
        volume_state = "Unknown"

    # --- Order Book Microstructure ---
    top_bid = order_book['bids'][0][0] if order_book.get('bids') else 'N/A'
    top_ask = order_book['asks'][0][0] if order_book.get('asks') else 'N/A'

    # --- Signal Strength & Confluence ---
    if checked >= 15:
        strength = "Very strong (institutional-grade)"
    elif checked >= 10:
        strength = "Strong"
    elif checked >= 6:
        strength = "Moderate"
    else:
        strength = "Weak or Mixed"

    # --- Market Regime Warnings/Actions ---
    caution = ""
    if regime == "MIXED/UNCERTAIN":
        caution = "\n[!] Timeframes are not aligned. Stand aside, reduce size, or consider 'no trade zone'."
    elif high_vol:
        caution = "\n[!] High volatility may cause fakeouts—tighten SL or avoid breakouts."
    elif low_vol:
        caution = "\n[!] Squeeze detected: Big move likely soon. Beware of false breakouts."

    # --- Pattern/News Context Warnings ---
    highlights = f"{', '.join(reasons[:5])}{', ...' if len(reasons) > 5 else ''}"
    if "news" in context:
        caution += "\n[!] News event detected. Expect volatility spikes and potential spread blow-outs."

    # --- Build Output ---
    summary = [
        f"{now_str} — [{symbol}]",
        f"Signal strength & confluence: {strength} ({checked} checks passed)",
        f"Market regime: {regime}. Trends by TF: " + " ".join([f"{tf}:{trend_mtf[tf]}" for tf in trend_mtf]),
        f"Momentum (RSI) by TF: " + " ".join([f"{tf}:{mom_mtf[tf]}" for tf in mom_mtf]),
        f"Volatility status: {volatility_note} | Volume: {volume_state}",
        f"Order book: Top bid {top_bid}, top ask {top_ask}",
        f"Heatmap/trending: {trending_status}",
        highlights,
        caution,
        headline_info,
        f"Latest price: {price:.2f}" if not np.isnan(price) else "Latest price: N/A"
    ]
    return "\n".join(filter(None, summary))
