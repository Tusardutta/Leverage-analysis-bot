import talib
import pandas_ta as ta
import numpy as np
import pandas as pd

# -- CONFIGURATION --

CHECK_WEIGHTS = {
    # (same as before, but ready for expansion)
    "ema21_ema200": 2.0, "ema21_ema200_1m": 1.0, "ema21_ema200_15m": 2.0, "ema21_ema200_1h": 2.0,
    "supertrend": 1.5, "supertrend_1m": 0.5, "supertrend_15m": 1.5, "supertrend_1h": 1.5,
    "hma21_ema21": 1.0, "hma21_ema21_1m": 0.5, "hma21_ema21_15m": 1.0, "hma21_ema21_1h": 1.0,
    "ichimoku_cloud": 1.5, "ichimoku_cloud_1m": 0.5, "ichimoku_cloud_15m": 1.5, "ichimoku_cloud_1h": 1.5,
    "choppiness": 1.2, "choppiness_1m": 0.5, "rsi": 1.5, "rsi_1m": 0.5, "rsi_15m": 1.5, "rsi_1h": 1.5,
    "stochrsi_k": 1.0, "macd_positive": 1.5, "cci": 1.2, "bb_squeeze": 1.0,
    "adx": 1.2, "obv_rising": 1.0, "price_above_vwap": 1.0,
    "bullish_engulfing": 1.5, "bullish_hammer": 1.5,
    "orderbook_buy_pressure": 1.5, "orderbook_sell_pressure": 1.5,
    "heatmap_trending": 1.0,
    # Add as needed for custom/alt data
}

THRESHOLDS = {
    "rsi_bullish_1m": 80,
    "rsi_bullish": 75,
    "rsi_bullish_15m": 65,
    "rsi_bullish_1h": 60,
    "stochrsi_k_overbought": 0.85,
    "cci_bullish": 120,
    "bb_squeeze_atr_mult": 1.8,
    "adx_strong_trend": 25,
    "orderbook_imbalance": 0.2,
    "choppiness_trending": 35,
}

# -- INDICATOR CALCULATION -- (as before)

def calc_indicators(df, rsi_period=9):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values
    df["ema8"] = talib.EMA(c, timeperiod=8)
    df["ema21"] = talib.EMA(c, timeperiod=21)
    df["ema200"] = talib.EMA(c, timeperiod=200)
    df["hma21"] = ta.hma(df["close"], length=21)
    supertrend_df = ta.supertrend(df["high"], df["low"], df["close"])
    df["supertrend"] = supertrend_df.get("SUPERT_7_3.0", np.nan)
    ichi_cloud, _ = ta.ichimoku(df["high"], df["low"], df["close"])
    df["ichimoku_a"] = ichi_cloud.get("ISA_9", np.nan)
    df["ichimoku_b"] = ichi_cloud.get("ISB_26", np.nan)
    df["choppiness"] = ta.chop(df["high"], df["low"], df["close"])
    df["rsi"] = talib.RSI(c, timeperiod=rsi_period)
    stochrsi_df = ta.stochrsi(df["close"], length=rsi_period)
    df["stochrsi_k"] = stochrsi_df.iloc[:, 0] if stochrsi_df is not None and not stochrsi_df.empty else np.nan
    macd, _, macdhist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    df["macdhist"] = macdhist
    df["cci"] = talib.CCI(h, l, c, timeperiod=20)
    df["obv"] = talib.OBV(c, v)
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    bb_upper, bb_mid, bb_lower = talib.BBANDS(c, timeperiod=20)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb_upper, bb_mid, bb_lower
    df["atr"] = talib.ATR(h, l, c, timeperiod=14)
    df["adx"] = talib.ADX(h, l, c, timeperiod=14)
    df["engulfing"] = talib.CDLENGULFING(o, h, l, c)
    df["hammer"] = talib.CDLHAMMER(o, h, l, c)
    df.fillna(value=np.nan, inplace=True)
    return df

# -- MULTIFRAME MERGE AS BEFORE --

def align_higher_tf(df_main, df_htf, suffix):
    htf = df_htf.add_suffix(suffix)
    return pd.merge_asof(
        df_main.sort_index(), htf.sort_index(),
        left_index=True, right_index=True, direction="backward"
    )

# -- ADVANCED STRATEGY CHECKS/EXPLANATIONS --

def comprehensive_strategy_checks(df, order_book, heatmap, custom_signals=None):
    last = df.iloc[-1]
    score = 0.0
    max_score = sum(CHECK_WEIGHTS.values())
    reasons = []

    def passed_check(flag, reason_key, reason_text, explanation=None):
        nonlocal score
        if flag:
            weight = CHECK_WEIGHTS.get(reason_key, 1.0)
            score += weight
            # Give both brief and context explanation if available
            text = reason_text
            if explanation:
                text += f" — {explanation}"
            reasons.append(text)
            return True
        return False

    # 1. Multi-TF Trend Consensus (pro playbook logic): Calculate trend for all TFs.
    tfs = [("", "5m"), ("_1m", "1m"), ("_15m", "15m"), ("_1h", "1h")]
    tf_trends = {}
    for suf, lab in tfs:
        ema21 = last.get(f"ema21{suf}", np.nan)
        ema200 = last.get(f"ema200{suf}", np.nan)
        if not np.isnan(ema21) and not np.isnan(ema200):
            tf_trends[lab] = "Bullish" if ema21 > ema200 else "Bearish" if ema21 < ema200 else "Neutral"

    trend_votes = {"Bullish": 0, "Bearish": 0}
    for v in tf_trends.values():
        if v in ["Bullish", "Bearish"]:
            trend_votes[v] += 1

    if trend_votes["Bullish"] >= 3:
        reasons.append("MULTI-TF BULLISH ALIGNMENT: Majority of timeframes show uptrend — strong confirmation by pro trader convention.")
    elif trend_votes["Bearish"] >= 3:
        reasons.append("MULTI-TF BEARISH ALIGNMENT: Most timeframes show downtrend — stronger signal per pro risk models.")
    elif sum(trend_votes.values()) >= 2:
        reasons.append("WARNING: Mixed or indecisive regime — avoid new positions per institutional guides.")

    # 2. Traditional signal checks, pro wording
    passed_check(last.get("supertrend") == 1, "supertrend", "5m: Supertrend bullish", explanation="Momentum models (Supertrend) confirm trend, increases setup reliability.")
    passed_check(last.get("rsi") > THRESHOLDS["rsi_bullish"], "rsi", "5m: RSI above bullish threshold", explanation="Strong momentum; pro traders often require RSI as confirmation layer.")
    passed_check(last.get("adx", 0) > THRESHOLDS["adx_strong_trend"], "adx", "Strong trend (ADX)", explanation="ADX filter often used in institutional models: removes signals in choppy/range.")
    passed_check(last.get("choppiness", 100) < THRESHOLDS["choppiness_trending"], "choppiness", "Market is trending (low Choppiness Index)", explanation="Choppiness below threshold = trending conditions per major quant studies.")
    passed_check(last.get("cci", 0) > THRESHOLDS["cci_bullish"], "cci", "CCI strong uptrend", explanation="CCI signal-based filters well-cited in momentum funds.")

    # Volatility regime explainer
    bb_width = last.get("bb_upper", np.nan) - last.get("bb_lower", np.nan)
    atr_val = last.get("atr", np.nan)
    if not np.isnan(bb_width) and not np.isnan(atr_val):
        ratio = bb_width / (atr_val + 1e-8)
        if ratio < 1:
            reasons.append("VOLATILITY SQUEEZE: Narrow BB vs ATR — suggests breakout setup imminent (as recommended in 'The Volatility Edge').")

    # Volume, liquidity & structure (as in pro desks)
    obv_prev = df["obv"].iloc[-2] if len(df) > 1 else np.nan
    passed_check((not np.isnan(last.get("obv", np.nan)) and not np.isnan(obv_prev) and last["obv"] > obv_prev),
                 "obv_rising", "OBV rising", explanation="Evidence of real money flow (OBV rising) — accumulation phase per market structure research.")

    # Order book imbalance (per leading institutional crypto trading guides)
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    bid_volume = sum(float(b[1]) for b in bids)
    ask_volume = sum(float(a[1]) for a in asks)
    total_vol = bid_volume + ask_volume
    if total_vol > 0:
        imbalance = (bid_volume - ask_volume) / total_vol
        if imbalance > THRESHOLDS["orderbook_imbalance"]:
            reasons.append("ORDER BOOK DOMINATED BY BUY BIDS: Spot buy side pressure — this context often filters low-conviction shorts in pro logic.")
        elif imbalance < -THRESHOLDS["orderbook_imbalance"]:
            reasons.append("ORDER BOOK DOMINATED BY SELL BIDS: Spot sell pressure — used to filter long signals per exchange microstructure handbooks.")

    # Heatmap/trending risk context
    symbol_base = last.get("symbol", None)
    if heatmap and "coins" in heatmap and symbol_base:
        base_symbol_code = symbol_base.split("/")[0].lower()
        trending_coins = [coin["item"]["symbol"].lower() for coin in heatmap["coins"]]
        if base_symbol_code in trending_coins:
            reasons.append("TRENDING ON MARKET HEATMAP: Confirms broad liquidity/risk-on status (common in crypto desk overlay screens).")

    # Candlestick patterns, as per classic teaching
    passed_check(last.get("engulfing", 0) == 100, "bullish_engulfing", "Bullish engulfing candle", explanation="Professional swing traders use this for additional confidence.")
    passed_check(last.get("hammer", 0) == 100, "bullish_hammer", "Bullish hammer pattern", explanation="Classic reversal candle found in many institutional traders' playbooks.")

    # Allows custom signals/AI overlays
    if custom_signals:
        for cs_reason in custom_signals:
            score += 1.0
            reasons.append(cs_reason)
    confidence = score / max_score if max_score > 0 else 0.0
    return confidence, reasons
