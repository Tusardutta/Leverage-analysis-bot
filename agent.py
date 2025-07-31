import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from data_feed import fetch_ohlcv, fetch_order_book, fetch_heatmap, close_exchange
from strategy_engine import calc_indicators, comprehensive_strategy_checks, align_higher_tf
from reasoning_layer import reasoning
from output_module import trader_speak
import uuid
import numpy as np

TOP_SYMBOLS = ["SOL/USDT", "ETH/USDT", "AVAX/USDT"]
SIGNAL_COOLDOWN_MINS = 30
STICKY_CONFIRMS = 3
MIN_SIGNAL_HOLD_MINUTES = 120

signal_log = []
active_signals = {}
signal_cooldowns = {}
cooldown_locks = {sym: asyncio.Lock() for sym in TOP_SYMBOLS}
recent_signals = {sym: [] for sym in TOP_SYMBOLS}
last_signal_type = {sym: None for sym in TOP_SYMBOLS}
last_signal_time = {sym: None for sym in TOP_SYMBOLS}

agent_start_time = datetime.utcnow()
WARMUP_SECONDS = 300  # 5 minutes

# NEW: Per-symbol memory of all warmup analyses (list of dicts)
warmup_memory = {sym: [] for sym in TOP_SYMBOLS}
warmup_reviewed = {sym: False for sym in TOP_SYMBOLS}

def get_now():
    return datetime.utcnow()

def should_fire_signal(sig_list, new_signal, min_confirms=3):
    if len(sig_list) < min_confirms - 1:
        return False
    return new_signal is not None and all(s == new_signal for s in sig_list[-(min_confirms - 1):])

async def can_fire_signal(symbol, signal_type):
    async with cooldown_locks[symbol]:
        now = get_now()
        last_time = signal_cooldowns.get((symbol, signal_type))
        if last_time is None or (now - last_time) > timedelta(minutes=SIGNAL_COOLDOWN_MINS):
            signal_cooldowns[(symbol, signal_type)] = now
            return True
        return False

class SignalEntry:
    def __init__(self, symbol, signal_type, confidence, rationale, entry_price,
                 entry_time=None, target_pct=0.02, stop_pct=0.01, hold_duration_mins=120, status="CONFIRMED"):
        self.id = uuid.uuid4()
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.rationale = rationale
        self.entry_price = entry_price
        self.status = status  # "CONFIRMED"
        self.entry_time = entry_time or datetime.utcnow()
        self.target_price = (self.entry_price * (1 + target_pct) if signal_type == "LONG"
                             else self.entry_price * (1 - target_pct)) if signal_type in ("LONG", "SHORT") else None
        self.stop_price = (self.entry_price * (1 - stop_pct) if signal_type == "LONG"
                           else self.entry_price * (1 + stop_pct)) if signal_type in ("LONG", "SHORT") else None
        self.hold_duration = timedelta(minutes=hold_duration_mins)
        self.exit_time = None
        self.exit_price = None
        self.outcome = None

    def mark_exit(self, price, timestamp):
        self.exit_price = price
        self.exit_time = timestamp
        if self.signal_type == "LONG":
            if price >= self.target_price:
                self.outcome = "TARGET_HIT"
            elif price <= self.stop_price:
                self.outcome = "STOP_HIT"
            else:
                self.outcome = "TIME_EXPIRED"
        elif self.signal_type == "SHORT":
            if price <= self.target_price:
                self.outcome = "TARGET_HIT"
            elif price >= self.stop_price:
                self.outcome = "STOP_HIT"
            else:
                self.outcome = "TIME_EXPIRED"

    def as_dict(self):
        return {
            "id": str(self.id),
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "outcome": self.outcome,
            "status": self.status,
            "rationale": self.rationale,
        }

async def record_signal(symbol, signal_type, confidence, rationale, df, status="CONFIRMED"):
    entry_price = df.iloc[-1]['close']
    signal = SignalEntry(symbol, signal_type, confidence, rationale, entry_price, status=status)
    signal_log.append(signal)
    if status == "CONFIRMED":
        active_signals[signal.id] = signal
    print(f"[Signal Recorded] {signal.symbol} {signal.signal_type} at price {entry_price:.2f}, conf {confidence:.2%}, status {status}")
    return signal

def export_signal_log_csv(filename="signal_log.csv"):
    try:
        rows = [s.as_dict() for s in signal_log if s.status == "CONFIRMED"]
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"[Export] Signal log saved to {filename}")
    except Exception as e:
        print(f"[Export ERROR] Failed to save signal log: {e}")

def review_majority_signal(warmup_log):
    """
    Returns: (final_direction, confidence, majority_ratio, summary_reasons)
    """
    # Count each distinct (LONG/SHORT/NONE)
    dirs = [entry["direction"] for entry in warmup_log if entry["direction"]]
    if not dirs:
        return None, 0, 0, "NO SIGNAL SEEN"
    from collections import Counter
    c = Counter(dirs)
    majority_signal, count = c.most_common(1)[0]
    ratio = count / len(dirs)
    avg_conf = np.mean([entry["confidence"] for entry in warmup_log if entry["direction"] == majority_signal]) if count > 0 else 0
    # Gather all reasons that occurred with majority direction
    summary_reasons = []
    for entry in warmup_log:
        if entry["direction"] == majority_signal:
            summary_reasons.extend(entry["reasons"])
    summary_reasons = list(dict.fromkeys(summary_reasons))  # unique reasons order-preserved
    return majority_signal, avg_conf, ratio, summary_reasons

async def analyze_symbol_continuous(symbol):
    print(f"[{get_now():%H:%M:%S}] >>> Continuous analysis started for {symbol}...")
    while True:
        try:
            ohlcv_5m = await fetch_ohlcv(symbol, "5m")
            ohlcv_15m = await fetch_ohlcv(symbol, "15m")
            order_book = await fetch_order_book(symbol)
            heatmap = await fetch_heatmap()

            if not ohlcv_5m or not ohlcv_15m or order_book is None:
                print(f"[{get_now():%H:%M:%S}] {symbol} insufficient data; skipping analysis.")
                await asyncio.sleep(2)
                continue

            df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], unit="ms")
            df_5m.set_index("timestamp", inplace=True)
            df_5m = calc_indicators(df_5m, rsi_period=9)
            df_5m["symbol"] = symbol

            df_15m = pd.DataFrame(ohlcv_15m, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"], unit="ms")
            df_15m.set_index("timestamp", inplace=True)
            df_15m = calc_indicators(df_15m, rsi_period=9)
            df_15m["symbol"] = symbol

            df = align_higher_tf(df_5m, df_15m, "_15m")
            checks_passed, reasons = comprehensive_strategy_checks(df, order_book, heatmap)

            direction = None
            confidence_norm = min(max(checks_passed, 0), 1)
            if confidence_norm > 0.7:
                direction = "LONG"
            elif confidence_norm < 0.3:
                direction = "SHORT"

            now = get_now()
            # ----------- WARMUP MEMORY PHASE ---------------
            if (now - agent_start_time).total_seconds() < WARMUP_SECONDS:
                # Log all analyses into memory (not main log or CSV)
                warmup_memory[symbol].append({
                    "timestamp": now,
                    "direction": direction,
                    "confidence": checks_passed,
                    "reasons": reasons[:],  # copy to avoid mutation,
                    "price": df.iloc[-1]["close"],
                })
                print(f"[{now:%H:%M:%S}] [WARMUP] {symbol}: direction={direction}, conf={checks_passed:.2f}, len={len(warmup_memory[symbol])}")
                await asyncio.sleep(1)
                continue

            # At first run after warmup for this symbol: Review log and act
            if not warmup_reviewed[symbol]:
                majority_dir, maj_conf, ratio, reasons_major = review_majority_signal(warmup_memory[symbol])
                if majority_dir in ["LONG", "SHORT"] and ratio >= 0.6:  # require ≥60% majority
                    entry_price = df.iloc[-1]["close"]
                    atr = df.iloc[-1]["atr"] if "atr" in df.columns else 0
                    sl = entry_price - atr if majority_dir == "LONG" else entry_price + atr
                    tp = entry_price + 2 * atr if majority_dir == "LONG" else entry_price - 2 * atr

                    rationale = f"Final warmup review: {maj_conf:.2f} confidence, {int(ratio*100)}% persistence. Reasons: {', '.join(reasons_major)}"
                    rationale += f"\nSL: {sl:.2f}, TP: {tp:.2f}"

                    await record_signal(symbol, majority_dir, maj_conf, rationale, df)
                    export_signal_log_csv()
                    last_signal_type[symbol] = majority_dir
                    last_signal_time[symbol] = now

                    output = trader_speak(symbol, [majority_dir], rationale)
                    print(f"\n[{now:%H:%M:%S}] [{symbol}] FINAL (warmup consensus) SIGNAL: {majority_dir}\n{output}\n")
                else:
                    print(f"\n[{now:%H:%M:%S}] [{symbol}] No strong consensus in warmup ({ratio:.2f}, {majority_dir}). Skipping entry.\n")
                warmup_reviewed[symbol] = True  # Only do warmup review once!
                await asyncio.sleep(1)
                continue

            # ----------- NORMAL POST-WARMUP SIGNAL LOGIC -----------
            recent_signals[symbol].append(direction)
            if len(recent_signals[symbol]) > STICKY_CONFIRMS:
                recent_signals[symbol].pop(0)

            signal_hold_expired = True
            if last_signal_time[symbol]:
                elapsed = (now - last_signal_time[symbol]).total_seconds() / 60.0
                if elapsed < MIN_SIGNAL_HOLD_MINUTES:
                    signal_hold_expired = False

            if direction and should_fire_signal(recent_signals[symbol], direction, STICKY_CONFIRMS):
                if last_signal_type[symbol] != direction and signal_hold_expired:
                    entry_price = df.iloc[-1]["close"]
                    atr = df.iloc[-1]["atr"] if "atr" in df.columns else 0
                    sl = entry_price - atr if direction == "LONG" else entry_price + atr
                    tp = entry_price + 2 * atr if direction == "LONG" else entry_price - 2 * atr

                    await record_signal(symbol, direction, checks_passed, reasons, df)
                    export_signal_log_csv()
                    last_signal_type[symbol] = direction
                    last_signal_time[symbol] = now

                    rationale = reasoning(symbol, df, checks_passed, reasons, order_book, heatmap)
                    rationale += f"\nSL: {sl:.2f}, TP: {tp:.2f}"
                    output = trader_speak(symbol, [direction], rationale)
                    print(f"\n[{now:%H:%M:%S}] [{symbol}] FINAL SIGNAL: {direction}\n{output}\n")

        except Exception as e:
            print(f"[{get_now():%H:%M:%S}] [ERROR] Analysis failed for {symbol}: {e}")

        await asyncio.sleep(1)

async def evaluate_signals():
    while True:
        now = datetime.utcnow()
        to_remove = []
        for signal in list(active_signals.values()):
            elapsed = now - signal.entry_time
            if signal.outcome is not None or elapsed > signal.hold_duration:
                latest_ohlcv = await fetch_ohlcv(signal.symbol, "1m")
                if not latest_ohlcv or len(latest_ohlcv) == 0:
                    continue
                latest_price = latest_ohlcv[-1][4]
                if signal.outcome is None:
                    signal.mark_exit(latest_price, now)
                    print(f"[Signal Evaluation] {signal.id} {signal.symbol} ended outcome: {signal.outcome} at {latest_price:.2f}")
                if signal.outcome is not None:
                    to_remove.append(signal.id)
        for sid in to_remove:
            active_signals.pop(sid, None)
        await asyncio.sleep(300)

async def run():
    print(f"[{get_now():%H:%M:%S}] Agent started. Monitoring symbols: {', '.join(TOP_SYMBOLS)}")
    evaluator_task = asyncio.create_task(evaluate_signals())
    try:
        await asyncio.gather(*[asyncio.shield(analyze_symbol_continuous(sym)) for sym in TOP_SYMBOLS])
    except KeyboardInterrupt:
        print("\nAgent stopped by user (KeyboardInterrupt). Saving signals...")
        export_signal_log_csv()
    finally:
        evaluator_task.cancel()
        await close_exchange()
        print(f"[{get_now():%H:%M:%S}] Exchange connections closed. Goodbye.")

if __name__ == "__main__":
    asyncio.run(run())
