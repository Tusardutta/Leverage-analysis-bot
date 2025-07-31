import uuid
from datetime import datetime, timedelta

class SignalEntry:
    def __init__(self, symbol, signal_type, confidence, rationale, entry_price,
                 entry_time=None, target_pct=0.02, stop_pct=0.01, hold_duration_mins=120):
        self.id = uuid.uuid4()
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.rationale = rationale
        self.entry_price = entry_price
        self.entry_time = entry_time or datetime.utcnow()
        self.target_price = (
            self.entry_price * (1 + target_pct)
            if signal_type == "LONG"
            else self.entry_price * (1 - target_pct)
        )
        self.stop_price = (
            self.entry_price * (1 - stop_pct)
            if signal_type == "LONG"
            else self.entry_price * (1 + stop_pct)
        )
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
        else:
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
            "rationale": self.rationale,
        }
