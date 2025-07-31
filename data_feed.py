import os
import ccxt.async_support as ccxt
import aiohttp
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
    "enableRateLimit": True,
})

async def fetch_ohlcv(symbol: str, timeframe: str = "1m", limit: int = 100):
    print(f"[DataFeed] Fetching {timeframe} OHLCV for {symbol}...")
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        print(f"[DataFeed] Fetched {len(data)} OHLCV candles for {symbol}.")
        return data
    except Exception as e:
        print(f"[DataFeed] Error fetching OHLCV for {symbol}: {e}")
        return []

async def fetch_order_book(symbol: str, limit: int = 100):
    print(f"[DataFeed] Fetching order book for {symbol}...")
    try:
        ob = await exchange.fetch_order_book(symbol, limit=limit)
        print(f"[DataFeed] Order book fetched for {symbol}. Bids: {len(ob['bids'])}, Asks: {len(ob['asks'])}.")
        return ob
    except Exception as e:
        print(f"[DataFeed] Error fetching order book for {symbol}: {e}")
        return None

async def fetch_heatmap():
    print("[DataFeed] Fetching market heatmap from CoinGecko...")
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                response.raise_for_status()
                result = await response.json()
                print("[DataFeed] Market heatmap fetched successfully.")
                return result
    except Exception as e:
        print(f"[DataFeed] Error fetching CoinGecko heatmap data: {e}")
        return {}

async def close_exchange():
    print("[DataFeed] Closing Binance exchange connection...")
    try:
        await exchange.close()
        print("[DataFeed] Binance exchange connection closed.")
    except Exception as e:
        print(f"[DataFeed] Error closing exchange connection: {e}")
