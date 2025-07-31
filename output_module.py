import os
from huggingface_hub import InferenceClient

# Initialize the client once globally (adjust provider/api_key as needed)
client = InferenceClient(
    provider="novita",  # Change provider as required
    api_key=os.environ.get("HF_API_KEY")
)

def trader_speak(symbol, signals, rationale, indicators_dict=None, order_book=None, heatmap=None,
                 confidence=None, sl=None, tp=None):
    print(f"[AI] Generating AI commentary for {symbol} via Hugging Face InferenceClient...")

    sig_text = ', '.join(signals) if signals else 'No active signals'

    # Build multi-timeframe indicator summary (example)
    ind_summary = []
    if indicators_dict:
        for tf_suffix, label in [("", "5m"), ("_1m", "1m"), ("_15m", "15m"), ("_1h", "1h")]:
            ema21 = indicators_dict.get(f"ema21{tf_suffix}", "n/a")
            ema200 = indicators_dict.get(f"ema200{tf_suffix}", "n/a")
            rsi = indicators_dict.get(f"rsi{tf_suffix}", "n/a")
            ind_summary.append(
                f"{label} | EMA21: {ema21}, EMA200: {ema200}, RSI: {rsi}"
            )

    # Order book summary
    book_summary = "N/A"
    if order_book:
        top_bid = order_book['bids'][0][0] if order_book.get('bids') else 'N/A'
        top_ask = order_book['asks'][0][0] if order_book.get('asks') else 'N/A'
        book_summary = f"Top bid: {top_bid}, Top ask: {top_ask}"

    # Heatmap summary
    heatmap_summary = "None"
    if heatmap and "coins" in heatmap:
        trending_coins = [coin['item']['symbol'] for coin in heatmap.get('coins', [])]
        heatmap_summary = "Trending coins: " + ", ".join(trending_coins) if trending_coins else "None"

    # Confidence display
    confidence_text = f"{confidence:.2%}" if confidence is not None else "N/A"

    # Construct detailed prompt (without news)
    prompt = (
        f"You are a top grade crypto long/short market analyst. I am a human asking your advice on whether to go long or short for {symbol}. " 
        f"I am aiming for a large gain and want a 99% accurate prediction with zero chance of error. "
        f"Signal produced: {sig_text} (Confidence: {confidence_text}).\n"
        f"Stop Loss: {sl if sl is not None else 'N/A'}, Take Profit: {tp if tp is not None else 'N/A'}.\n"
        f"Indicator summary by timeframe:\n" + "\n".join(ind_summary) + "\n"
        f"Order Book Snapshot:\n{book_summary}\n"
        f"Market Heatmap:\n{heatmap_summary}\n"
        f"Model rationale & reasonings:\n{rationale}\n"
        "Please provide a concise, factual, expert trading idea in 1-2 sentences suitable for terminal output. Do not speculate or exaggerate. Warn if data is inconclusive."
    )

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        if hasattr(completion.choices[0], "message"):
            text = completion.choices[0].message.content.strip()
        else:
            text = completion.choices[0].text.strip()
        return text
    except Exception as e:
        print(f"[AI ERROR] Failed to generate commentary: {e}")
        return "No AI commentary available due to error."
