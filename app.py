import os
import time
import math
import requests
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Set this as an environment variable in your hosting dashboard.
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
API_URL = "https://api.twelvedata.com/time_series"
SUPPORTED_PAIRS = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "USD/CHF": "USD/CHF",
    "NZD/USD": "NZD/USD", "EUR/GBP": "EUR/GBP"
}
CACHE = {}
CACHE_SECONDS = 20

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def fetch_candles(pair, interval, outputsize=100):
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("API key missing. Add TWELVEDATA_API_KEY in your hosting environment.")
    cache_key = (pair, interval, outputsize)
    cached = CACHE.get(cache_key)
    if cached and time.time() - cached["time"] < CACHE_SECONDS:
        return cached["values"], cached["meta"]
    params = {
        "symbol": pair, "interval": interval, "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY, "format": "JSON", "timezone": "UTC"
    }
    response = requests.get(API_URL, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error" or "values" not in payload:
        message = payload.get("message", "Provider did not return candle data.")
        raise RuntimeError(message)
    values = []
    for item in reversed(payload["values"]):
        try:
            values.append({
                "datetime": item["datetime"],
                "open": float(item["open"]), "high": float(item["high"]),
                "low": float(item["low"]), "close": float(item["close"]),
                "volume": float(item.get("volume") or 0)
            })
        except (KeyError, TypeError, ValueError):
            continue
    if len(values) < 25:
        raise RuntimeError("Not enough candle history returned by the data provider.")
    meta = {"source": "Twelve Data", "symbol": pair, "provider_interval": interval}
    CACHE[cache_key] = {"time": time.time(), "values": values, "meta": meta}
    return values, meta

def aggregate_candles(candles, minutes):
    # Aggregate UTC minute candles into fixed minute bars.
    groups = {}
    for c in candles:
        try:
            dt = datetime.fromisoformat(c["datetime"].replace("Z", "+00:00"))
            epoch_min = int(dt.timestamp() // 60)
        except Exception:
            continue
        bucket = epoch_min - (epoch_min % minutes)
        groups.setdefault(bucket, []).append(c)
    out = []
    for bucket in sorted(groups):
        rows = sorted(groups[bucket], key=lambda x: x["datetime"])
        if len(rows) < minutes:
            continue  # Ignore incomplete aggregate candles.
        out.append({
            "datetime": datetime.fromtimestamp(bucket * 60, timezone.utc).isoformat(timespec="minutes"),
            "open": rows[0]["open"],
            "high": max(x["high"] for x in rows),
            "low": min(x["low"] for x in rows),
            "close": rows[-1]["close"],
            "volume": sum(x["volume"] for x in rows)
        })
    return out

def ema(values, period):
    if len(values) < period:
        return None
    alpha = 2 / (period + 1)
    result = sum(values[:period]) / period
    for value in values[period:]:
        result = alpha * value + (1 - alpha) * result
    return result

def make_signal(candles):
    # Transparent rule-based baseline, not a trained ML model or a guarantee.
    if len(candles) < 30:
        return {"signal": "NO SIGNAL", "confidence": 0, "reason": "Not enough candles."}
    closes = [c["close"] for c in candles]
    last = candles[-1]
    previous = candles[-2]
    e9, e21 = ema(closes, 9), ema(closes, 21)
    if e9 is None or e21 is None or last["close"] == 0:
        return {"signal": "NO SIGNAL", "confidence": 0, "reason": "Indicator history unavailable."}
    recent = candles[-14:]
    ranges = [max(c["high"] - c["low"], 1e-12) for c in recent]
    avg_range = sum(ranges[:-1]) / max(1, len(ranges) - 1)
    current_range = max(last["high"] - last["low"], 1e-12)
    body = abs(last["close"] - last["open"])
    body_ratio = body / current_range
    momentum = (closes[-1] - closes[-4]) / closes[-4] if closes[-4] else 0
    volatility_ok = current_range <= avg_range * 2.5
    bull = e9 > e21 and momentum > 0 and last["close"] > last["open"] and last["close"] > previous["close"]
    bear = e9 < e21 and momentum < 0 and last["close"] < last["open"] and last["close"] < previous["close"]
    if not volatility_ok:
        return {"signal": "NO SIGNAL", "confidence": 0, "reason": "Volatility spike filter: wait for conditions to settle."}
    if body_ratio < 0.25:
        return {"signal": "NO SIGNAL", "confidence": 0, "reason": "Weak candle body; no clear momentum."}
    if bull or bear:
        # A heuristic score, not a statistically calibrated probability.
        score = 55 + min(15, abs(momentum) * 100000) + min(10, body_ratio * 10)
        return {
            "signal": "BUY" if bull else "SELL",
            "confidence": int(min(75, score)),
            "reason": "EMA trend, recent momentum, and candle direction align. Heuristic score, not win probability."
        }
    return {"signal": "NO SIGNAL", "confidence": 0, "reason": "Trend and momentum do not agree."}

@app.get("/")
def index():
    return render_template("index.html", pairs=list(SUPPORTED_PAIRS.keys()))

@app.get("/api/signal")
def signal():
    pair = request.args.get("pair", "EUR/USD").upper()
    tf = request.args.get("tf", "1m")
    if pair not in SUPPORTED_PAIRS:
        return jsonify({"error": "Unsupported pair."}), 400
    if tf not in ("1m", "3m", "5m", "15m"):
        return jsonify({"error": "Unsupported timeframe."}), 400
    try:
        if tf == "1m":
            candles, meta = fetch_candles(pair, "1min", 100)
        elif tf == "3m":
            raw, meta = fetch_candles(pair, "1min", 300)
            candles = aggregate_candles(raw, 3)
        elif tf == "5m":
            candles, meta = fetch_candles(pair, "5min", 100)
        else:
            candles, meta = fetch_candles(pair, "15min", 100)
        if len(candles) < 30:
            raise RuntimeError("Not enough complete candles for this timeframe.")
        result = make_signal(candles)
        last = candles[-1]
        return jsonify({
            "pair": pair, "timeframe": tf, "source": meta["source"],
            "market_type": "Forex spot/FX data provider; not OTC",
            "as_of_utc": utc_now(), "last_candle_utc": last["datetime"],
            "last_close": last["close"], "candles": candles[-60:],
            "signal": result["signal"], "confidence_score": result["confidence"],
            "reason": result["reason"],
            "disclaimer": "Signal is informational only. Confidence score is heuristic, not a win probability. Verify provider data and paper-test before any live decision."
        })
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "signal": "NO SIGNAL",
            "source": "Unavailable",
            "market_type": "No data; no signal generated",
            "as_of_utc": utc_now()
        }), 503

@app.get("/health")
def health():
    return jsonify({"status": "ok", "api_key_configured": bool(TWELVEDATA_API_KEY), "utc": utc_now()})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
