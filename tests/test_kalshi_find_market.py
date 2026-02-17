"""
KALSHI - Find any tradeable market
"""

import os
import sys
import json
import time
import base64
import requests
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# Load credentials
API_KEY_ID = os.getenv("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY_ID")
PRIVATE_KEY_PATH = os.path.expanduser(
    os.getenv("KALSHI_PRIVATE_KEY_PATH", "~/.trading_keys/kalshi_private_key.pem")
)
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Load private key
key_paths = [
    PRIVATE_KEY_PATH,
    os.path.join(os.path.dirname(__file__), PRIVATE_KEY_PATH),
]
private_key = None
for path in key_paths:
    if os.path.exists(path):
        with open(path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        break


def sign_request(method: str, path: str, timestamp: int) -> str:
    message = f"{timestamp}{method}{path}"
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def api_request(method: str, endpoint: str, json_data=None, params=None):
    timestamp = int(time.time() * 1000)
    path = f"/trade-api/v2{endpoint}"
    signature = sign_request(method.upper(), path, timestamp)

    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
        "Content-Type": "application/json",
    }

    url = f"{BASE_URL}{endpoint}"
    if method.upper() == "GET":
        return requests.get(url, headers=headers, params=params)
    else:
        return requests.post(url, headers=headers, json=json_data)


print("=" * 70)
print("KALSHI - Finding Tradeable Markets")
print("=" * 70)

# Get ALL open markets with pagination
all_markets = []
cursor = None
page = 0

while True:
    page += 1
    params = {"status": "open", "limit": 200}
    if cursor:
        params["cursor"] = cursor

    resp = api_request("GET", "/markets", params=params)
    if resp.status_code != 200:
        print(f"Error fetching markets: {resp.text[:200]}")
        break

    data = resp.json()
    markets = data.get("markets", [])
    all_markets.extend(markets)

    cursor = data.get("cursor")
    print(f"Page {page}: fetched {len(markets)} markets, total={len(all_markets)}")

    if not cursor or len(markets) == 0:
        break

    if page >= 10:  # Limit to 10 pages = 2000 markets
        break

print(f"\nTotal markets: {len(all_markets)}")

# Analyze price distribution
tradeable = []
price_stats = {
    "zero": 0,
    "1-10": 0,
    "10-25": 0,
    "25-50": 0,
    "50-75": 0,
    "75-90": 0,
    "90-99": 0,
    "hundred": 0,
}

for m in all_markets:
    yes_ask = m.get("yes_ask")
    no_ask = m.get("no_ask")
    yes_bid = m.get("yes_bid")
    no_bid = m.get("no_bid")

    # Track price distribution
    if yes_ask == 0:
        price_stats["zero"] += 1
    elif yes_ask == 100:
        price_stats["hundred"] += 1
    elif 1 <= yes_ask <= 10:
        price_stats["1-10"] += 1
    elif 10 < yes_ask <= 25:
        price_stats["10-25"] += 1
    elif 25 < yes_ask <= 50:
        price_stats["25-50"] += 1
    elif 50 < yes_ask <= 75:
        price_stats["50-75"] += 1
    elif 75 < yes_ask <= 90:
        price_stats["75-90"] += 1
    elif 90 < yes_ask < 100:
        price_stats["90-99"] += 1

    # Find tradeable markets (with real bid/ask spread)
    if yes_ask and no_ask and 1 < yes_ask < 99 and 1 < no_ask < 99:
        tradeable.append(m)
    elif yes_bid and yes_ask and yes_bid > 0 and yes_ask < 100:
        tradeable.append(m)
    elif no_bid and no_ask and no_bid > 0 and no_ask < 100:
        tradeable.append(m)

print("\nPrice distribution (yes_ask):")
for k, v in price_stats.items():
    print(f"  {k}: {v}")

print(f"\nTradeable markets (with real spread): {len(tradeable)}")

if tradeable:
    print("\nFirst 10 tradeable markets:")
    for m in tradeable[:10]:
        ticker = m.get("ticker")
        title = m.get("title", "")[:40]
        yes_bid = m.get("yes_bid", 0)
        yes_ask = m.get("yes_ask", 0)
        no_bid = m.get("no_bid", 0)
        no_ask = m.get("no_ask", 0)
        volume = m.get("volume", 0)
        print(
            f"  {ticker}: YES {yes_bid}/{yes_ask}, NO {no_bid}/{no_ask}, vol={volume}"
        )
        print(f"    {title}")

    # Pick best market
    best = tradeable[0]
    ticker = best.get("ticker")
    yes_ask = best.get("yes_ask", 50)
    no_ask = best.get("no_ask", 50)

    if yes_ask and 1 < yes_ask < 99:
        side = "yes"
        price = yes_ask
    else:
        side = "no"
        price = no_ask

    print(f"\n--- Placing Order ---")
    print(f"Market: {ticker}")
    print(f"Side: {side}, Price: {price}c")

    order_data = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": 1,
        "type": "limit",
    }
    if side == "yes":
        order_data["yes_price"] = price
    else:
        order_data["no_price"] = price

    print(f"Order: {json.dumps(order_data)}")

    resp = api_request("POST", "/portfolio/orders", json_data=order_data)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")

    if resp.status_code in [200, 201]:
        result = resp.json()
        order_id = result.get("order", {}).get("order_id")
        print(f"\nSUCCESS! Order ID: {order_id}")
else:
    # Try getting events and see individual markets
    print("\n--- Trying Events Endpoint ---")
    resp = api_request("GET", "/events", params={"status": "open", "limit": 20})
    if resp.status_code == 200:
        events = resp.json().get("events", [])
        print(f"Found {len(events)} events")
        for evt in events[:5]:
            print(f"  {evt.get('event_ticker')}: {evt.get('title', '')[:50]}")
            print(
                f"    Category: {evt.get('category')}, Markets: {evt.get('markets_count')}"
            )

    # Also try getting a specific series
    print("\n--- Checking series with actual liquidity ---")
    series_to_check = ["KXBTC", "KXETH", "FED", "INXD"]
    for series in series_to_check:
        resp = api_request(
            "GET",
            "/markets",
            params={"status": "open", "series_ticker": series, "limit": 10},
        )
        if resp.status_code == 200:
            markets = resp.json().get("markets", [])
            if markets:
                print(f"\n{series}: {len(markets)} markets")
                for m in markets[:3]:
                    print(
                        f"  {m.get('ticker')}: YES {m.get('yes_bid')}/{m.get('yes_ask')}, "
                        f"NO {m.get('no_bid')}/{m.get('no_ask')}"
                    )

print("\n" + "=" * 70)
