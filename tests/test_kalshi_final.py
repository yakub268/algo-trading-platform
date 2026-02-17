"""
KALSHI - Final attempt to place a trade
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
print("KALSHI - Final Trade Attempt")
print("=" * 70)

# Get balance
resp = api_request("GET", "/portfolio/balance")
if resp.status_code == 200:
    balance = resp.json().get("balance", 0) / 100
    print(f"Balance: ${balance:.2f}")

# Find markets with 1-25 cent YES prices (most likely to have liquidity)
all_markets = []
cursor = None
for page in range(10):
    params = {"status": "open", "limit": 200}
    if cursor:
        params["cursor"] = cursor
    resp = api_request("GET", "/markets", params=params)
    if resp.status_code != 200:
        break
    data = resp.json()
    all_markets.extend(data.get("markets", []))
    cursor = data.get("cursor")
    if not cursor:
        break

print(f"Fetched {len(all_markets)} markets")

# Find markets with prices in tradeable range
cheap_markets = []
for m in all_markets:
    yes_ask = m.get("yes_ask", 0)
    # Look for markets with YES ask between 1 and 50 cents
    if 1 <= yes_ask <= 50:
        cheap_markets.append((yes_ask, m))

cheap_markets.sort(key=lambda x: x[0])  # Sort by price ascending

print(f"Found {len(cheap_markets)} markets with YES ask 1-50c")

if cheap_markets:
    print("\nCheap markets (sorted by price):")
    for price, m in cheap_markets[:10]:
        ticker = m.get("ticker")
        title = m.get("title", "")[:40]
        yes_bid = m.get("yes_bid", 0)
        no_bid = m.get("no_bid", 0)
        no_ask = m.get("no_ask", 0)
        volume = m.get("volume", 0)
        print(f"  {price}c: {ticker}")
        print(f"       YES {yes_bid}/{price}, NO {no_bid}/{no_ask}, vol={volume}")
        print(f"       {title}")

    # Pick first cheap market
    price, best_market = cheap_markets[0]
    ticker = best_market.get("ticker")

    print(f"\n--- Selected Market ---")
    print(f"Ticker: {ticker}")
    print(f"Price: {price}c")
    print(f"Title: {best_market.get('title')}")

    # Try to place order - BUYING YES at the ask price
    print("\n--- Attempting Order ---")

    order_data = {
        "ticker": ticker,
        "action": "buy",
        "side": "yes",
        "count": 1,
        "type": "limit",
        "yes_price": price,
    }

    print(f"Order payload: {json.dumps(order_data, indent=2)}")

    resp = api_request("POST", "/portfolio/orders", json_data=order_data)
    print(f"Response status: {resp.status_code}")
    print(f"Response body: {resp.text}")

    if resp.status_code in [200, 201]:
        result = resp.json()
        order = result.get("order", {})
        order_id = order.get("order_id")
        status = order.get("status")
        print(f"\n*** SUCCESS ***")
        print(f"Order ID: {order_id}")
        print(f"Status: {status}")
    else:
        # Try with client_order_id
        print("\n--- Trying with client_order_id ---")
        order_data["client_order_id"] = f"test-{int(time.time() * 1000)}"
        print(f"Order payload v2: {json.dumps(order_data, indent=2)}")

        resp = api_request("POST", "/portfolio/orders", json_data=order_data)
        print(f"Response status: {resp.status_code}")
        print(f"Response body: {resp.text}")

        if resp.status_code in [200, 201]:
            result = resp.json()
            order_id = result.get("order", {}).get("order_id")
            print(f"\n*** SUCCESS ***")
            print(f"Order ID: {order_id}")
        else:
            # Parse error details
            try:
                error = resp.json()
                print(f"\nError code: {error.get('code')}")
                print(f"Error message: {error.get('message')}")
                print(f"Error details: {error.get('details')}")
            except Exception:
                pass

else:
    print("No cheap markets found!")

    # Try BTC/ETH crypto markets specifically
    print("\n--- Trying Crypto Markets ---")
    for series in ["KXBTC", "KXETH"]:
        resp = api_request(
            "GET",
            "/markets",
            params={"status": "open", "series_ticker": series, "limit": 20},
        )
        if resp.status_code == 200:
            markets = resp.json().get("markets", [])
            print(f"\n{series}: {len(markets)} markets")

            # Look for any market with ask < 50
            for m in markets:
                yes_ask = m.get("yes_ask", 0)
                no_ask = m.get("no_ask", 100)
                if 1 <= yes_ask <= 50:
                    print(f"  Found: {m.get('ticker')} @ {yes_ask}c")
                    # Try to buy
                    order_data = {
                        "ticker": m.get("ticker"),
                        "action": "buy",
                        "side": "yes",
                        "count": 1,
                        "type": "limit",
                        "yes_price": yes_ask,
                    }
                    resp = api_request(
                        "POST", "/portfolio/orders", json_data=order_data
                    )
                    print(f"  Order response: {resp.status_code} - {resp.text[:200]}")
                    if resp.status_code in [200, 201]:
                        print("  *** SUCCESS ***")
                        break
                    break

print("\n" + "=" * 70)
