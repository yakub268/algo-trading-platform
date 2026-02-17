"""
KALSHI DEBUG - Test order placement with proper API format
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

print("=" * 70)
print("KALSHI DEBUG - Direct API Test")
print("=" * 70)

# Load credentials
API_KEY_ID = os.getenv("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY_ID")
PRIVATE_KEY_PATH = os.path.expanduser(
    os.getenv("KALSHI_PRIVATE_KEY_PATH", "~/.trading_keys/kalshi_private_key.pem")
)
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

print(f"API Key ID: {API_KEY_ID[:8]}..." if API_KEY_ID else "API Key ID: NOT SET")
print(f"Private Key Path: {PRIVATE_KEY_PATH}")

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
        print(f"Loaded private key from: {path}")
        break

if not private_key:
    print("FAILURE: Private key not found")
    sys.exit(1)


def sign_request(method: str, path: str, timestamp: int) -> str:
    """Create RSA-PSS signature for Kalshi API request."""
    message = f"{timestamp}{method}{path}"
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def api_request(method: str, endpoint: str, json_data=None):
    """Make authenticated request to Kalshi API."""
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
        response = requests.get(url, headers=headers)
    else:
        response = requests.post(url, headers=headers, json=json_data)

    return response


# Test 1: Get balance
print("\n--- Test 1: Get Balance ---")
resp = api_request("GET", "/portfolio/balance")
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    balance_cents = data.get("balance", 0)
    print(f"Balance: ${balance_cents/100:.2f}")
else:
    print(f"Error: {resp.text[:200]}")

# Test 2: Get markets and find one to trade
print("\n--- Test 2: Get Markets ---")
resp = api_request("GET", "/markets?status=open&limit=100")
print(f"Status: {resp.status_code}")

if resp.status_code != 200:
    print(f"Error: {resp.text[:200]}")
    sys.exit(1)

markets = resp.json().get("markets", [])
print(f"Found {len(markets)} markets")

# Find tradeable market
best_market = None
for market in markets:
    yes_ask = market.get("yes_ask")
    no_ask = market.get("no_ask")

    # Look for market with reasonable price
    if yes_ask and 1 < yes_ask < 95:
        best_market = market
        side = "yes"
        price = yes_ask
        break
    elif no_ask and 1 < no_ask < 95:
        best_market = market
        side = "no"
        price = no_ask
        break

if not best_market:
    print("No suitable market found")
    # Show some markets for debugging
    print("\nFirst 5 markets:")
    for m in markets[:5]:
        print(
            f"  {m.get('ticker')}: yes_ask={m.get('yes_ask')}, no_ask={m.get('no_ask')}"
        )
    sys.exit(1)

ticker = best_market.get("ticker")
title = best_market.get("title", "")[:60]
print(f"\nSelected: {ticker}")
print(f"Title: {title}")
print(f"Side: {side}, Price: {price}c")

# Test 3: Get orderbook for this market
print("\n--- Test 3: Get Orderbook ---")
resp = api_request("GET", f"/markets/{ticker}/orderbook")
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    orderbook = resp.json().get("orderbook", {})
    print(f"Orderbook: {json.dumps(orderbook, indent=2)[:500]}")
else:
    print(f"Error: {resp.text[:200]}")

# Test 4: Place order with correct API format
print("\n--- Test 4: Place Order ---")

# According to Kalshi API docs, the order format is:
# POST /portfolio/orders
# Body: { "ticker": str, "action": "buy"|"sell", "side": "yes"|"no",
#         "count": int, "type": "limit"|"market", "yes_price": int (if side=yes), "no_price": int (if side=no) }

# Try the documented format
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

print(f"Order request: {json.dumps(order_data, indent=2)}")

resp = api_request("POST", "/portfolio/orders", json_data=order_data)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:500]}")

if resp.status_code in [200, 201]:
    result = resp.json()
    order = result.get("order", {})
    order_id = order.get("order_id")
    print(f"\nSUCCESS!")
    print(f"Order ID: {order_id}")
    print(f"Full response: {json.dumps(result, indent=2)[:800]}")
else:
    print(f"\nFAILURE: {resp.status_code}")
    print(f"Error details: {resp.text}")

    # Try alternative format without action field
    print("\n--- Trying alternative format ---")
    order_data2 = {
        "ticker": ticker,
        "client_order_id": f"test-{int(time.time())}",
        "side": side,
        "count": 1,
        "type": "limit",
    }
    if side == "yes":
        order_data2["yes_price"] = price
    else:
        order_data2["no_price"] = price

    print(f"Order request v2: {json.dumps(order_data2, indent=2)}")
    resp2 = api_request("POST", "/portfolio/orders", json_data=order_data2)
    print(f"Status: {resp2.status_code}")
    print(f"Response: {resp2.text[:500]}")

print("\n" + "=" * 70)
