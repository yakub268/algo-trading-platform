"""
ORDER FILL HELPER
=================
Core fix for zero-P&L bug: Poll for confirmed fill prices instead of reading
filled_avg_price from submit_order() immediate response (which is always None).

CRITICAL: NEVER read filled_avg_price from submit_order() response.
ALWAYS poll get_order_by_id() until status == 'filled'.
"""

import time
import logging
from typing import Dict, Optional

logger = logging.getLogger('OrderFillHelper')


def submit_and_wait_for_fill(trading_client, order_data, timeout=30) -> Dict:
    """
    Submit order and poll until filled. Returns confirmed fill price.

    Args:
        trading_client: Alpaca TradingClient instance
        order_data: Order request object (MarketOrderRequest, etc.)
        timeout: Max seconds to wait for fill (default: 30)

    Returns:
        Dict with:
            - order_id: Order ID string
            - fill_price: Confirmed fill price (float, validated > 0)
            - fill_qty: Confirmed fill quantity (float, validated > 0)
            - status: 'filled'

    Raises:
        ValueError: If fill price or quantity is invalid (≤ 0)
        RuntimeError: If order failed (canceled, rejected, etc.)
        TimeoutError: If order not filled within timeout
    """
    # Submit the order
    order = trading_client.submit_order(order_data)
    order_id = str(order.id)

    logger.info(f"Order {order_id} submitted, polling for fill...")

    start = time.time()
    poll_count = 0

    while time.time() - start < timeout:
        poll_count += 1

        # Poll for updated order status
        updated = trading_client.get_order_by_id(order_id)

        if updated.status.value == 'filled':
            # Order filled - extract and validate fill data
            fill_price = float(updated.filled_avg_price) if updated.filled_avg_price else 0
            fill_qty = float(updated.filled_qty) if updated.filled_qty else 0

            # CRITICAL VALIDATION - never accept zero or None
            if fill_price <= 0:
                raise ValueError(
                    f"Order {order_id} filled but price is {fill_price}. "
                    f"This indicates a bug in fill price capture."
                )
            if fill_qty <= 0:
                raise ValueError(
                    f"Order {order_id} filled but qty is {fill_qty}. "
                    f"This indicates a bug in fill quantity capture."
                )

            logger.info(
                f"Order {order_id} FILLED after {poll_count} polls "
                f"({time.time() - start:.2f}s): "
                f"{fill_qty} @ ${fill_price:.6f}"
            )

            return {
                'order_id': order_id,
                'fill_price': fill_price,
                'fill_qty': fill_qty,
                'status': 'filled'
            }

        # Check for failure states
        if updated.status.value in ('canceled', 'expired', 'rejected', 'suspended'):
            raise RuntimeError(
                f"Order {order_id} failed with status: {updated.status.value}"
            )

        # Still pending - wait and retry
        if poll_count % 10 == 0:
            logger.debug(
                f"Order {order_id} still {updated.status.value} "
                f"after {time.time() - start:.2f}s"
            )

        time.sleep(0.5)

    # Timeout - attempt to cancel the order
    logger.warning(f"Order {order_id} not filled within {timeout}s, attempting cancel")
    try:
        trading_client.cancel_order_by_id(order_id)
        logger.info(f"Order {order_id} canceled due to timeout")
    except Exception as cancel_err:
        logger.error(f"Failed to cancel order {order_id}: {cancel_err}")

    raise TimeoutError(
        f"Order {order_id} not filled within {timeout}s. "
        f"Final status: {updated.status.value if 'updated' in locals() else 'unknown'}"
    )


def validate_fill_price_before_db(entry_price: float, exit_price: Optional[float] = None) -> None:
    """
    Validate fill prices before storing to database.
    Raises ValueError if prices are invalid.

    Args:
        entry_price: Entry fill price
        exit_price: Exit fill price (optional, for exit validation)
    """
    if entry_price <= 0:
        raise ValueError(f"Invalid entry_price: {entry_price} (must be > 0)")

    if exit_price is not None:
        if exit_price <= 0:
            raise ValueError(f"Invalid exit_price: {exit_price} (must be > 0)")

        # Check for suspicious identical prices (possible bug indicator)
        if abs(exit_price - entry_price) < 0.0000001:
            logger.warning(
                f"SUSPICIOUS: entry_price ({entry_price}) ≈ exit_price ({exit_price}). "
                f"This may indicate a fill price bug. Prices are valid but identical."
            )
