"""
Order management with Bybit API integration.
Handles order placement, tracking, and fills.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hmac
import hashlib
import time
import json

import aiohttp

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Fill:
    """Trade fill."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: datetime


@dataclass
class OrderManagerConfig:
    """Order manager configuration."""
    api_key: str = ""
    api_secret: str = ""
    mode: str = "demo"  # "demo", "testnet", or "live"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class BybitOrderManager:
    """
    Manages orders on Bybit exchange.

    Handles:
    - Order submission
    - Order tracking
    - Fill management
    - Position queries
    """

    def __init__(self, config: OrderManagerConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

        # URLs
        # Demo and Live use production URLs (demo is distinguished by credentials)
        # Testnet uses separate testnet URLs
        if config.mode == "testnet":
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []

        # Position cache
        self.positions: Dict[str, Dict] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
        return self._session

    def _generate_signature(
        self,
        timestamp: int,
        params: Dict
    ) -> str:
        """Generate HMAC signature for authenticated requests."""
        param_str = f"{timestamp}{self.config.api_key}{5000}"  # 5000ms recv_window

        if params:
            param_str += json.dumps(params, separators=(',', ':'))

        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False
    ) -> Optional[Dict]:
        """Make API request."""
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        headers = {
            "Content-Type": "application/json"
        }

        if signed:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(timestamp, params or {})

            headers.update({
                "X-BAPI-API-KEY": self.config.api_key,
                "X-BAPI-TIMESTAMP": str(timestamp),
                "X-BAPI-SIGN": signature,
                "X-BAPI-RECV-WINDOW": "5000"
            })

        for attempt in range(self.config.max_retries):
            try:
                if method == "GET":
                    async with session.get(url, params=params, headers=headers) as resp:
                        data = await resp.json()
                else:
                    async with session.post(url, json=params, headers=headers) as resp:
                        data = await resp.json()

                if data.get("retCode") == 0:
                    return data.get("result")
                else:
                    logger.error(f"API error: {data.get('retMsg')}")
                    return None

            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)

        return None

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        reduce_only: bool = False
    ) -> Optional[Order]:
        """
        Submit an order to Bybit.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            side: Buy or Sell
            order_type: Market or Limit
            quantity: Order quantity
            price: Limit price (required for limit orders)
            reduce_only: If True, only reduces position

        Returns:
            Order object if successful
        """
        order_id = f"bot_{int(time.time() * 1000)}"

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING
        )

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.value,
            "orderType": order_type.value,
            "qty": str(quantity),
            "orderLinkId": order_id,
            "reduceOnly": reduce_only
        }

        if order_type == OrderType.LIMIT and price:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"

        result = await self._request("POST", "/v5/order/create", params, signed=True)

        if result:
            order.exchange_order_id = result.get("orderId")
            order.status = OrderStatus.SUBMITTED
            logger.info(f"Order submitted: {order_id} -> {order.exchange_order_id}")
        else:
            order.status = OrderStatus.REJECTED
            order.error_message = "Failed to submit order"
            logger.error(f"Order rejected: {order_id}")

        self.orders[order_id] = order
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if not order or not order.exchange_order_id:
            return False

        params = {
            "category": "linear",
            "symbol": order.symbol,
            "orderId": order.exchange_order_id
        }

        result = await self._request("POST", "/v5/order/cancel", params, signed=True)

        if result:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            logger.info(f"Order cancelled: {order_id}")
            return True

        return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status."""
        order = self.orders.get(order_id)
        if not order or not order.exchange_order_id:
            return order

        params = {
            "category": "linear",
            "symbol": order.symbol,
            "orderId": order.exchange_order_id
        }

        result = await self._request("GET", "/v5/order/realtime", params, signed=True)

        if result and "list" in result and len(result["list"]) > 0:
            order_data = result["list"][0]

            # Update order
            status_map = {
                "New": OrderStatus.SUBMITTED,
                "PartiallyFilled": OrderStatus.PARTIAL,
                "Filled": OrderStatus.FILLED,
                "Cancelled": OrderStatus.CANCELLED,
                "Rejected": OrderStatus.REJECTED
            }

            order.status = status_map.get(
                order_data.get("orderStatus"),
                OrderStatus.SUBMITTED
            )
            order.filled_quantity = float(order_data.get("cumExecQty", 0))
            order.avg_fill_price = float(order_data.get("avgPrice", 0))
            order.updated_at = datetime.now()

        return order

    async def get_positions(self) -> Dict[str, Dict]:
        """Get all open positions."""
        params = {
            "category": "linear",
            "settleCoin": "USDT"
        }

        result = await self._request("GET", "/v5/position/list", params, signed=True)

        if result and "list" in result:
            self.positions = {}
            for pos in result["list"]:
                symbol = pos.get("symbol")
                size = float(pos.get("size", 0))

                if size != 0:
                    self.positions[symbol] = {
                        "size": size,
                        "side": pos.get("side"),
                        "entry_price": float(pos.get("avgPrice", 0)),
                        "unrealized_pnl": float(pos.get("unrealisedPnl", 0)),
                        "leverage": float(pos.get("leverage", 1)),
                        "liq_price": float(pos.get("liqPrice", 0))
                    }

        return self.positions

    async def get_account_balance(self) -> Dict:
        """Get account balance."""
        params = {
            "accountType": "UNIFIED"
        }

        result = await self._request("GET", "/v5/account/wallet-balance", params, signed=True)

        if result and "list" in result:
            for account in result["list"]:
                if account.get("accountType") == "UNIFIED":
                    coins = account.get("coin", [])
                    for coin in coins:
                        if coin.get("coin") == "USDT":
                            return {
                                "total": float(coin.get("walletBalance", 0)),
                                "available": float(coin.get("availableToWithdraw", 0)),
                                "unrealized_pnl": float(coin.get("unrealisedPnl", 0))
                            }

        return {"total": 0, "available": 0, "unrealized_pnl": 0}

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }

        result = await self._request("POST", "/v5/position/set-leverage", params, signed=True)
        return result is not None

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


class OrderBookAwareOrderManager(BybitOrderManager):
    """
    Order manager that uses order book for intelligent order placement.
    """

    def __init__(self, config: OrderManagerConfig):
        super().__init__(config)
        self.orderbook_cache: Dict[str, Dict] = {}

    def update_orderbook(self, symbol: str, bids: List, asks: List):
        """Update order book cache."""
        self.orderbook_cache[symbol] = {
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now()
        }

    def get_optimal_limit_price(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        aggression: float = 0.5
    ) -> Optional[float]:
        """
        Calculate optimal limit price based on order book.

        Args:
            symbol: Trading symbol
            side: Buy or Sell
            quantity: Order quantity
            aggression: 0 = passive (back of book), 1 = aggressive (cross spread)

        Returns:
            Optimal limit price
        """
        if symbol not in self.orderbook_cache:
            return None

        book = self.orderbook_cache[symbol]
        bids = book["bids"]
        asks = book["asks"]

        if not bids or not asks:
            return None

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid

        if side == OrderSide.BUY:
            # Buy: start from best bid, move toward best ask based on aggression
            price = best_bid + spread * aggression
            # Round down for buy
            tick = self._get_tick_size(symbol)
            price = (price // tick) * tick
        else:
            # Sell: start from best ask, move toward best bid based on aggression
            price = best_ask - spread * aggression
            # Round up for sell
            tick = self._get_tick_size(symbol)
            price = ((price // tick) + 1) * tick

        return price

    def _get_tick_size(self, symbol: str) -> float:
        """Get tick size for symbol."""
        # Common tick sizes for Bybit perpetuals
        tick_sizes = {
            "BTCUSDT": 0.1,
            "ETHUSDT": 0.01,
            "SOLUSDT": 0.001
        }
        return tick_sizes.get(symbol, 0.01)

    def estimate_slippage(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> float:
        """
        Estimate slippage for a market order.

        Returns slippage as a percentage.
        """
        if symbol not in self.orderbook_cache:
            return 0.0

        book = self.orderbook_cache[symbol]

        if side == OrderSide.BUY:
            levels = book["asks"]
        else:
            levels = book["bids"]

        if not levels:
            return 0.0

        best_price = float(levels[0][0])
        remaining = quantity
        total_cost = 0.0

        for level in levels:
            price = float(level[0])
            size = float(level[1])

            filled = min(remaining, size)
            total_cost += filled * price
            remaining -= filled

            if remaining <= 0:
                break

        if remaining > 0:
            # Not enough liquidity
            return 1.0  # 100% slippage indicator

        avg_price = total_cost / quantity
        slippage = abs(avg_price - best_price) / best_price

        return slippage


class SimulatedOrderManager:
    """
    Simulated order manager for backtesting and paper trading.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict] = {}
        self.fills: List[Fill] = []
        self.order_counter = 0

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float,
        slippage_bps: float = 5.0
    ) -> Order:
        """Submit simulated order with instant fill."""
        self.order_counter += 1
        order_id = f"sim_{self.order_counter}"

        # Apply slippage
        slippage = price * slippage_bps / 10000
        if side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            avg_fill_price=fill_price
        )

        # Record fill
        self.fills.append(Fill(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            fee=quantity * fill_price * 0.0006,  # 0.06% fee
            timestamp=datetime.now()
        ))

        # Update position
        self._update_position(symbol, side, quantity, fill_price)

        self.orders[order_id] = order
        return order

    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float
    ):
        """Update position after fill."""
        if symbol not in self.positions:
            self.positions[symbol] = {"size": 0, "entry_price": 0}

        pos = self.positions[symbol]
        current_size = pos["size"]

        if side == OrderSide.BUY:
            new_size = current_size + quantity
        else:
            new_size = current_size - quantity

        if new_size != 0 and current_size != 0 and np.sign(new_size) == np.sign(current_size):
            # Average entry price
            pos["entry_price"] = (pos["entry_price"] * abs(current_size) + price * quantity) / abs(new_size)
        else:
            pos["entry_price"] = price

        pos["size"] = new_size

    def get_position(self, symbol: str) -> Dict:
        """Get position for symbol."""
        return self.positions.get(symbol, {"size": 0, "entry_price": 0})

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L."""
        pos = self.get_position(symbol)
        if pos["size"] == 0:
            return 0.0

        return pos["size"] * (current_price - pos["entry_price"])
