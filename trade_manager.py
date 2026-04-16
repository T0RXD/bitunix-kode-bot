"""
Trade Manager for Bitunix Futures - With Q-Learning Integration

Tracks the state and action taken for each position so that when a position
is closed (via SL or trailing stop), the reward (percent profit/loss) can be
computed and fed back to the Q-Learning agent.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bitunix_client import BitunixAPIError, BitunixFuturesClient

logger = logging.getLogger(__name__)

StateTuple = Tuple[int, int, int, int]


@dataclass
class Position:
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    qty: float
    stop_loss: float
    trailing_stop: float
    trailing_offset: float
    highest_price: float
    order_id: str = ""
    client_id: str = ""
    opened_at: float = field(default_factory=time.time)
    # Q-Learning tracking fields
    rl_state: Optional[StateTuple] = None
    rl_action: Optional[int] = None

    @property
    def is_long(self) -> bool:
        return self.side == "LONG"


@dataclass
class PendingEntry:
    symbol: str
    signal_candle_close: float
    atr: float
    queued_at: float = field(default_factory=time.time)
    rl_state: Optional[StateTuple] = None
    rl_action: Optional[int] = None


@dataclass
class ClosedTrade:
    """Record of a closed trade for RL learning."""
    symbol: str
    state: StateTuple
    action: int
    reward: float
    next_state: StateTuple
    entry_price: float
    exit_price: float
    pnl_pct: float
    closed_at: float = field(default_factory=time.time)


class TradeManager:
    def __init__(
        self,
        client: BitunixFuturesClient,
        max_concurrent: int = 3,
        risk_per_trade: float = 0.02,
        atr_sl_multiplier: float = 1.5,
        atr_trail_multiplier: float = 1.0,
        max_pending_age_s: float = 1200.0,
    ):
        self.client = client
        self.max_concurrent = max_concurrent
        self.risk_per_trade = risk_per_trade
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_trail_multiplier = atr_trail_multiplier
        self.max_pending_age_s = max_pending_age_s
        self.positions: Dict[str, Position] = {}
        self.pending_entries: Dict[str, PendingEntry] = {}
        self.closed_trades: List[ClosedTrade] = []

    def sync_positions(self, current_atr_provider):
        """Synchronize in-memory positions with the exchange."""
        try:
            exchange_positions = self.client.get_positions()
            active_symbols = set()

            for ep in exchange_positions:
                size = float(ep.get("size", 0))
                if size <= 0:
                    continue

                symbol = ep["symbol"]
                active_symbols.add(symbol)

                if symbol not in self.positions:
                    logger.info(f"Recovering position for {symbol} from exchange data")
                    side = "LONG" if ep.get("side") in ["BUY", "LONG"] else "SHORT"
                    entry_price = float(ep.get("entryPrice", 0))

                    atr = current_atr_provider(symbol)
                    sl_dist = atr * self.atr_sl_multiplier
                    trail_dist = atr * self.atr_trail_multiplier

                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        qty=size,
                        stop_loss=entry_price - sl_dist if side == "LONG" else entry_price + sl_dist,
                        trailing_stop=entry_price - trail_dist if side == "LONG" else entry_price + trail_dist,
                        trailing_offset=trail_dist,
                        highest_price=entry_price,
                    )

            for symbol in list(self.positions.keys()):
                if symbol not in active_symbols:
                    logger.info(f"Removing {symbol} from memory (closed on exchange)")
                    del self.positions[symbol]

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    def can_open_trade(self) -> bool:
        return len(self.positions) < self.max_concurrent

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def has_pending(self, symbol: str) -> bool:
        return symbol in self.pending_entries

    def queue_pending_entry(
        self,
        symbol: str,
        signal_candle_close: float,
        atr: float,
        rl_state: Optional[StateTuple] = None,
        rl_action: Optional[int] = None,
    ) -> bool:
        if self.has_position(symbol) or self.has_pending(symbol) or not self.can_open_trade():
            return False
        self.pending_entries[symbol] = PendingEntry(
            symbol, signal_candle_close, atr,
            rl_state=rl_state, rl_action=rl_action,
        )
        logger.info(f"Queued LONG for {symbol} (ATR={atr:.4f}, RL state={rl_state}, RL action={rl_action})")
        return True

    def execute_pending_entries(self, prices: Dict[str, float], balance: float) -> List[str]:
        now = time.time()
        opened = []
        for symbol in list(self.pending_entries.keys()):
            pending = self.pending_entries[symbol]
            if now - pending.queued_at > self.max_pending_age_s:
                del self.pending_entries[symbol]
                continue
            if symbol in prices and self.can_open_trade() and not self.has_position(symbol):
                pos = self.open_long(
                    symbol, prices[symbol], pending.atr, balance,
                    rl_state=pending.rl_state, rl_action=pending.rl_action,
                )
                if pos:
                    opened.append(symbol)
                del self.pending_entries[symbol]
        return opened

    def calculate_position_size(self, balance: float, atr: float) -> float:
        risk_amount = balance * self.risk_per_trade
        stop_dist = atr * self.atr_sl_multiplier
        return risk_amount / stop_dist if stop_dist > 0 else 0

    def open_long(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        balance: float,
        rl_state: Optional[StateTuple] = None,
        rl_action: Optional[int] = None,
    ) -> Optional[Position]:
        qty = self.calculate_position_size(balance, atr)
        if qty <= 0:
            return None

        sl_price = entry_price - (atr * self.atr_sl_multiplier)
        trail_dist = atr * self.atr_trail_multiplier

        qty_str = f"{qty:.6f}".rstrip("0").rstrip(".")
        sl_str = f"{sl_price:.2f}"
        client_id = f"KODE_{symbol}_{int(time.time())}"

        try:
            res = self.client.place_market_buy(symbol=symbol, qty=qty_str, slPrice=sl_str, clientId=client_id)
            pos = Position(
                symbol=symbol, side="LONG", entry_price=entry_price, qty=qty,
                stop_loss=sl_price, trailing_stop=entry_price - trail_dist,
                trailing_offset=trail_dist, highest_price=entry_price,
                order_id=res.get("orderId", ""), client_id=client_id,
                rl_state=rl_state, rl_action=rl_action,
            )
            self.positions[symbol] = pos
            return pos
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return None

    def _record_closed_trade(
        self,
        pos: Position,
        exit_price: float,
        next_state: StateTuple = (0, 0, 0, 0),
    ) -> Optional[ClosedTrade]:
        """Record a closed trade for RL feedback."""
        if pos.rl_state is None or pos.rl_action is None:
            return None

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
        reward = pnl_pct * 100  # reward = percent profit/loss

        trade = ClosedTrade(
            symbol=pos.symbol,
            state=pos.rl_state,
            action=pos.rl_action,
            reward=reward,
            next_state=next_state,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
        )
        self.closed_trades.append(trade)
        return trade

    def close_long(
        self,
        symbol: str,
        exit_price: float = 0.0,
        next_state: StateTuple = (0, 0, 0, 0),
    ) -> bool:
        if symbol not in self.positions:
            return False
        pos = self.positions[symbol]
        try:
            self.client.close_position(symbol, "SELL", f"{pos.qty:.6f}".rstrip("0").rstrip("."))
            self._record_closed_trade(pos, exit_price or pos.entry_price, next_state)
            del self.positions[symbol]
            return True
        except Exception as e:
            logger.error(f"Close failed for {symbol}: {e}")
            return False

    def check_stop_losses(
        self,
        prices: Dict[str, float],
        current_states: Optional[Dict[str, StateTuple]] = None,
    ) -> List[str]:
        closed = []
        states = current_states or {}
        for sym, pos in list(self.positions.items()):
            if sym not in prices:
                continue
            price = prices[sym]
            next_state = states.get(sym, (0, 0, 0, 0))

            # Fixed SL
            if price <= pos.stop_loss:
                if self.close_long(sym, exit_price=price, next_state=next_state):
                    closed.append(sym)
                continue

            # Trailing Stop
            if price <= pos.trailing_stop:
                if self.close_long(sym, exit_price=price, next_state=next_state):
                    closed.append(sym)
                continue

            # Update Trail
            if price > pos.highest_price:
                pos.highest_price = price
                new_trail = price - pos.trailing_offset
                if new_trail > pos.trailing_stop:
                    pos.trailing_stop = new_trail

        return closed

    def get_and_clear_closed_trades(self) -> List[ClosedTrade]:
        trades = list(self.closed_trades)
        self.closed_trades.clear()
        return trades
