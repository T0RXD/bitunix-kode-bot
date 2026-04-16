#!/usr/bin/env python3
"""
Bitunix Futures Bot - With Q-Learning Reinforcement Learning Integration

The Q-Learning agent supplements the hardcoded strategy by deciding whether
the bot should actually enter a trade when a signal fires. When a position
is closed, the agent learns from the outcome (reward = percent profit/loss).
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from bitunix_client import BitunixAPIError, BitunixFuturesClient
from learning_agent import ACTION_ENTER_LONG, ACTION_NO_OP, QLearningAgent
from strategy import StateTuple, analyze_symbol, extract_state_tuple, get_current_atr
from trade_manager import TradeManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_running = True


def _shutdown(signum, frame):
    global _running
    logger.info("Shutdown signal received. Cleaning up...")
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(klines)

    # Handle list-of-lists format
    if isinstance(klines[0], (list, tuple)):
        col_names = ["ts", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(klines, columns=col_names[: len(klines[0])])
    else:
        rename_map = {
            "openPrice": "open", "highPrice": "high", "lowPrice": "low",
            "closePrice": "close", "baseVol": "volume", "openTime": "ts",
            "o": "open", "h": "high", "l": "low", "c": "close",
            "vol": "volume", "time": "ts",
        }
        for k, v in rename_map.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ts" in df.columns:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def get_top_symbols(client: BitunixFuturesClient, top_n: int = 15) -> List[str]:
    """Return top symbols sorted by quote volume."""
    try:
        tickers = client.get_tickers()
        tickers.sort(key=lambda x: float(x.get("quoteVol", 0)), reverse=True)
        return [t["symbol"] for t in tickers[:top_n]]
    except BitunixAPIError as e:
        logger.error(f"Failed to get tickers: {e}")
        return []


def fetch_closed_klines(
    client: BitunixFuturesClient,
    symbol: str,
    interval: str = "15",
    limit: int = 60,
) -> pd.DataFrame:
    """Fetch klines and drop the last (possibly forming) candle."""
    try:
        raw = client.get_klines(symbol, interval=interval, limit=limit)
        df = klines_to_dataframe(raw)
        if len(df) > 1:
            return df.iloc[:-1].reset_index(drop=True)
        return df
    except BitunixAPIError as e:
        logger.error(f"Failed to fetch klines for {symbol}: {e}")
        return pd.DataFrame()


def run_strategy(client: BitunixFuturesClient, args: argparse.Namespace):
    # Initialize Q-Learning agent
    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        q_table_path=args.q_table_path,
    )

    manager = TradeManager(
        client=client,
        max_concurrent=args.max_trades,
        risk_per_trade=args.risk,
        atr_sl_multiplier=1.5,
        atr_trail_multiplier=1.0,
    )

    # RECOVERY: Sync positions on startup
    def atr_provider(symbol):
        raw = client.get_klines(symbol, limit=20)
        return get_current_atr(klines_to_dataframe(raw))

    logger.info("Syncing positions with exchange...")
    manager.sync_positions(atr_provider)

    while _running:
        try:
            # 1. Update Market Info
            balance = float(client.get_balance("USDT"))
            tickers = client.get_tickers()
            tickers.sort(key=lambda x: float(x.get("quoteVol", 0)), reverse=True)
            top_symbols = [t["symbol"] for t in tickers[:args.top]]
            prices = {t["symbol"]: float(t["lastPrice"]) for t in tickers}

            # 2. Check Exits (Stops) — with current states for RL feedback
            manager.check_stop_losses(prices)

            # 3. Execute Pending Entries (Next Candle)
            manager.execute_pending_entries(prices, balance)

            # 4. Learn from closed trades
            closed_trades = manager.get_and_clear_closed_trades()
            for trade in closed_trades:
                agent.update(trade.state, trade.action, trade.reward, trade.next_state)

            # 5. Scan for New Signals
            for sym in top_symbols:
                if not _running:
                    break
                if manager.has_position(sym) or manager.has_pending(sym):
                    continue
                if not manager.can_open_trade():
                    break

                df_closed = fetch_closed_klines(client, sym)
                if len(df_closed) < 52:
                    continue

                signal_fired, state = analyze_symbol(df_closed)

                if signal_fired:
                    # Ask Q-Learning agent whether to actually enter
                    action = agent.get_action(state, epsilon=args.epsilon)

                    if action == ACTION_ENTER_LONG:
                        atr = get_current_atr(df_closed)
                        manager.queue_pending_entry(
                            sym,
                            float(df_closed.iloc[-1]["close"]),
                            atr,
                            rl_state=state,
                            rl_action=action,
                        )

            logger.info(
                f"Status: {len(manager.positions)} Open, "
                f"{len(manager.pending_entries)} Pending. Bal: {balance:.2f} USDT"
            )

        except BitunixAPIError as e:
            logger.error(f"API Error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected Error: {e}")

        time.sleep(args.poll)

    # Save Q-Table on shutdown
    agent.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitunix Futures Bot with Q-Learning RL")
    parser.add_argument("--top", type=int, default=15, help="Top N symbols by volume to scan")
    parser.add_argument("--poll", type=int, default=30, help="Poll interval in seconds")
    parser.add_argument("--max-trades", type=int, default=3, help="Max concurrent trades")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade as fraction of balance")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Q-Learning exploration rate (0=exploit, 1=explore)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Q-Learning learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Q-Learning discount factor")
    parser.add_argument("--q-table-path", type=str, default="q_table.npy", help="Path to Q-Table file")
    args = parser.parse_args()

    client = BitunixFuturesClient()
    run_strategy(client, args)
