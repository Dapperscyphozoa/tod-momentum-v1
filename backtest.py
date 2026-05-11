"""
backtest.py — engine-local driver. Uses the shared backtester.py harness.

Run:
    python3 backtest.py             # default 60d on ACTIVE_UNIVERSE
    python3 backtest.py 30          # 30 days
    python3 backtest.py 90 BTC,ETH  # 90 days, specific coins

Outputs BACKTEST_RESULTS.md in the repo root.
"""
from __future__ import annotations
import os
import sys
import time

# Ensure config defaults — backtest doesn't need HL/PM env vars
os.environ.setdefault("ENGINE_NAME", "backtest")
os.environ.setdefault("STATE_DIR", "/tmp/backtest-state")

from backtester import fetch_hl_candles, run_backtest, sweep_results
from engine.config import ACTIVE_UNIVERSE, TRADE_PARAMS, ENGINE_NAME
from engine.signal_detector import evaluate_latest_bar


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    coins = (sys.argv[2].split(",") if len(sys.argv) > 2
             else ACTIVE_UNIVERSE)

    print(f"=== {ENGINE_NAME} backtest | {days}d | {len(coins)} coins ===")
    results = []
    for coin in coins:
        print(f"  fetching {coin}...", end=" ", flush=True)
        bars = fetch_hl_candles(coin, days=days, interval="1h")
        if len(bars) < 200:
            print(f"insufficient bars ({len(bars)})")
            results.append({"coin": coin, "n_trades": 0, "n_bars": len(bars)})
            continue
        print(f"{len(bars)} bars  →  running...", end=" ", flush=True)
        r = run_backtest(coin, bars, evaluate_latest_bar, TRADE_PARAMS)
        results.append(r)
        if r.get("n_trades", 0) > 0:
            print(f"n={r['n_trades']} WR={r['wr_pct']}% PF={r['pf']} sumR={r['sum_r']}")
        else:
            print("no fires")
        time.sleep(0.3)   # rate-limit courtesy

    md = sweep_results(results, out_path="BACKTEST_RESULTS.md",
                       engine_name=ENGINE_NAME)
    print("\n=== summary ===")
    print(md[:2000])


if __name__ == "__main__":
    main()
