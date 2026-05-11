"""
backtester.py — generic harness for any multica engine.

Each engine adds a `backtest.py` driver that:
  1. Imports its `engine.signal_detector.evaluate_latest_bar`
  2. Imports its `engine.config.{ACTIVE_UNIVERSE, TRADE_PARAMS}`
  3. Calls `run_backtest(...)` from this module

This file is dropped INTO each engine repo by `commit_backtest.py` so engines
don't share imports across repos.

Usage in engine fork:

    # backtest.py
    import json, sys
    from backtester import run_backtest, sweep_results
    from engine.config import ACTIVE_UNIVERSE, TRADE_PARAMS
    from engine.signal_detector import evaluate_latest_bar

    results = []
    for coin in ACTIVE_UNIVERSE:
        bars = fetch_hl(coin, days=60)
        r = run_backtest(coin, bars, evaluate_latest_bar, TRADE_PARAMS)
        results.append(r)
    sweep_results(results, out_path="BACKTEST_RESULTS.md")
"""
from __future__ import annotations
import json
import time
import urllib.request
import statistics
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional, List
import pandas as pd


# ────────────────────────────────────────────────────────────────────────
# HL historical fetch
# ────────────────────────────────────────────────────────────────────────
def fetch_hl_candles(coin: str, days: int = 60,
                      interval: str = "1h") -> pd.DataFrame:
    """Pull historical OHLCV from HL. Returns DataFrame indexed by UTC timestamps."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    body = json.dumps({"type": "candleSnapshot",
                       "req": {"coin": coin, "interval": interval,
                                "startTime": start_ms, "endTime": end_ms}}).encode()
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=body, headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=20).read())
    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t").rename(
        columns={"o": "open", "h": "high", "l": "low",
                  "c": "close", "v": "volume"})
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    df.attrs["coin"] = coin
    return df


# ────────────────────────────────────────────────────────────────────────
# Trade simulator
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    coin: str
    fire_ts: str
    entry_px: float
    sl_px: float
    tp_px: float
    is_long: bool
    max_hold_bars: int
    fire_reason: str
    exit_ts: str = ""
    exit_px: float = 0.0
    close_reason: str = ""
    pnl_pct: float = 0.0
    pnl_r: float = 0.0   # P/L in R-multiples


def _resolve(trade: Trade, bars_ahead: pd.DataFrame) -> Trade:
    """Walk forward from entry, resolve SL / TP / time-stop."""
    if len(bars_ahead) == 0:
        trade.close_reason = "no_data"
        return trade
    n = min(trade.max_hold_bars, len(bars_ahead))
    for i in range(n):
        bar = bars_ahead.iloc[i]
        h, l = float(bar["high"]), float(bar["low"])
        if trade.is_long:
            if l <= trade.sl_px:
                trade.exit_px = trade.sl_px
                trade.close_reason = "SL"
                break
            if h >= trade.tp_px:
                trade.exit_px = trade.tp_px
                trade.close_reason = "TP"
                break
        else:
            if h >= trade.sl_px:
                trade.exit_px = trade.sl_px
                trade.close_reason = "SL"
                break
            if l <= trade.tp_px:
                trade.exit_px = trade.tp_px
                trade.close_reason = "TP"
                break
    else:
        i = n - 1
        trade.exit_px = float(bars_ahead.iloc[i]["close"])
        trade.close_reason = "TIME"
    trade.exit_ts = str(bars_ahead.index[i])
    # PnL
    if trade.is_long:
        trade.pnl_pct = (trade.exit_px - trade.entry_px) / trade.entry_px
    else:
        trade.pnl_pct = (trade.entry_px - trade.exit_px) / trade.entry_px
    sl_dist_pct = abs(trade.entry_px - trade.sl_px) / trade.entry_px
    trade.pnl_r = trade.pnl_pct / sl_dist_pct if sl_dist_pct > 0 else 0.0
    return trade


# ────────────────────────────────────────────────────────────────────────
# Main backtest entry
# ────────────────────────────────────────────────────────────────────────
def run_backtest(coin: str, bars: pd.DataFrame,
                  detector: Callable[[pd.DataFrame], Optional[dict]],
                  trade_params: dict,
                  warmup_bars: int = 200) -> dict:
    """
    Walk `bars` chronologically. At each bar, slice [0..i+1] and call detector.
    If it returns a signal, simulate the trade across the following bars.
    No look-ahead.
    """
    trades: List[Trade] = []
    skipped = 0
    if len(bars) < warmup_bars + 5:
        return {"coin": coin, "n_trades": 0, "n_bars": len(bars),
                "note": "insufficient bars"}

    # NB: detector reads df.attrs.get("coin") for coin-aware strategies.
    bars.attrs["coin"] = coin
    # Open positions guard — one trade per coin at a time
    open_until = -1

    for i in range(warmup_bars, len(bars) - 1):
        if i <= open_until:
            continue
        slice_df = bars.iloc[:i + 1]
        slice_df.attrs["coin"] = coin
        try:
            sig = detector(slice_df)
        except Exception:
            sig = None
        if sig is None:
            continue
        # Hot fields
        required = {"ref_price", "sl_px", "tp_px", "is_long", "max_hold_bars"}
        if not required.issubset(sig.keys()):
            skipped += 1
            continue
        t = Trade(
            coin=coin,
            fire_ts=str(bars.index[i]),
            entry_px=float(sig["ref_price"]),
            sl_px=float(sig["sl_px"]),
            tp_px=float(sig["tp_px"]),
            is_long=bool(sig["is_long"]),
            max_hold_bars=int(sig["max_hold_bars"]),
            fire_reason=str(sig.get("fire_reason", "")),
        )
        # Sanity on SL/TP placement
        sl_pct = abs(t.entry_px - t.sl_px) / t.entry_px
        if sl_pct < 0.001 or sl_pct > 0.10:
            skipped += 1
            continue
        bars_ahead = bars.iloc[i + 1:]
        t = _resolve(t, bars_ahead)
        trades.append(t)
        # Hold this trade's window — block re-fires on overlapping bars
        bars_to_resolution = t.max_hold_bars
        if t.close_reason in ("SL", "TP"):
            # Find actual bars used
            try:
                end_idx = bars.index.get_loc(pd.Timestamp(t.exit_ts))
                bars_to_resolution = end_idx - i
            except Exception:
                pass
        open_until = i + max(1, bars_to_resolution)

    # Aggregate stats
    n = len(trades)
    if n == 0:
        return {"coin": coin, "n_trades": 0, "n_bars": len(bars),
                "n_skipped": skipped}
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    gross_win = sum(t.pnl_r for t in wins)
    gross_loss = abs(sum(t.pnl_r for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    rs = [t.pnl_r for t in trades]
    closes = {"SL": 0, "TP": 0, "TIME": 0, "no_data": 0}
    for t in trades:
        closes[t.close_reason] = closes.get(t.close_reason, 0) + 1

    return {
        "coin": coin,
        "n_trades": n,
        "n_bars": len(bars),
        "n_skipped": skipped,
        "wr_pct": round(100 * len(wins) / n, 1),
        "avg_r": round(statistics.mean(rs), 3),
        "median_r": round(statistics.median(rs), 3),
        "sum_r": round(sum(rs), 2),
        "pf": round(pf, 2) if pf != float("inf") else "inf",
        "max_win_r": round(max((t.pnl_r for t in trades), default=0), 2),
        "max_loss_r": round(min((t.pnl_r for t in trades), default=0), 2),
        "close_reasons": closes,
        "first_ts": str(bars.index[0]),
        "last_ts":  str(bars.index[-1]),
    }


# ────────────────────────────────────────────────────────────────────────
# Markdown report
# ────────────────────────────────────────────────────────────────────────
def sweep_results(results: list[dict], out_path: str = "BACKTEST_RESULTS.md",
                   engine_name: str = "engine"):
    """Aggregate and write a markdown summary."""
    valid = [r for r in results if r.get("n_trades", 0) > 0]
    total_trades = sum(r["n_trades"] for r in valid)
    if total_trades == 0:
        body = f"# {engine_name} — Backtest\n\nNo trades fired across {len(results)} coins.\n"
        open(out_path, "w").write(body)
        return body

    agg_wr = round(
        100 * sum(r["n_trades"] * r["wr_pct"] / 100 for r in valid) / total_trades, 1)
    agg_sum_r = round(sum(r["sum_r"] for r in valid), 2)
    # PF aggregated by gross win/loss reconstruction is approximate from per-coin pf;
    # use median of per-coin pf for stability
    pfs = [r["pf"] for r in valid if r["pf"] != "inf"]
    median_pf = round(statistics.median(pfs), 2) if pfs else "n/a"

    lines = [
        f"# {engine_name} — Backtest",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_",
        "",
        "## Aggregate",
        "",
        f"- **Total trades:** {total_trades}",
        f"- **Aggregate WR:** {agg_wr}%",
        f"- **Sum R:** {agg_sum_r}",
        f"- **Median per-coin PF:** {median_pf}",
        f"- **Universe coverage:** {len(valid)} / {len(results)} coins fired",
        "",
        "## Per-coin",
        "",
        "| coin | n | WR% | avg R | sum R | PF | max W | max L | SL/TP/TIME |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in sorted(results, key=lambda r: -r.get("n_trades", 0)):
        if r.get("n_trades", 0) == 0:
            continue
        cr = r.get("close_reasons", {})
        lines.append(
            f"| `{r['coin']}` | {r['n_trades']} | {r['wr_pct']} | "
            f"{r['avg_r']} | {r['sum_r']} | {r['pf']} | "
            f"{r['max_win_r']} | {r['max_loss_r']} | "
            f"{cr.get('SL',0)}/{cr.get('TP',0)}/{cr.get('TIME',0)} |"
        )
    # Coins that didn't fire
    quiet = [r['coin'] for r in results if r.get("n_trades", 0) == 0]
    if quiet:
        lines += ["", "## No-fire coins", "", ", ".join(f"`{c}`" for c in quiet)]
    body = "\n".join(lines) + "\n"
    open(out_path, "w").write(body)
    return body
