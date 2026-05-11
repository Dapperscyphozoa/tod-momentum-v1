"""
tod-reversion-v1 — Time-of-day mean reversion engine.
Fires MR fades during statistically-profitable hours per asset.
v1 uses a SIMPLE whitelist (overridable via env) — hours during which
mean reversion has shown edge across crypto in general:
  Asian wee hours (UTC 02:00-05:00): thin liquidity sweeps
  London close to NY mid-session (UTC 15:00-18:00): MR setups
Calibration is left to a subsequent commit; v1 ships the framework.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from .config import STRATEGY_PARAMS, TRADE_PARAMS


def calc_vwap(highs, lows, closes, vols):
    typical = (highs + lows + closes) / 3.0
    cum_pv = (typical * vols).cumsum()
    cum_v = vols.cumsum()
    return cum_pv / np.where(cum_v == 0, 1, cum_v)


def calc_atr(highs, lows, closes, period: int = 14) -> float:
    h_s = pd.Series(highs); l_s = pd.Series(lows); pc = pd.Series(closes).shift(1)
    tr = pd.concat([h_s - l_s, (h_s - pc).abs(), (l_s - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def evaluate_latest_bar(df: pd.DataFrame) -> Optional[dict]:
    # Whitelist hours-of-day UTC (CSV via STRATEGY_PARAMS for override)
    whitelist_str = STRATEGY_PARAMS.get("hour_whitelist",
                                         "2,3,4,15,16,17,18")
    whitelist = set(int(x) for x in whitelist_str.split(","))
    DEV_PCT = STRATEGY_PARAMS.get("vwap_dev_threshold_pct", 0.004)
    if df is None or len(df) < 24: return None

    # Get hour in UTC from index
    last_ts = df.index[-1]
    try: hour_utc = int(last_ts.hour)
    except: return None
    if hour_utc not in whitelist: return None

    # Session VWAP (since hour start). Resample to 1m if df is finer;
    # for 1h frame, use rolling 24-bar VWAP as proxy.
    highs = df["high"].values; lows = df["low"].values
    closes = df["close"].values
    vols = df["volume"].values if "volume" in df.columns else np.ones(len(df))

    # Anchored VWAP over the last 24 bars (rough daily anchor for hourly)
    anchor = max(0, len(df) - 24)
    h24 = highs[anchor:]; l24 = lows[anchor:]; c24 = closes[anchor:]; v24 = vols[anchor:]
    vwap_arr = calc_vwap(h24, l24, c24, v24)
    cur_vwap = float(vwap_arr[-1])
    last_c = float(closes[-1])
    dev = (last_c - cur_vwap) / cur_vwap

    if abs(dev) < DEV_PCT: return None

    is_long = dev < 0   # price below vwap → mean revert UP

    atr = calc_atr(highs, lows, closes, TRADE_PARAMS["atr_period"])
    if not atr or atr <= 0: return None
    sl_m = TRADE_PARAMS["sl_atr_mult"]; tp_m = TRADE_PARAMS["tp_atr_mult"]
    if is_long: sl_p = last_c - sl_m * atr; tp_p = cur_vwap  # revert to VWAP
    else:       sl_p = last_c + sl_m * atr; tp_p = cur_vwap

    # Sanity: tp must be different from entry by >= 0.5×atr
    if abs(tp_p - last_c) < 0.5 * atr:
        if is_long: tp_p = last_c + tp_m * atr
        else:       tp_p = last_c - tp_m * atr

    return {
        "fire_ts": df.index[-1], "ref_price": last_c, "atr": atr,
        "trade_side": "B" if is_long else "A", "is_long": is_long,
        "sl_px": float(sl_p), "tp_px": float(tp_p),
        "max_hold_bars": TRADE_PARAMS.get("max_hold_bars", 8),
        "fire_reason": f"tod_h{hour_utc}_dev{dev*100:+.2f}pct",
        "raw_direction": "LONG" if is_long else "SHORT",
        "fade_direction": "LONG" if is_long else "SHORT",
        "hour_utc": int(hour_utc),
        "vwap": float(cur_vwap), "deviation_pct": float(dev),
    }
