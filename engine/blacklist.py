"""
Coin blacklist + consecutive-loss tracker.

Persistent (SQLite) per-coin counter that increments on every losing trade
and resets on every winning trade. When a coin's counter reaches the
threshold (default 5), it joins the blacklist and the scan loop skips it.

The threshold is configurable via env var BLACKLIST_LOSS_THRESHOLD.

This module shares the same SQLite database as persistence.py — adds one
new table `coin_blacklist`. Threadsafe via WAL mode (already set by
persistence.init_db).
"""
from __future__ import annotations
import os
import sqlite3
import time
import threading
from typing import Optional

from .config import STATE_DIR, DB_FILE

_lock = threading.Lock()
THRESHOLD = int(os.environ.get("BLACKLIST_LOSS_THRESHOLD", "5"))


def _db_path() -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, DB_FILE)


def _conn() -> sqlite3.Connection:
    """One short-lived connection per call. WAL mode handles concurrency."""
    c = sqlite3.connect(_db_path(), check_same_thread=False, isolation_level=None, timeout=10.0)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_table():
    """Idempotent. Adds coin_blacklist table to the existing engine DB."""
    with _lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS coin_blacklist (
                coin              TEXT PRIMARY KEY,
                consec_losses     INTEGER NOT NULL DEFAULT 0,
                blacklisted       INTEGER NOT NULL DEFAULT 0,
                blacklist_ts      INTEGER,
                last_outcome_ts   INTEGER,
                last_net_pnl      REAL
            )
        """)


def record_outcome(coin: str, net_pnl: float, outcome: Optional[str] = None):
    """
    Update the consecutive-loss counter for `coin` based on net_pnl.
      - net_pnl >  0  → reset counter to 0
      - net_pnl <= 0  → increment counter by 1; blacklist if ≥ THRESHOLD
    `outcome` is informational ("TP"/"SL"/"TIME"/"MANUAL"/"HALT").
    """
    if not coin:
        return
    now = int(time.time() * 1000)
    is_loss = net_pnl <= 0  # break-even counts as loss (fees ate it)
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT consec_losses, blacklisted FROM coin_blacklist WHERE coin = ?",
            (coin,),
        ).fetchone()
        if row is None:
            consec = 1 if is_loss else 0
            blk = 1 if (is_loss and consec >= THRESHOLD) else 0
            blk_ts = now if blk else None
            c.execute("""
                INSERT INTO coin_blacklist
                  (coin, consec_losses, blacklisted, blacklist_ts, last_outcome_ts, last_net_pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (coin, consec, blk, blk_ts, now, net_pnl))
        else:
            consec, already_blk = row
            if is_loss:
                consec += 1
            else:
                consec = 0
            new_blk = 1 if consec >= THRESHOLD else int(already_blk)  # blacklist is sticky
            blk_ts_set = now if (new_blk and not already_blk) else None
            if blk_ts_set is not None:
                c.execute("""
                    UPDATE coin_blacklist
                       SET consec_losses=?, blacklisted=?, blacklist_ts=?,
                           last_outcome_ts=?, last_net_pnl=?
                     WHERE coin = ?
                """, (consec, new_blk, blk_ts_set, now, net_pnl, coin))
            else:
                c.execute("""
                    UPDATE coin_blacklist
                       SET consec_losses=?, blacklisted=?,
                           last_outcome_ts=?, last_net_pnl=?
                     WHERE coin = ?
                """, (consec, new_blk, now, net_pnl, coin))


def is_blacklisted(coin: str) -> bool:
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT blacklisted FROM coin_blacklist WHERE coin = ?", (coin,)
        ).fetchone()
        return bool(row and row[0])


def get_blacklisted() -> list[str]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT coin FROM coin_blacklist WHERE blacklisted = 1 ORDER BY blacklist_ts ASC"
        ).fetchall()
        return [r[0] for r in rows]


def get_consec_losses() -> dict[str, int]:
    """Returns {coin: consec_losses} for coins with non-zero counters."""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT coin, consec_losses FROM coin_blacklist WHERE consec_losses > 0"
        ).fetchall()
        return {r[0]: r[1] for r in rows}


def get_state() -> dict:
    """Full state snapshot for /state endpoint."""
    with _lock, _conn() as c:
        rows = c.execute("""
            SELECT coin, consec_losses, blacklisted, blacklist_ts, last_outcome_ts, last_net_pnl
              FROM coin_blacklist ORDER BY blacklisted DESC, consec_losses DESC, coin ASC
        """).fetchall()
    blacklisted = [r[0] for r in rows if r[2]]
    counters = {r[0]: r[1] for r in rows if r[1] > 0}
    return {
        "threshold": THRESHOLD,
        "blacklisted": blacklisted,
        "blacklisted_count": len(blacklisted),
        "consec_losses": counters,
        "tracked_coins": len(rows),
    }


def reset_coin(coin: str):
    """Manual override — clears blacklist + counter for a coin."""
    with _lock, _conn() as c:
        c.execute("""
            UPDATE coin_blacklist
               SET consec_losses=0, blacklisted=0, blacklist_ts=NULL
             WHERE coin = ?
        """, (coin,))


def reset_all():
    """Full reset of the blacklist (use with caution)."""
    with _lock, _conn() as c:
        c.execute("UPDATE coin_blacklist SET consec_losses=0, blacklisted=0, blacklist_ts=NULL")
