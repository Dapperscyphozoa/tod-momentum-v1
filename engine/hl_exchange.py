"""
Hyperliquid live-trading wrapper.

Wraps the hyperliquid-python-sdk Exchange + Info clients with engine-specific
guardrails: cloid generation, size rounding, pre-flight checks, and safe
fallback when keys are missing.

CLOID STRATEGY:
HL on-chain cloids are 0x + 32 hex (16 bytes). Our internal cloids are
human-readable `vsqf_<24hex>` for debugging. We deterministically hash internal
→ on-chain via SHA-256. Both are stored in DB; reverse lookup from HL fills
back to our trades is done by hashing every open internal cloid and matching.
"""
from __future__ import annotations
import hashlib
import logging
import threading
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Lazy imports — only required when LIVE_TRADING=1
_SDK_LOADED = False
Exchange = None
Info = None
Cloid = None
Account = None
constants = None


def _load_sdk():
    """Import HL SDK only when needed. Raises ImportError if SDK missing."""
    global _SDK_LOADED, Exchange, Info, Cloid, Account, constants
    if _SDK_LOADED:
        return
    from hyperliquid.exchange import Exchange as _Exchange
    from hyperliquid.info import Info as _Info
    from hyperliquid.utils.types import Cloid as _Cloid
    from hyperliquid.utils import constants as _constants
    from eth_account import Account as _Account
    Exchange = _Exchange
    Info = _Info
    Cloid = _Cloid
    Account = _Account
    constants = _constants
    _SDK_LOADED = True


def to_exchange_cloid(internal_cloid: str) -> str:
    """
    Convert internal cloid (e.g. 'vsqf_a1b2...') to HL-acceptable on-chain
    cloid (0x + 32 hex chars = 16 bytes).

    Deterministic: same internal → same on-chain. Reverse lookup is done by
    re-hashing all candidate internal cloids and matching the result.
    """
    h = hashlib.sha256(internal_cloid.encode()).hexdigest()
    return "0x" + h[:32]


def reverse_cloid_lookup(exchange_cloid: str, candidate_internals: list[str]) -> Optional[str]:
    """Find which internal cloid hashed to a given on-chain cloid."""
    target = exchange_cloid.lower()
    for internal in candidate_internals:
        if to_exchange_cloid(internal).lower() == target:
            return internal
    return None


class HLClient:
    """
    Thin wrapper around hyperliquid-python-sdk Exchange + Info.

    Constructed once at engine startup. If `private_key` is empty/None, the
    client is in 'unarmed' state — calls return errors but don't crash.
    """

    def __init__(self, private_key: str, expected_wallet: str, testnet: bool = False):
        _load_sdk()
        self._lock = threading.Lock()
        self.expected_wallet = expected_wallet.lower()
        self._meta_cache: Optional[Dict[str, Any]] = None
        self._sz_decimals: Dict[str, int] = {}
        self._max_leverage: Dict[str, int] = {}

        if not private_key:
            self.armed = False
            self.exchange = None
            self.info = None
            self.actual_wallet = None
            logger.warning("HLClient unarmed — no private_key provided")
            return

        try:
            account = Account.from_key(private_key)
            actual = account.address.lower()
        except Exception as e:
            self.armed = False
            self.exchange = None
            self.info = None
            self.actual_wallet = None
            logger.error(f"HLClient init failed deriving address from key: {e}")
            return

        if actual != self.expected_wallet:
            self.armed = False
            self.exchange = None
            self.info = None
            self.actual_wallet = actual
            logger.error(
                f"HLClient WALLET MISMATCH: expected={self.expected_wallet} "
                f"derived={actual}. Refusing to arm."
            )
            return

        api_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.exchange = Exchange(account, api_url, account_address=actual)
        self.info = Info(api_url, skip_ws=True)
        self.actual_wallet = actual
        self.armed = True
        logger.info(f"HLClient armed: wallet={actual} api={api_url}")

        # Pre-fetch meta to populate sz_decimals
        try:
            self._refresh_meta()
        except Exception as e:
            logger.warning(f"HLClient meta pre-fetch failed: {e}")

    def _refresh_meta(self):
        with self._lock:
            self._meta_cache = self.info.meta()
            for asset in self._meta_cache.get("universe", []):
                name = asset.get("name")
                if name:
                    self._sz_decimals[name] = int(asset.get("szDecimals", 4))
                    self._max_leverage[name] = int(asset.get("maxLeverage", 1))

    def get_sz_decimals(self, coin: str) -> int:
        if coin not in self._sz_decimals:
            try:
                self._refresh_meta()
            except Exception:
                pass
        return self._sz_decimals.get(coin, 4)

    def get_max_leverage(self, coin: str) -> int:
        return self._max_leverage.get(coin, 1)

    def round_size(self, coin: str, size: float) -> float:
        """Round size DOWN to coin's sz_decimals (HL rejects too-precise sizes)."""
        decimals = self.get_sz_decimals(coin)
        factor = 10 ** decimals
        # Round down (truncate) so we never exceed risk budget
        return int(size * factor) / factor

    def round_price(self, coin: str, price: float, is_buy: bool) -> float:
        """
        Round price to HL's tick precision. HL allows up to 5 significant figs
        OR sz_decimals + 6 decimal places, whichever is more permissive.
        Round towards LESS aggressive fill (down for buy limit, up for sell limit)
        when used as post-only entry — but we want passive maker fills, so:
          buy limit → round DOWN (stays passive below market)
          sell limit → round UP (stays passive above market)
        """
        # 5 significant figures
        if price <= 0:
            return price
        from math import log10, floor
        sig_digits = 5
        if price < 1:
            decimals = sig_digits - int(floor(log10(price))) - 1
        else:
            decimals = max(0, sig_digits - int(floor(log10(price))) - 1)
        # Constraint: HL also caps decimals at sz_decimals + 6 (for perps it's lenient)
        max_decimals = self.get_sz_decimals(coin) + 6
        decimals = min(decimals, max_decimals)
        factor = 10 ** decimals
        if is_buy:
            return floor(price * factor) / factor
        else:
            from math import ceil
            return ceil(price * factor) / factor

    def get_account_value(self) -> Optional[float]:
        if not self.armed:
            return None
        try:
            state = self.info.user_state(self.actual_wallet)
            return float(state.get("marginSummary", {}).get("accountValue", 0))
        except Exception as e:
            logger.error(f"get_account_value failed: {e}")
            return None

    def get_position(self, coin: str) -> Optional[Dict[str, Any]]:
        """Return current position dict or None if no position."""
        if not self.armed:
            return None
        try:
            state = self.info.user_state(self.actual_wallet)
            for p in state.get("assetPositions", []):
                pos = p.get("position", {})
                if pos.get("coin") == coin:
                    szi = float(pos.get("szi", 0))
                    if abs(szi) > 1e-12:
                        return {
                            "coin": coin,
                            "szi": szi,
                            "entry_px": float(pos.get("entryPx", 0)) if pos.get("entryPx") else None,
                            "is_long": szi > 0,
                            "leverage": pos.get("leverage", {}),
                            "unrealized_pnl": float(pos.get("unrealizedPnl", 0)) if pos.get("unrealizedPnl") else 0,
                        }
            return None
        except Exception as e:
            logger.error(f"get_position failed for {coin}: {e}")
            return None

    def update_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> Dict[str, Any]:
        if not self.armed:
            return {"status": "error", "error": "client_unarmed"}
        try:
            return self.exchange.update_leverage(leverage, coin, is_cross)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def place_post_only_limit(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        limit_px: float,
        internal_cloid: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place an Add-Liquidity-Only (post-only) limit order.
        If price would cross spread, HL rejects it (prevents accidental taker fills).

        Returns:
          {"status": "ok", "oid": int, "exchange_cloid": str, "filled": bool, "details": {...}}
          {"status": "error", "error": str, "details": {...}}
        """
        if not self.armed:
            return {"status": "error", "error": "client_unarmed"}

        size = self.round_size(coin, size)
        limit_px = self.round_price(coin, limit_px, is_buy)

        if size <= 0:
            return {"status": "error", "error": f"size_rounded_to_zero (sz_decimals={self.get_sz_decimals(coin)})"}

        try:
            exch_cloid_str = to_exchange_cloid(internal_cloid)
            cloid_obj = Cloid(exch_cloid_str)

            order_type = {"limit": {"tif": "Alo"}}  # Add-Liquidity-Only = post-only
            result = self.exchange.order(
                coin, is_buy, size, limit_px, order_type,
                reduce_only=reduce_only,
                cloid=cloid_obj,
            )
            return self._parse_order_response(result, exch_cloid_str)
        except Exception as e:
            logger.error(f"place_post_only_limit {coin} {is_buy} {size}@{limit_px}: {e}")
            return {"status": "error", "error": str(e)}

    def place_market_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        internal_cloid: str,
        slippage: float = 0.05,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place a market order via market_open (uses IOC at slippage-adjusted price).
        Used for SL exits where speed matters more than maker rebate.
        """
        if not self.armed:
            return {"status": "error", "error": "client_unarmed"}

        size = self.round_size(coin, size)
        if size <= 0:
            return {"status": "error", "error": "size_rounded_to_zero"}

        try:
            exch_cloid_str = to_exchange_cloid(internal_cloid)
            cloid_obj = Cloid(exch_cloid_str)

            if reduce_only:
                # market_close handles reduce-only flag automatically
                result = self.exchange.market_close(
                    coin, sz=size, slippage=slippage, cloid=cloid_obj,
                )
            else:
                result = self.exchange.market_open(
                    coin, is_buy, size, px=None, slippage=slippage, cloid=cloid_obj,
                )
            return self._parse_order_response(result, exch_cloid_str)
        except Exception as e:
            logger.error(f"place_market_order {coin} {is_buy} {size}: {e}")
            return {"status": "error", "error": str(e)}

    def market_close_position(self, coin: str, internal_cloid: str, slippage: float = 0.05) -> Dict[str, Any]:
        """Close ENTIRE existing position via market order. SL exit path."""
        if not self.armed:
            return {"status": "error", "error": "client_unarmed"}

        try:
            exch_cloid_str = to_exchange_cloid(internal_cloid)
            cloid_obj = Cloid(exch_cloid_str)
            # market_close with sz=None closes whole position
            result = self.exchange.market_close(
                coin, sz=None, slippage=slippage, cloid=cloid_obj,
            )
            return self._parse_order_response(result, exch_cloid_str)
        except Exception as e:
            logger.error(f"market_close_position {coin}: {e}")
            return {"status": "error", "error": str(e)}

    def cancel_order(self, coin: str, oid: int) -> Dict[str, Any]:
        if not self.armed:
            return {"status": "error", "error": "client_unarmed"}
        try:
            return {"status": "ok", "details": self.exchange.cancel(coin, oid)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _parse_order_response(raw: dict, exchange_cloid: str) -> Dict[str, Any]:
        """
        Parse HL's order-response shape into a flat dict.
        Raw shape (success):
          {"status": "ok", "response": {"type": "order",
              "data": {"statuses": [{"resting": {"oid": 123}}]}}}
          or {"filled": {"totalSz": "1.0", "avgPx": "100.0", "oid": 123}}
        Raw shape (error):
          {"status": "err", "response": "<message>"}
          or status="ok" but statuses contain "error"
        """
        if not isinstance(raw, dict):
            return {"status": "error", "error": "non_dict_response", "details": raw}

        top_status = raw.get("status")
        if top_status == "err":
            return {"status": "error", "error": str(raw.get("response", "unknown")), "details": raw}

        try:
            statuses = raw.get("response", {}).get("data", {}).get("statuses", [])
        except Exception:
            return {"status": "error", "error": "malformed_response", "details": raw}

        if not statuses:
            return {"status": "error", "error": "empty_statuses", "details": raw}

        st = statuses[0]
        if isinstance(st, dict):
            if "error" in st:
                return {"status": "error", "error": st["error"], "details": raw, "exchange_cloid": exchange_cloid}
            if "resting" in st:
                oid = st["resting"].get("oid")
                return {"status": "ok", "oid": int(oid) if oid is not None else None,
                        "exchange_cloid": exchange_cloid, "filled": False, "details": raw}
            if "filled" in st:
                f = st["filled"]
                return {"status": "ok", "oid": int(f.get("oid", 0)) if f.get("oid") else None,
                        "exchange_cloid": exchange_cloid, "filled": True,
                        "filled_sz": float(f.get("totalSz", 0)),
                        "avg_px": float(f.get("avgPx", 0)),
                        "details": raw}
        return {"status": "error", "error": "unparseable_status", "details": raw, "exchange_cloid": exchange_cloid}


# ===== Pre-live safety checks =====
class PreLiveCheckResult:
    def __init__(self, passed: bool, failures: list[str], warnings: list[str], details: dict):
        self.passed = passed
        self.failures = failures
        self.warnings = warnings
        self.details = details

    def to_dict(self):
        return {"passed": self.passed, "failures": self.failures,
                "warnings": self.warnings, "details": self.details}


def pre_live_checks(
    client: HLClient,
    expected_wallet: str,
    target_coin: str,
    min_account_value: float = 200.0,
    require_no_existing_position: bool = True,
) -> PreLiveCheckResult:
    """
    Verify safety conditions BEFORE placing a live trade.

    Checks:
      1. Client armed (private key valid + matches expected wallet)
      2. Account value >= min_account_value
      3. No existing position on target_coin (or warning if same-side)
      4. Coin exists in HL universe with non-zero max leverage
      5. Account value retrievable (HL connectivity sane)
    """
    failures = []
    warnings = []
    details = {"target_coin": target_coin, "expected_wallet": expected_wallet}

    if not client.armed:
        failures.append("client_unarmed")
        return PreLiveCheckResult(False, failures, warnings, details)

    if client.actual_wallet != expected_wallet.lower():
        failures.append(f"wallet_mismatch: expected={expected_wallet} actual={client.actual_wallet}")
        return PreLiveCheckResult(False, failures, warnings, details)

    acct_val = client.get_account_value()
    details["account_value"] = acct_val
    if acct_val is None:
        failures.append("hl_unreachable_cant_fetch_account_value")
        return PreLiveCheckResult(False, failures, warnings, details)
    if acct_val < min_account_value:
        failures.append(f"account_value_too_low: ${acct_val:.2f} < ${min_account_value:.2f}")

    if target_coin not in client._sz_decimals:
        try:
            client._refresh_meta()
        except Exception as e:
            warnings.append(f"meta_refresh_failed: {e}")
    if target_coin not in client._sz_decimals:
        failures.append(f"coin_not_in_hl_universe: {target_coin}")
        return PreLiveCheckResult(False, failures, warnings, details)

    details["sz_decimals"] = client.get_sz_decimals(target_coin)
    details["max_leverage"] = client.get_max_leverage(target_coin)
    if details["max_leverage"] < 1:
        failures.append(f"coin_has_zero_leverage_available: {target_coin}")

    if require_no_existing_position:
        pos = client.get_position(target_coin)
        if pos is not None:
            failures.append(f"existing_position_on_{target_coin}: szi={pos['szi']}")
            details["existing_position"] = pos

    return PreLiveCheckResult(len(failures) == 0, failures, warnings, details)
