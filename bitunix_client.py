"""
Bitunix Futures API Client - Robust Version

Authenticated REST client for Bitunix Futures with:
- Account balance fetching (USDT)
- Market order placement (Long/Short)
- Ticker price polling
- Kline (candlestick) data fetching
- Position querying and closing

Security & Robustness:
- Timeouts on all requests
- Rate limit (429) handling with backoff
- No logging of sensitive credentials
- Secure signature generation
"""

import hashlib
import json
import logging
import os
import time
import uuid
import re
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.bitunix.com"
DEFAULT_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _sort_query_params(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    # Sort by key and concat key+value
    return "".join(f"{k}{v}" for k, v in sorted(params.items()))

def _generate_signature(
    api_key: str,
    secret_key: str,
    nonce: str,
    timestamp: str,
    query_params: str = "",
    body: str = "",
) -> str:
    digest_input = nonce + timestamp + api_key + query_params + body
    digest = _sha256_hex(digest_input)
    sign = _sha256_hex(digest + secret_key)
    return sign

def _auth_headers(
    api_key: str,
    secret_key: str,
    query_params: str = "",
    body: str = "",
) -> Dict[str, str]:
    nonce = uuid.uuid4().hex
    timestamp = str(int(time.time() * 1000))
    sign = _generate_signature(api_key, secret_key, nonce, timestamp, query_params, body)
    return {
        "api-key": api_key,
        "sign": sign,
        "nonce": nonce,
        "timestamp": timestamp,
        "Content-Type": "application/json",
        "language": "en-US",
    }

# ---------------------------------------------------------------------------
# BitunixFuturesClient
# ---------------------------------------------------------------------------

class BitunixFuturesClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = BASE_URL,
    ):
        self.api_key = api_key or os.environ.get("BITUNIX_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BITUNIX_API_SECRET", "")
        self.base_url = base_url.rstrip("/")
        
        # Setup session with retry logic for network errors
        self._session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

        if not self.api_key or not self.api_secret:
            logger.warning("API credentials not fully set. Private endpoints will fail.")

    def _handle_response(self, resp: requests.Response) -> Any:
        # Handle Rate Limiting (429)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 5))
            logger.warning("Rate limit hit. Backing off for %d seconds...", retry_after)
            time.sleep(retry_after)
            # We don't retry here to avoid complex recursion, caller will retry next cycle
            raise BitunixAPIError("Rate limit exceeded (429)")

        if resp.status_code != 200:
            # Avoid logging raw body if it might contain sensitive info
            raise BitunixAPIError(f"HTTP {resp.status_code}: Request failed")

        try:
            payload = resp.json()
        except ValueError:
            raise BitunixAPIError("Invalid JSON response from API")

        code = payload.get("code")
        if code != 0:
            msg = payload.get("msg", "unknown error")
            raise BitunixAPIError(f"API error {code}: {msg}")
            
        return payload.get("data")

    def _request(self, method: str, path: str, params=None, data=None, auth=False) -> Any:
        url = f"{self.base_url}{path}"
        headers = {}
        body_str = ""
        
        if data:
            body_str = json.dumps(data, separators=(",", ":"))
            
        if auth:
            if not self.api_key or not self.api_secret:
                raise BitunixAPIError("Authentication required but credentials missing")
            
            query_str = _sort_query_params(params or {})
            headers = _auth_headers(self.api_key, self.api_secret, query_params=query_str, body=body_str)

        try:
            resp = self._session.request(
                method=method,
                url=url,
                params=params,
                data=body_str if method == "POST" else None,
                headers=headers,
                timeout=DEFAULT_TIMEOUT
            )
            return self._handle_response(resp)
        except requests.exceptions.RequestException as e:
            raise BitunixAPIError(f"Network error: {type(e).__name__}")

    def _get(self, path: str, params=None, auth=False) -> Any:
        return self._request("GET", path, params=params, auth=auth)

    def _post(self, path: str, data: Dict[str, Any]) -> Any:
        return self._request("POST", path, data=data, auth=True)

    # -- Validation --
    def _validate_symbol(self, symbol: str):
        if not re.match(r"^[A-Z0-9]{2,12}$", symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")

    # -- Public endpoints --
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        self._validate_symbol(symbol)
        data = self._get("/api/v1/futures/market/tickers", params={"symbols": symbol})
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data or {}

    def get_tickers(self, symbols: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if symbols:
            params["symbols"] = symbols
        return self._get("/api/v1/futures/market/tickers", params=params) or []

    def get_klines(self, symbol: str, interval: str = "15", limit: int = 100) -> List[Dict[str, Any]]:
        self._validate_symbol(symbol)
        params = {"symbol": symbol, "interval": interval, "limit": str(limit)}
        return self._get("/api/v1/futures/market/klines", params=params) or []

    # -- Private endpoints --
    def get_account(self, margin_coin: str = "USDT") -> Any:
        return self._get("/api/v1/futures/account", params={"marginCoin": margin_coin}, auth=True)

    def get_balance(self, margin_coin: str = "USDT") -> str:
        data = self.get_account(margin_coin)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("available", "0")
        if isinstance(data, dict):
            return data.get("available", "0")
        return "0"

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if symbol:
            self._validate_symbol(symbol)
            params["symbol"] = symbol
        data = self._get("/api/v1/futures/position", params=params, auth=True)
        if data is None: return []
        if isinstance(data, list): return data
        return [data] if isinstance(data, dict) else []

    def place_order(self, symbol: str, side: str, qty: str, order_type: str = "MARKET", **kwargs) -> Dict[str, Any]:
        self._validate_symbol(symbol)
        body = {
            "symbol": symbol,
            "side": side.upper(),
            "orderType": order_type.upper(),
            "qty": qty,
            "tradeSide": kwargs.get("trade_side", "OPEN").upper(),
            "reduceOnly": kwargs.get("reduce_only", False),
        }
        # Add optional params
        for key in ["price", "effect", "clientId", "tpPrice", "slPrice"]:
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        
        # Special handling for TP/SL stop types if provided
        if "tpPrice" in body:
            body["tpStopType"] = "LAST_PRICE"
            body["tpOrderType"] = "MARKET"
        if "slPrice" in body:
            body["slStopType"] = "LAST_PRICE"
            body["slOrderType"] = "MARKET"

        return self._post("/api/v1/futures/trade/place_order", body)

    def place_market_buy(self, symbol: str, qty: str, **kwargs) -> Dict[str, Any]:
        return self.place_order(symbol, "BUY", qty, order_type="MARKET", trade_side="OPEN", **kwargs)

    def place_market_sell(self, symbol: str, qty: str, **kwargs) -> Dict[str, Any]:
        return self.place_order(symbol, "SELL", qty, order_type="MARKET", trade_side="OPEN", **kwargs)

    def close_position(self, symbol: str, side: str, qty: str) -> Dict[str, Any]:
        return self.place_order(symbol, side, qty, order_type="MARKET", trade_side="CLOSE", reduce_only=True)

class BitunixAPIError(Exception):
    pass
