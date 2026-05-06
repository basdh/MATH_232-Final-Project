#!/usr/bin/env python3
"""
Active Polymarket markets grouped by Gamma tag classifiers (Sports, Crypto, Politics, …).

Defaults load from **polymarket_active_classified_config.json** (cwd, then next to this script).
In an interactive terminal, the script **asks** for each setting; press **Enter** to keep the
shown default. Use ``--no-prompt`` to skip questions (config only). ``--config PATH`` still selects
the JSON file.

CLOB has no native sub-hour bucket for arbitrary steps; we merge ``1m``/``max``/``1h`` then
pandas-resample to your ``interval_hours``. **Order book** fields come from a **current** ``/book``
snapshot (repeated on each time row); historical L2 is not in the public API. **Per-bucket** trade
density uses counts of CLOB price prints in each grid interval. **Metadata** merges Gamma
volume/liquidity, that snapshot, and per-market **PCA feature** scalars; **features CSV** holds the
same numeric features **Z-scored** across markets (when ≥2 rows).

CLOB requires ``market=<yes_outcome_token_id>``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES_HISTORY = "https://clob.polymarket.com/prices-history"
CLOB_BATCH_PRICES_HISTORY = "https://clob.polymarket.com/batch-prices-history"
CLOB_BOOK = "https://clob.polymarket.com/book"

# Grid step-to-step |pct_change| thresholds for jump counts/rates.
JUMP_ABS_RETURN_GE_10PCT = 0.10
JUMP_ABS_RETURN_GE_40PCT = 0.40
ORDERBOOK_TOP_N = 5

CLOB_MAX_REQUESTS = 900
CLOB_WINDOW_SEC = 10.0
CLOB_CHUNK_HOURS = 24.0
BATCH_TOKEN_CHUNK_SIZE = 20
BATCH_MAX_CONCURRENCY = 4
DEFAULT_MIN_VOLUME_NUM = 10_000.0
DEFAULT_MIN_LIQUIDITY_NUM = 1_000.0
DEFAULT_USE_ORDERBOOK = False
DEFAULT_CACHE_DIR = ".cache/polymarket_active_classified"

# Built-in umbrella tags (Gamma tag_id, slug, human label).
DEFAULT_CLASSIFIERS: Tuple[Tuple[str, str, str], ...] = (
    ("Sports", "1", "sports"),
    ("Politics", "2", "politics"),
    ("Crypto", "21", "crypto"),
    ("Pop culture", "596", "pop-culture"),
    ("Tech", "1401", "tech"),
    ("Finance", "120", "finance"),
    ("Business", "107", "business"),
    ("Geopolitics", "100265", "geopolitics"),
)

CONFIG_FILENAME = "polymarket_active_classified_config.json"

DEFAULT_MARKETS_PER_CATEGORY = 5
DEFAULT_INTERVAL_HOURS = 0.5
DEFAULT_LOOKBACK_DAYS = 1.0
DEFAULT_MAX_SCAN_OFFSET = 2000
DEFAULT_RAW_OUT = "polymarket_active_classified_raw_30m.csv"
DEFAULT_META_OUT = "polymarket_active_classified_metadata.csv"
DEFAULT_FEATURES_OUT = "polymarket_active_classified_features.csv"

PCA_FEATURE_KEYS = (
    "feat_jump_zcount_ge_2",
    "feat_jump_zrate_ge_2",
    "feat_jump_zcount_ge_3",
    "feat_jump_zrate_ge_3",
    "feat_jump_count_ge_p95",
    "feat_jump_rate_ge_p95",
    "feat_jump_count_ge_p99",
    "feat_jump_rate_ge_p99",
    "feat_jump_max_abs",
    "feat_jump_top3_sum_abs",
    "feat_jump_top3_frac_total_move",
    "feat_return_volatility",
    "feat_time_to_close_hours",
    "feat_raw_prints_per_hour",
    "feat_ob_spread",
    "feat_ob_spread_over_mid",
    "feat_ob_imbalance_topn",
    "feat_ob_mid",
    "feat_ob_log_bid_ask_depth_ratio",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("active_classified")
logging.getLogger("httpx").setLevel(logging.WARNING)


def default_config_dict() -> Dict[str, Any]:
    return {
        "markets_per_category": DEFAULT_MARKETS_PER_CATEGORY,
        "interval_hours": DEFAULT_INTERVAL_HOURS,
        "lookback_days": DEFAULT_LOOKBACK_DAYS,
        "max_scan_offset": DEFAULT_MAX_SCAN_OFFSET,
        "raw_csv": DEFAULT_RAW_OUT,
        "metadata_csv": DEFAULT_META_OUT,
        "features_csv": DEFAULT_FEATURES_OUT,
        "min_volume_num": DEFAULT_MIN_VOLUME_NUM,
        "min_liquidity_num": DEFAULT_MIN_LIQUIDITY_NUM,
        "use_orderbook": DEFAULT_USE_ORDERBOOK,
        "cache_dir": DEFAULT_CACHE_DIR,
        "batch_chunk_size": BATCH_TOKEN_CHUNK_SIZE,
        "batch_concurrency": BATCH_MAX_CONCURRENCY,
    }


def resolve_config_path(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        p = os.path.abspath(explicit)
        return p if os.path.isfile(p) else None
    cwd_p = os.path.join(os.getcwd(), CONFIG_FILENAME)
    if os.path.isfile(cwd_p):
        return cwd_p
    script_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    if os.path.isfile(script_p):
        return script_p
    return None


def load_merged_config(config_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Returns (merged config dict, path read from or None if defaults only)."""
    base = default_config_dict()
    path = resolve_config_path(config_path)
    if not path:
        return base, None
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    if not isinstance(user, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    base.update(user)
    if "markets_per_category" not in user and "markets_per_tag" in user:
        base["markets_per_category"] = int(user["markets_per_tag"])
    base["markets_per_category"] = int(base["markets_per_category"])
    base["interval_hours"] = float(base["interval_hours"])
    base["lookback_days"] = float(base["lookback_days"])
    base["max_scan_offset"] = int(base["max_scan_offset"])
    base["raw_csv"] = str(base["raw_csv"])
    base["metadata_csv"] = str(base["metadata_csv"])
    base["features_csv"] = str(base.get("features_csv", DEFAULT_FEATURES_OUT))
    base["min_volume_num"] = float(base.get("min_volume_num", DEFAULT_MIN_VOLUME_NUM))
    base["min_liquidity_num"] = float(base.get("min_liquidity_num", DEFAULT_MIN_LIQUIDITY_NUM))
    base["use_orderbook"] = bool(base.get("use_orderbook", DEFAULT_USE_ORDERBOOK))
    base["cache_dir"] = str(base.get("cache_dir", DEFAULT_CACHE_DIR))
    base["batch_chunk_size"] = max(1, int(base.get("batch_chunk_size", BATCH_TOKEN_CHUNK_SIZE)))
    base["batch_concurrency"] = max(1, int(base.get("batch_concurrency", BATCH_MAX_CONCURRENCY)))
    return base, path


def interval_hours_to_pandas_freq(interval_hours: float) -> str:
    """Whole-minute multiples only (e.g. 0.5 -> 30min, 1 -> 1h, 2 -> 2h)."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    total_minutes = float(interval_hours) * 60.0
    whole = round(total_minutes)
    if abs(total_minutes - float(whole)) > 1e-6:
        raise ValueError(f"interval_hours={interval_hours} must equal a whole number of minutes")
    m = int(whole)
    if m < 1:
        raise ValueError("interval_hours yields grid shorter than 1 minute")
    if m % 60 == 0:
        h = m // 60
        return f"{h}h"
    return f"{m}min"


def _prompt_optional_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def _prompt_optional_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return float(raw)


def _prompt_optional_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def interactive_fill_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ask stdin for each knob; empty line keeps the current (config) value."""
    out = dict(cfg)
    print("\nConfigure run (Enter = keep default in brackets)\n")
    out["markets_per_category"] = _prompt_optional_int(
        "Markets per category (per Gamma tag)", int(out["markets_per_category"])
    )
    out["interval_hours"] = _prompt_optional_float(
        "Grid step in hours (e.g. 0.5 = 30 min; must be whole minutes)", float(out["interval_hours"])
    )
    out["lookback_days"] = _prompt_optional_float(
        "Lookback window in days (CLOB from max(market start, now − this))", float(out["lookback_days"])
    )
    out["max_scan_offset"] = _prompt_optional_int(
        "Max Gamma offset per tag when paging markets", int(out["max_scan_offset"])
    )
    out["raw_csv"] = _prompt_optional_str("Raw CSV output path", str(out["raw_csv"]))
    out["metadata_csv"] = _prompt_optional_str("Metadata CSV output path", str(out["metadata_csv"]))
    out["features_csv"] = _prompt_optional_str(
        "PCA features CSV (Z-scored numerics, one row per market)", str(out["features_csv"])
    )
    out["min_volume_num"] = _prompt_optional_float(
        "Min Gamma volumeNum to include market", float(out["min_volume_num"])
    )
    out["min_liquidity_num"] = _prompt_optional_float(
        "Min Gamma liquidityNum to include market", float(out["min_liquidity_num"])
    )
    ob_default = "y" if bool(out["use_orderbook"]) else "n"
    ob_raw = input(f"Fetch orderbook snapshots? [y/n, default {ob_default}]: ").strip().lower()
    if ob_raw:
        out["use_orderbook"] = ob_raw in {"y", "yes", "1", "true"}
    out["cache_dir"] = _prompt_optional_str("Cache directory", str(out["cache_dir"]))
    print()
    return out


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window_sec: float) -> None:
        self.max_calls = max_calls
        self.window_sec = window_sec
        self._ts: Deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._ts and now - self._ts[0] > self.window_sec:
            self._ts.popleft()
        if len(self._ts) >= self.max_calls:
            sleep_for = self.window_sec - (now - self._ts[0]) + 0.001
            if sleep_for > 0:
                time.sleep(sleep_for)
            return self.acquire()
        self._ts.append(time.monotonic())


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "polymarket-active-classified/1.0",
            "Accept": "application/json",
        }
    )
    return s


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    limiter: Optional[SlidingWindowRateLimiter] = None,
) -> Any:
    backoff = 0.5
    last_exc: Optional[BaseException] = None
    for _ in range(6):
        if limiter is not None:
            limiter.acquire()
        try:
            r = session.get(url, params=params, timeout=90)
            if r.status_code == 429 or (500 <= r.status_code < 600):
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP retries exhausted")


def parse_gamma_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return None
    parsed: Any = json.loads(s)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return parsed


def fetch_active_markets_page(
    session: requests.Session,
    tag_id: str,
    *,
    offset: int,
    limit: int,
    limiter: SlidingWindowRateLimiter,
) -> List[dict[str, Any]]:
    params = {
        "closed": "false",
        "active": "true",
        "tag_id": tag_id,
        "limit": str(limit),
        "offset": str(offset),
        "order": "volumeNum",
        "ascending": "false",
    }
    data = get_json(session, GAMMA_MARKETS, params=params, limiter=limiter)
    if not isinstance(data, list):
        raise TypeError(f"Expected list from Gamma markets, got {type(data)}")
    return data


def yes_token_for_market(m: dict[str, Any]) -> Optional[str]:
    """First outcome token (Gamma lists clobTokenIds parallel to outcomes)."""
    toks = parse_gamma_json_field(m.get("clobTokenIds"))
    if not isinstance(toks, list) or not toks:
        return None
    return str(toks[0])


def market_passes_prefilter(m: dict[str, Any], *, min_volume_num: float, min_liquidity_num: float) -> bool:
    def _f(key: str) -> float:
        v = m.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    volume = _f("volumeNum")
    liquidity = _f("liquidityNum")
    return volume >= float(min_volume_num) and liquidity >= float(min_liquidity_num)


def _cache_file_for_batch(
    cache_dir: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    token_ids: List[str],
) -> Path:
    key = "|".join([interval, str(start_ts), str(end_ts), ",".join(sorted(token_ids))]).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:24]
    p = Path(cache_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p / f"batch_{interval}_{start_ts}_{end_ts}_{digest}.json"


def _extract_batch_histories(payload: Any) -> Dict[str, List[dict[str, Any]]]:
    """
    Accept multiple response shapes from /batch-prices-history and normalize to:
    {token_id: [ {"t": ..., "p": ...}, ... ]}.
    """
    out: Dict[str, List[dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return out

    # Common shape: {"history": {"token": [...], ...}}
    history = payload.get("history")
    if isinstance(history, dict):
        for tok, arr in history.items():
            if isinstance(tok, str) and isinstance(arr, list):
                out[tok] = [x for x in arr if isinstance(x, dict)]
    if out:
        return out

    # Alternate shape: {"data": [{"token_id": "...", "history": [...]}, ...]}
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            tok = item.get("token_id") or item.get("market") or item.get("asset_id")
            hist = item.get("history")
            if tok is None or not isinstance(hist, list):
                continue
            out[str(tok)] = [x for x in hist if isinstance(x, dict)]
    return out


async def _fetch_batch_history_once(
    client: httpx.AsyncClient,
    token_ids: List[str],
    *,
    start_ts: int,
    end_ts: int,
    interval: str,
    cache_dir: str,
) -> Dict[str, List[dict[str, Any]]]:
    cache_file = _cache_file_for_batch(cache_dir, interval, start_ts, end_ts, token_ids)
    if cache_file.exists():
        try:
            return _extract_batch_histories(json.loads(cache_file.read_text(encoding="utf-8")))
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "markets": token_ids,
        "interval": interval,
        "fidelity": 500,
        "startTs": start_ts,
        "endTs": end_ts,
    }
    backoff = 0.8
    for _ in range(5):
        try:
            r = await client.post(CLOB_BATCH_PRICES_HISTORY, json=payload, timeout=90.0)
            if r.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue
            r.raise_for_status()
            js = r.json()
            cache_file.write_text(json.dumps(js), encoding="utf-8")
            return _extract_batch_histories(js)
        except Exception:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 10.0)
    return {}


async def fetch_batch_histories_async(
    token_ids: List[str],
    *,
    start_ts: int,
    end_ts: int,
    interval: str,
    cache_dir: str,
    chunk_size: int,
    concurrency: int,
) -> Dict[str, List[dict[str, Any]]]:
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    chunks = [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
    merged: Dict[str, List[dict[str, Any]]] = {}

    async with httpx.AsyncClient(
        headers={"User-Agent": "polymarket-active-classified/1.0", "Accept": "application/json"}
    ) as client:
        async def one(ch: List[str]) -> None:
            async with sem:
                data = await _fetch_batch_history_once(
                    client,
                    ch,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    interval=interval,
                    cache_dir=cache_dir,
                )
                merged.update(data)
                LOG.info(
                    "  … batch chunk done: %d token id(s) returned history (chunk size %d).",
                    len(data),
                    len(ch),
                )

        await asyncio.gather(*(one(ch) for ch in chunks))
    return merged


def fetch_clob_window(
    session: requests.Session,
    asset_id: str,
    start_ts: int,
    end_ts: int,
    limiter: SlidingWindowRateLimiter,
    *,
    interval: str,
) -> List[dict[str, Any]]:
    limiter.acquire()
    params = {
        "market": asset_id,
        "interval": interval,
        "fidelity": "500",
        "startTs": str(start_ts),
        "endTs": str(end_ts),
    }
    try:
        data = get_json(session, CLOB_PRICES_HISTORY, params=params, limiter=None)
    except requests.RequestException as exc:
        LOG.warning("CLOB %s: %s", asset_id[:16], exc)
        return []
    hist = data.get("history") if isinstance(data, dict) else None
    return hist if isinstance(hist, list) else []


def history_points_to_df(hist: List[dict[str, Any]]) -> pd.DataFrame:
    acc: Dict[int, float] = {}
    for pt in hist:
        if not isinstance(pt, dict):
            continue
        t = pt.get("t")
        p = pt.get("p")
        if t is None or p is None:
            continue
        try:
            acc[int(t)] = float(p)
        except (TypeError, ValueError):
            continue
    if not acc:
        return pd.DataFrame(columns=["timestamp", "price_yes"])
    ts = sorted(acc.keys())
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(ts, unit="s", utc=True), "price_yes": [acc[k] for k in ts]}
    ).sort_values("timestamp")


def fetch_clob_chunked(
    session: requests.Session,
    asset_id: str,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    limiter: SlidingWindowRateLimiter,
    *,
    interval: str,
) -> pd.DataFrame:
    win_start = pd.Timestamp(win_start).tz_convert("UTC")
    win_end = pd.Timestamp(win_end).tz_convert("UTC")
    start_i = int(win_start.timestamp())
    end_i = int(win_end.timestamp())
    chunk = int(CLOB_CHUNK_HOURS * 3600)
    acc: Dict[int, float] = {}
    cur = start_i
    while cur < end_i:
        nxt = min(end_i, cur + chunk)
        for pt in fetch_clob_window(session, asset_id, cur, nxt, limiter, interval=interval):
            if not isinstance(pt, dict):
                continue
            t, p = pt.get("t"), pt.get("p")
            if t is None or p is None:
                continue
            try:
                acc[int(t)] = float(p)
            except (TypeError, ValueError):
                pass
        cur = nxt
    if not acc:
        return pd.DataFrame(columns=["timestamp", "price_yes"])
    ts = sorted(acc.keys())
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(ts, unit="s", utc=True), "price_yes": [acc[k] for k in ts]}
    ).sort_values("timestamp")


def merge_finest_clob(
    session: requests.Session,
    asset_id: str,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    limiter: SlidingWindowRateLimiter,
) -> pd.DataFrame:
    """
    Use API-native series only: try finest resolutions in order without merging
    multiple intervals into one synthetic series (avoids mixing granularities).
    Order: 1m → max → 1h. Use the first interval that returns usable points (≥2);
    if none, return the last non-empty attempt or empty.
    """
    last_nonempty = pd.DataFrame(columns=["timestamp", "price_yes"])
    for interval in ("1m", "max", "1h"):
        df = fetch_clob_chunked(session, asset_id, win_start, win_end, limiter, interval=interval)
        if not df.empty:
            last_nonempty = df
        if len(df) >= 2:
            return df.sort_values("timestamp").reset_index(drop=True)
    return last_nonempty.sort_values("timestamp").reset_index(drop=True)


def gamma_volume_liquidity_snapshot(m: dict[str, Any]) -> Dict[str, Any]:
    """Gamma aggregates at fetch time (not time-varying per bucket)."""

    def _f(key: str) -> float:
        v = m.get(key)
        if v is None or v == "":
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "gamma_volume_num": _f("volumeNum"),
        "gamma_liquidity_num": _f("liquidityNum"),
        "gamma_volume": _f("volume"),
        "gamma_liquidity_clob": _f("liquidityClob"),
    }


def fetch_order_book_summary(
    session: requests.Session,
    token_id: str,
    limiter: SlidingWindowRateLimiter,
    *,
    top_n: int = ORDERBOOK_TOP_N,
) -> Dict[str, Any]:
    """
    Current CLOB order book (public API). Historical L2 is not exposed; all ``ob_*`` fields
    are **snapshot-at-fetch** values, repeated on each grid row for convenience.
    """
    nan = float("nan")
    empty: Dict[str, Any] = {
        "ob_best_bid": nan,
        "ob_best_ask": nan,
        "ob_mid": nan,
        "ob_spread": nan,
        "ob_bid_depth_topn": nan,
        "ob_ask_depth_topn": nan,
        "ob_imbalance_topn": nan,
        "ob_timestamp_ms": "",
    }
    limiter.acquire()
    try:
        r = session.get(CLOB_BOOK, params={"token_id": token_id}, timeout=45)
    except requests.RequestException as exc:
        LOG.warning("Order book HTTP error %s: %s", token_id[:16], exc)
        return empty
    if r.status_code != 200:
        LOG.warning("Order book %s for token %s…", r.status_code, token_id[:20])
        return empty
    try:
        book = r.json()
    except json.JSONDecodeError:
        return empty
    if not isinstance(book, dict):
        return empty

    def _depth(levels: Any) -> Tuple[float, float]:
        if not isinstance(levels, list) or not levels:
            return nan, nan
        best_p = nan
        tot_sz = 0.0
        for lvl in levels[:top_n]:
            if not isinstance(lvl, dict):
                continue
            try:
                p = float(lvl["price"])
                s = float(lvl["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if pd.isna(best_p):
                best_p = p
            tot_sz += s
        return best_p, tot_sz

    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bb_p, bb_sz = _depth(bids)
    ba_p, ba_sz = _depth(asks)
    out = dict(empty)
    out["ob_timestamp_ms"] = str(book.get("timestamp", ""))
    if not pd.isna(bb_p):
        out["ob_best_bid"] = float(bb_p)
    if not pd.isna(ba_p):
        out["ob_best_ask"] = float(ba_p)
    if not (pd.isna(bb_p) or pd.isna(ba_p)):
        out["ob_spread"] = float(ba_p - bb_p)
        out["ob_mid"] = float((ba_p + bb_p) / 2.0)
    if not (pd.isna(bb_sz) or pd.isna(ba_sz)) and (bb_sz + ba_sz) > 0:
        out["ob_bid_depth_topn"] = float(bb_sz)
        out["ob_ask_depth_topn"] = float(ba_sz)
        out["ob_imbalance_topn"] = float((bb_sz - ba_sz) / (bb_sz + ba_sz))
    return out


def augment_grid_with_bucket_hist(
    grid: pd.DataFrame,
    hist: pd.DataFrame,
    freq_str: str,
) -> pd.DataFrame:
    """Per grid bucket: count of CLOB prints and simple dispersion from raw prints in that bucket."""
    g = grid.copy()
    if hist.empty:
        g["raw_prints_in_bucket"] = 0
        g["intrabucket_price_range"] = np.nan
        g["intrabucket_price_stdev"] = np.nan
        return g

    h = hist.copy()
    h["bucket"] = pd.to_datetime(h["timestamp"], utc=True).dt.floor(freq_str)
    agg = h.groupby("bucket", sort=False)["price_yes"].agg(
        raw_prints_in_bucket="count",
        intrabucket_price_range=lambda s: float(s.max() - s.min()) if len(s) >= 1 else np.nan,
        intrabucket_price_stdev=lambda s: float(s.std(ddof=0)) if len(s) >= 2 else np.nan,
    )
    g["ts_key"] = pd.to_datetime(g["timestamp"], utc=True).dt.floor(freq_str)
    g = g.merge(
        agg,
        left_on="ts_key",
        right_index=True,
        how="left",
    )
    g.drop(columns=["ts_key"], inplace=True)
    g["raw_prints_in_bucket"] = g["raw_prints_in_bucket"].fillna(0).astype(int)
    for col in ("intrabucket_price_range", "intrabucket_price_stdev"):
        if col in g.columns:
            g[col] = g[col].astype(float)
    return g


def compute_market_features(
    grid: pd.DataFrame,
    hist: pd.DataFrame,
    ob: Dict[str, Any],
    lookback_hours: float,
    time_to_close_hours: float,
) -> Dict[str, float]:
    """Scalar features for PCA (one row per market)."""
    nan = float("nan")
    out: Dict[str, float] = {k: float("nan") for k in PCA_FEATURE_KEYS}

    px = pd.to_numeric(grid["price_yes"], errors="coerce")
    valid = px.notna() & (px > 0) & (px < 1)
    if "price_yes_norm" in grid.columns:
        ser = pd.to_numeric(grid["price_yes_norm"], errors="coerce").loc[valid]
    else:
        ser = px.loc[valid]
    jump_keys = (
        "feat_jump_zcount_ge_2",
        "feat_jump_zrate_ge_2",
        "feat_jump_zcount_ge_3",
        "feat_jump_zrate_ge_3",
        "feat_jump_count_ge_p95",
        "feat_jump_rate_ge_p95",
        "feat_jump_count_ge_p99",
        "feat_jump_rate_ge_p99",
        "feat_jump_max_abs",
        "feat_jump_top3_sum_abs",
        "feat_jump_top3_frac_total_move",
        "feat_return_volatility",
        "feat_time_to_close_hours",
    )
    out["feat_time_to_close_hours"] = float(time_to_close_hours)
    if len(ser) < 2:
        for k in jump_keys:
            out[k] = float("nan")
    else:
        # Use normalized probability-point changes (not raw price levels).
        r = ser.diff().dropna()
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) == 0:
            for k in (
                "feat_jump_zcount_ge_2",
                "feat_jump_zrate_ge_2",
                "feat_jump_zcount_ge_3",
                "feat_jump_zrate_ge_3",
                "feat_jump_count_ge_p95",
                "feat_jump_rate_ge_p95",
                "feat_jump_count_ge_p99",
                "feat_jump_rate_ge_p99",
                "feat_jump_max_abs",
                "feat_jump_top3_sum_abs",
                "feat_jump_top3_frac_total_move",
                "feat_return_volatility",
            ):
                out[k] = 0.0
        else:
            absr = r.abs()
            ret_std = float(r.std(ddof=0))
            out["feat_return_volatility"] = float(r.std(ddof=0))
            out["feat_jump_max_abs"] = float(absr.max())
            top3 = absr.nlargest(min(3, len(absr)))
            top3_sum = float(top3.sum())
            total_move = float(absr.sum())
            out["feat_jump_top3_sum_abs"] = top3_sum
            out["feat_jump_top3_frac_total_move"] = (
                top3_sum / total_move if total_move > 1e-12 else float("nan")
            )

            if ret_std > 1e-12:
                z_abs = absr / ret_std
                j2 = (z_abs >= 2.0).astype(float)
                j3 = (z_abs >= 3.0).astype(float)
                out["feat_jump_zcount_ge_2"] = float(j2.sum())
                out["feat_jump_zrate_ge_2"] = float(j2.mean())
                out["feat_jump_zcount_ge_3"] = float(j3.sum())
                out["feat_jump_zrate_ge_3"] = float(j3.mean())
            else:
                out["feat_jump_zcount_ge_2"] = 0.0
                out["feat_jump_zrate_ge_2"] = 0.0
                out["feat_jump_zcount_ge_3"] = 0.0
                out["feat_jump_zrate_ge_3"] = 0.0

            p95 = float(absr.quantile(0.95))
            p99 = float(absr.quantile(0.99))
            jp95 = (absr >= p95).astype(float)
            jp99 = (absr >= p99).astype(float)
            out["feat_jump_count_ge_p95"] = float(jp95.sum())
            out["feat_jump_rate_ge_p95"] = float(jp95.mean())
            out["feat_jump_count_ge_p99"] = float(jp99.sum())
            out["feat_jump_rate_ge_p99"] = float(jp99.mean())

    out["feat_raw_prints_per_hour"] = float(len(hist) / max(lookback_hours, 1e-9))

    mid = ob.get("ob_mid", nan)
    spr = ob.get("ob_spread", nan)
    imb = ob.get("ob_imbalance_topn", nan)
    bd = ob.get("ob_bid_depth_topn", nan)
    ad = ob.get("ob_ask_depth_topn", nan)

    out["feat_ob_spread"] = float(spr) if spr == spr else float("nan")
    out["feat_ob_mid"] = float(mid) if mid == mid else float("nan")
    out["feat_ob_imbalance_topn"] = float(imb) if imb == imb else float("nan")
    if mid == mid and abs(float(mid)) > 1e-12 and spr == spr:
        out["feat_ob_spread_over_mid"] = float(spr) / float(mid)
    else:
        out["feat_ob_spread_over_mid"] = float("nan")
    if bd == bd and ad == ad and bd > 0 and ad > 0:
        out["feat_ob_log_bid_ask_depth_ratio"] = float(np.log(bd / ad))
    else:
        out["feat_ob_log_bid_ask_depth_ratio"] = float("nan")

    return out


def zscore_features_for_pca(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    if len(out) < 2:
        LOG.warning("Only %d market(s): skipping StandardScaler for PCA features.", len(out))
        return out
    X = out[list(cols)].astype(float)
    X_imp = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    out[list(cols)] = scaler.fit_transform(X_imp)
    return out


def resample_to_price_grid(
    hist: pd.DataFrame,
    market_start: pd.Timestamp,
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    *,
    interval_hours: float,
    freq_str: str,
) -> pd.DataFrame:
    """
    Last price in each ``freq_str`` bucket, forward-filled on the full grid from
    ``grid_start`` (floor) to ``grid_end`` (ceil). ``normalized_step`` counts grid
    steps from ``market_start`` floored to the same frequency.
    """
    step_td = pd.Timedelta(hours=float(interval_hours))
    start_utc = pd.Timestamp(market_start).tz_convert("UTC")
    origin = start_utc.floor(freq_str)
    grid_start = pd.Timestamp(grid_start).tz_convert("UTC").floor(freq_str)
    grid_end = pd.Timestamp(grid_end).tz_convert("UTC").ceil(freq_str)
    if grid_end <= grid_start:
        return pd.DataFrame()

    idx = pd.date_range(grid_start, grid_end, freq=freq_str, tz="UTC")
    if hist.empty:
        out = pd.DataFrame({"timestamp": idx, "price_yes": float("nan")})
    else:
        # Bucket last traded price per interval only — no forward-fill across empty buckets
        # (avoids interpolating stale prices into gaps; NaNs remain unless API returned nothing).
        s_b = (
            hist.set_index("timestamp")["price_yes"]
            .sort_index()
            .astype(float)
            .resample(freq_str)
            .last()
        )
        out = s_b.reindex(idx).reset_index()
        out.columns = ["timestamp", "price_yes"]

    out["normalized_step"] = ((out["timestamp"] - origin) // step_td).astype("int64")
    out["hours_since_market_start"] = (out["timestamp"] - start_utc).dt.total_seconds() / 3600.0
    out["lookback_step"] = pd.Series(range(len(out)), dtype="int64")
    return out


def add_normalized_price_column(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-market normalized price centered on first valid grid price.
    """
    if grid.empty:
        return grid.copy()
    out = grid.copy()
    px = pd.to_numeric(out["price_yes"], errors="coerce")
    valid = px.notna() & (px > 0) & (px < 1)
    if not valid.any():
        out["price_yes_norm"] = float("nan")
        out["price_norm_anchor"] = float("nan")
        return out
    anchor = float(px.loc[valid].iloc[0])
    out["price_yes_norm"] = px - anchor
    out["price_norm_anchor"] = anchor
    return out


def atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    path = os.path.abspath(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=parent)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def run_pipeline(
    session: requests.Session,
    limiter: SlidingWindowRateLimiter,
    *,
    classifiers: Tuple[Tuple[str, str, str], ...],
    markets_per_category: int,
    lookback_days: float,
    interval_hours: float,
    max_scan_offset: int,
    min_volume_num: float,
    min_liquidity_num: float,
    use_orderbook: bool,
    cache_dir: str,
    batch_chunk_size: int,
    batch_concurrency: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    now = pd.Timestamp.now(tz="UTC")
    lookback = pd.Timedelta(days=float(lookback_days))
    lookback_hours = float(lookback_days) * 24.0
    freq_str = interval_hours_to_pandas_freq(interval_hours)
    raw_rows: List[dict[str, Any]] = []
    meta_rows: List[dict[str, Any]] = []
    feat_rows: List[dict[str, Any]] = []

    selected: List[Dict[str, Any]] = []
    for label, tag_id, slug in classifiers:
        n = 0
        seen_this_tag: set[str] = set()
        offset = 0
        while n < markets_per_category and offset < max_scan_offset:
            batch = fetch_active_markets_page(session, tag_id, offset=offset, limit=100, limiter=limiter)
            if not batch:
                break
            offset += len(batch)
            for m in batch:
                if n >= markets_per_category:
                    break
                mid = str(m.get("id", "")).strip()
                if not mid or mid in seen_this_tag:
                    continue
                tok = yes_token_for_market(m)
                if not tok:
                    continue
                if not market_passes_prefilter(
                    m,
                    min_volume_num=min_volume_num,
                    min_liquidity_num=min_liquidity_num,
                ):
                    continue
                seen_this_tag.add(mid)
                n += 1
                selected.append(
                    {
                        "label": label,
                        "tag_id": tag_id,
                        "slug": slug,
                        "market": m,
                        "token": tok,
                    }
                )

        if n < markets_per_category:
            LOG.warning(
                "%s (%s): only %d / %d markets (raise --max-scan-offset?)",
                label,
                tag_id,
                n,
                markets_per_category,
            )
        else:
            LOG.info("%s: collected %d markets", label, n)

    if not selected:
        return (pd.DataFrame(raw_rows), pd.DataFrame(meta_rows), pd.DataFrame(feat_rows))

    token_ids = [str(x["token"]) for x in selected]
    win_end = now
    global_start = int((now - lookback).timestamp())
    global_end = int(now.timestamp())

    LOG.info(
        "Fetching CLOB prices (batch API): %d markets, window [%s .. %s] UTC, primary interval=1m "
        "(fallback per-market if sparse).",
        len(token_ids),
        pd.Timestamp(global_start, unit="s", tz="UTC").isoformat(),
        pd.Timestamp(global_end, unit="s", tz="UTC").isoformat(),
    )
    # Batch uses finest interval first (up to 1m); per-token fallback uses merge_finest_clob if sparse.
    coarse_histories = asyncio.run(
        fetch_batch_histories_async(
            token_ids,
            start_ts=global_start,
            end_ts=global_end,
            interval="1m",
            cache_dir=cache_dir,
            chunk_size=batch_chunk_size,
            concurrency=batch_concurrency,
        )
    )
    got_hist = sum(1 for t in token_ids if coarse_histories.get(t))
    LOG.info(
        "Batch prices received history for %d / %d tokens (others may use per-token 1m/max/1h fallback).",
        got_hist,
        len(token_ids),
    )

    for item in selected:
        m = item["market"]
        mid = str(m.get("id", "")).strip()
        label = str(item["label"])
        tag_id = str(item["tag_id"])
        slug = str(item["slug"])
        tok = str(item["token"])

        start_dt = pd.to_datetime(m.get("startDate"), utc=True, errors="coerce")
        if pd.isna(start_dt):
            start_dt = now - lookback
        win_start = max(start_dt, now - lookback)
        hist = history_points_to_df(coarse_histories.get(tok, []))
        hist = hist[(hist["timestamp"] >= win_start) & (hist["timestamp"] <= win_end)].copy()

        # Single-resolution API fallback only when batch history is too sparse (still no mixing intervals).
        if len(hist) < 2:
            hist = merge_finest_clob(session, tok, win_start, win_end, limiter)

        grid = resample_to_price_grid(
            hist,
            start_dt,
            win_start,
            win_end,
            interval_hours=interval_hours,
            freq_str=freq_str,
        )
        grid = add_normalized_price_column(grid)
        grid_out = augment_grid_with_bucket_hist(grid, hist, freq_str) if not grid.empty else grid
        ob = fetch_order_book_summary(session, tok, limiter) if use_orderbook else {
            "ob_best_bid": float("nan"),
            "ob_best_ask": float("nan"),
            "ob_mid": float("nan"),
            "ob_spread": float("nan"),
            "ob_bid_depth_topn": float("nan"),
            "ob_ask_depth_topn": float("nan"),
            "ob_imbalance_topn": float("nan"),
            "ob_timestamp_ms": "",
        }
        gamma_snap = gamma_volume_liquidity_snapshot(m)
        end_dt = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")
        time_to_close_hours = float((end_dt - now).total_seconds() / 3600.0) if not pd.isna(end_dt) else float("nan")
        feats = compute_market_features(
            grid_out if not grid_out.empty else grid,
            hist,
            ob,
            lookback_hours,
            time_to_close_hours,
        )

        meta_base: Dict[str, Any] = {
            "market_id": mid,
            "classifier_label": label,
            "tag_id": tag_id,
            "tag_slug": slug,
            "question": m.get("question"),
            "startDate": m.get("startDate"),
            "endDate": m.get("endDate"),
            "volumeNum": m.get("volumeNum"),
            "liquidityNum": m.get("liquidityNum"),
            "interval_hours": float(interval_hours),
            "lookback_days": float(lookback_days),
            "lookback_hours_used": float(lookback_hours),
            "clob_points_raw": int(len(hist)),
            "grid_rows": int(len(grid_out)),
            "grid_freq": freq_str,
            "ob_note": "disabled" if not use_orderbook else "snapshot_at_fetch_not_historical",
            "price_norm_anchor": float(grid_out["price_norm_anchor"].iloc[0])
            if ("price_norm_anchor" in grid_out.columns and len(grid_out) > 0)
            else float("nan"),
        }
        meta_base.update(gamma_snap)
        meta_base.update(ob)
        meta_base.update(feats)
        meta_rows.append(meta_base)

        feat_row: Dict[str, Any] = {
            "market_id": mid,
            "classifier_label": label,
            "tag_id": tag_id,
            "tag_slug": slug,
            **feats,
        }
        feat_rows.append(feat_row)

        snap_cols = {
            **gamma_snap,
            "ob_best_bid": ob.get("ob_best_bid"),
            "ob_best_ask": ob.get("ob_best_ask"),
            "ob_mid": ob.get("ob_mid"),
            "ob_spread": ob.get("ob_spread"),
            "ob_imbalance_topn": ob.get("ob_imbalance_topn"),
            "ob_bid_depth_topn": ob.get("ob_bid_depth_topn"),
            "ob_ask_depth_topn": ob.get("ob_ask_depth_topn"),
            "ob_timestamp_ms": ob.get("ob_timestamp_ms", ""),
        }

        if grid_out.empty:
            raw_rows.append(
                {
                    "market_id": mid,
                    "classifier_label": label,
                    "tag_id": tag_id,
                    "tag_slug": slug,
                    "question": m.get("question"),
                    "timestamp_iso": "",
                    "unix_ts": "",
                    "price_yes": float("nan"),
                    "price_yes_norm": float("nan"),
                    "normalized_step": "",
                    "lookback_step": "",
                    "hours_since_market_start": float("nan"),
                    "gamma_market_start_iso": pd.Timestamp(start_dt).isoformat() if not pd.isna(start_dt) else "",
                    "grid_interval": freq_str,
                    "interval_hours": float(interval_hours),
                    "lookback_days": float(lookback_days),
                    "raw_prints_in_bucket": "",
                    "intrabucket_price_range": "",
                    "intrabucket_price_stdev": "",
                    **snap_cols,
                    "note": "no_clob_in_window",
                }
            )
            continue

        for _, r in grid_out.iterrows():
            ts = pd.Timestamp(r["timestamp"])
            raw_rows.append(
                {
                    "market_id": mid,
                    "classifier_label": label,
                    "tag_id": tag_id,
                    "tag_slug": slug,
                    "question": m.get("question"),
                    "timestamp_iso": ts.isoformat(),
                    "unix_ts": int(ts.timestamp()),
                    "price_yes": float(r["price_yes"]) if pd.notna(r["price_yes"]) else float("nan"),
                    "price_yes_norm": float(r["price_yes_norm"]) if pd.notna(r.get("price_yes_norm")) else float("nan"),
                    "normalized_step": int(r["normalized_step"]),
                    "lookback_step": int(r["lookback_step"]),
                    "hours_since_market_start": float(r["hours_since_market_start"]),
                    "gamma_market_start_iso": pd.Timestamp(start_dt).tz_convert("UTC").isoformat(),
                    "grid_interval": freq_str,
                    "interval_hours": float(interval_hours),
                    "lookback_days": float(lookback_days),
                    "raw_prints_in_bucket": int(r.get("raw_prints_in_bucket", 0)),
                    "intrabucket_price_range": float(r["intrabucket_price_range"])
                    if pd.notna(r.get("intrabucket_price_range"))
                    else float("nan"),
                    "intrabucket_price_stdev": float(r["intrabucket_price_stdev"])
                    if pd.notna(r.get("intrabucket_price_stdev"))
                    else float("nan"),
                    **snap_cols,
                    "note": "",
                }
            )

    return (
        pd.DataFrame(raw_rows),
        pd.DataFrame(meta_rows),
        pd.DataFrame(feat_rows),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=f"JSON config file (default: look for {CONFIG_FILENAME} in cwd, then next to this script).",
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not ask terminal questions; use values from the JSON config (or built-in defaults) only.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg, cfg_path = load_merged_config(args.config)
    if cfg_path:
        LOG.info("Loaded config %s", cfg_path)
    else:
        LOG.info("No %s in cwd or script dir; using built-in defaults", CONFIG_FILENAME)

    use_prompt = sys.stdin.isatty() and not args.no_prompt
    if use_prompt:
        try:
            cfg = interactive_fill_settings(cfg)
        except EOFError:
            LOG.error("EOF while reading prompts; use --no-prompt for non-interactive runs.")
            return 1
        except ValueError as exc:
            LOG.error("Invalid number: %s", exc)
            return 1

    markets_per_category = int(cfg["markets_per_category"])
    interval_hours = float(cfg["interval_hours"])
    lookback_days = float(cfg["lookback_days"])
    max_scan_offset = int(cfg["max_scan_offset"])
    raw_out = str(cfg["raw_csv"])
    meta_out = str(cfg["metadata_csv"])
    features_out = str(cfg["features_csv"])
    min_volume_num = float(cfg.get("min_volume_num", DEFAULT_MIN_VOLUME_NUM))
    min_liquidity_num = float(cfg.get("min_liquidity_num", DEFAULT_MIN_LIQUIDITY_NUM))
    use_orderbook = bool(cfg.get("use_orderbook", DEFAULT_USE_ORDERBOOK))
    cache_dir = str(cfg.get("cache_dir", DEFAULT_CACHE_DIR))
    batch_chunk_size = int(cfg.get("batch_chunk_size", BATCH_TOKEN_CHUNK_SIZE))
    batch_concurrency = int(cfg.get("batch_concurrency", BATCH_MAX_CONCURRENCY))

    try:
        interval_hours_to_pandas_freq(interval_hours)
    except ValueError as exc:
        LOG.error("%s", exc)
        return 1

    session = http_session()
    limiter = SlidingWindowRateLimiter(CLOB_MAX_REQUESTS, CLOB_WINDOW_SEC)
    raw_df, meta_df, feat_df = run_pipeline(
        session,
        limiter,
        classifiers=DEFAULT_CLASSIFIERS,
        markets_per_category=markets_per_category,
        lookback_days=lookback_days,
        interval_hours=interval_hours,
        max_scan_offset=max_scan_offset,
        min_volume_num=min_volume_num,
        min_liquidity_num=min_liquidity_num,
        use_orderbook=use_orderbook,
        cache_dir=cache_dir,
        batch_chunk_size=batch_chunk_size,
        batch_concurrency=batch_concurrency,
    )
    if meta_df.empty:
        LOG.error("No markets collected.")
        return 1
    atomic_write_csv(raw_out, raw_df)
    atomic_write_csv(meta_out, meta_df)

    id_cols = ["market_id", "classifier_label", "tag_id", "tag_slug"]
    if feat_df.empty:
        atomic_write_csv(
            features_out,
            pd.DataFrame(columns=list(id_cols) + list(PCA_FEATURE_KEYS)),
        )
    else:
        scaled = zscore_features_for_pca(feat_df, PCA_FEATURE_KEYS)
        for c in id_cols:
            if c in feat_df.columns:
                scaled[c] = feat_df[c].values
        ordered = [c for c in id_cols if c in scaled.columns] + list(PCA_FEATURE_KEYS)
        atomic_write_csv(features_out, scaled[ordered])

    LOG.info(
        "Wrote %d raw -> %s; %d metadata -> %s; %d feature rows -> %s",
        len(raw_df),
        raw_out,
        len(meta_df),
        meta_out,
        len(feat_df),
        features_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
