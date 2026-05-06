#!/usr/bin/env python3
"""
Polymarket sports — closed-market discovery (Gamma) + Yes-token price history (CLOB)
+ scalar features for a PCA-ready, Z-scored feature matrix, plus a long-format raw
price CSV (see ``--raw-timeseries-out``): **hourly** CLOB first to find the first
two consecutive hours where the primary token is skewed (≥95% or ≤5%), treat the first
hour as inferred **game end**, then pull **1m** CLOB only for the **two hours** before
that point. Markets can be restricted to recent ``endDate`` (e.g. last 30 days).

Discovery takes the first N matching head-to-head moneylines per sport with Gamma
``endDate`` descending (recently closed first), then fetches CLOB without dropping
markets solely for sparse PCA features.

CLOB note: the public API requires the query parameter name ``market`` (asset id).
Some docs refer to this as the token id; we document both in README.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_PRICES_HISTORY = "https://clob.polymarket.com/prices-history"

GAME_LINES_TAG = "100639"

# (sport_label, league_tag_for_markets_fallback, series_id(s) for /events game lines)
# Soccer rotates several leagues so closed matchups are easier to find.
SPORT_SPECS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("NBA", "745", ("10345",)),
    ("NFL", "450", ("10187",)),
    ("MLB", "100381", ("3",)),
    ("NHL", "899", ("10346",)),
    ("Soccer", "100350", ("10188", "10193", "10204", "10189", "10194")),
)

CLOB_MAX_REQUESTS = 900
CLOB_WINDOW_SEC = 10.0

CLIP_LOW, CLIP_HIGH = 0.001, 0.999
PRE_RES_HOURS = 48.0  # feature window: [-48h, 0] relative to resolution
# Single bounded CLOB window before resolution (API rejects very long spans).
CLOB_LOOKBACK_DAYS = 10
# Chunk long [start, end] requests for prices-history (1m over multi-day windows).
CLOB_GAME_CHUNK_HOURS = 18.0
GAME_END_CLOB_BUFFER_SEC = 3600
# Resolution skew: two consecutive hourly buckets (last print in hour) extreme → game end.
DEFAULT_SKEW_HIGH = 0.95
DEFAULT_SKEW_LOW = 0.05
DEFAULT_SKEW_LOOKBACK_HOURS = 336.0  # 14d of hourly CLOB before Gamma endDate
DEFAULT_RAW_SEGMENT_HOURS = 2.0  # 1m fetch for this many hours ending at inferred game end
DEFAULT_MARKET_END_WITHIN_LAST_DAYS = 30
SEGMENT_1M_END_BUFFER_SEC = 120

DEFAULT_PER_SPORT = 20
DEFAULT_FEATURES = "polymarket_sports_pca_features.csv"
DEFAULT_METADATA = "polymarket_sports_pca_metadata.csv"
DEFAULT_RAW_TIMESERIES = "polymarket_sports_raw_timeseries.csv"

FEATURE_NUMERIC_KEYS = (
    "momentum_1h",
    "momentum_6h",
    "momentum_24h",
    "volatility_24h",
    "efficiency_ratio",
    "ripeness_hours",
)
EMPTY_FEATURE_ROW = {k: float("nan") for k in FEATURE_NUMERIC_KEYS}


@dataclass(frozen=True)
class SelectedMarket:
    sport: str
    market: dict[str, Any]
    question: str
    mid: str
    yes_tok: str
    end_dt: pd.Timestamp
    start_dt: pd.Timestamp
    resolved_primary: int
    voln_f: float

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("sports_features")


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


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "polymarket-sports-pca-features/1.0",
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


def logit_price(p: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, CLIP_LOW, CLIP_HIGH)
    return np.log(p / (1.0 - p))


def yes_token_index(outcomes: Sequence[str]) -> int:
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() == "yes":
            return i
    return 0


_VS_SPLIT = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)


def is_binary_yes_no(outcomes: Any) -> bool:
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return False
    a, b = str(outcomes[0]).lower(), str(outcomes[1]).lower()
    return {a, b} == {"yes", "no"}


def is_head_to_head_game_question(question: Any) -> bool:
    """
    Matchups like 'Lakers vs. Celtics' or 'Cowboys vs. Eagles'.
    Excludes props ('Will …'), spreads, totals, and soccer 1x2 'Will X win' binaries.
    """
    if not question or not isinstance(question, str):
        return False
    q = question.strip()
    low = q.lower()
    if low.startswith("will ") or low.startswith("spread:") or ": o/u" in low:
        return False
    if "o/u" in low and " vs" in low:
        return False
    if _VS_SPLIT.search(q) is None:
        return False
    parts = [p.strip() for p in _VS_SPLIT.split(q, maxsplit=1)]
    if len(parts) != 2:
        return False
    a, b = parts[0], parts[1]
    if len(a) < 2 or len(b) < 2:
        return False
    # Drop obvious non-matchup tails (e.g. ': O/U 47.5' sometimes glued — split already avoided)
    if a.lower().startswith("spread:") or b.lower().startswith("spread:"):
        return False
    return True


def is_two_team_moneyline(outcomes: Any) -> bool:
    """Two named outcomes that are not the generic Yes/No pair."""
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return False
    if is_binary_yes_no(outcomes):
        return False
    return all(str(x).strip() for x in outcomes)


def primary_token_index_for_market(outcomes: Any) -> int:
    """Price series for first listed team / Yes side."""
    if is_binary_yes_no(outcomes):
        return yes_token_index(outcomes)
    return 0


def resolved_primary_label(outcomes: Any, outcome_prices_raw: Any) -> Optional[int]:
    """1 if primary (first) outcome won, 0 if second won; None if ambiguous."""
    prices = parse_gamma_json_field(outcome_prices_raw)
    if not isinstance(outcomes, list) or not isinstance(prices, list) or len(outcomes) != len(prices):
        return None
    i = primary_token_index_for_market(outcomes)
    try:
        p0 = float(prices[i])
    except (TypeError, ValueError, IndexError):
        return None
    if p0 > 0.5:
        return 1
    if p0 < 0.5:
        return 0
    return None


def fetch_closed_markets_page(
    session: requests.Session,
    tag_id: str,
    *,
    offset: int,
    limit: int,
) -> List[dict[str, Any]]:
    params = {
        "closed": "true",
        "tag_id": tag_id,
        "limit": str(limit),
        "offset": str(offset),
        "order": "endDate",
        "ascending": "false",
    }
    data = get_json(session, GAMMA_MARKETS, params=params)
    if not isinstance(data, list):
        raise TypeError(f"Expected list from Gamma markets, got {type(data)}")
    return data


def fetch_closed_game_events_page(
    session: requests.Session,
    series_id: str,
    *,
    offset: int,
    limit: int,
) -> List[dict[str, Any]]:
    """Game-line events (fast path for real 'Team vs Team' listings)."""
    params = {
        "closed": "true",
        "series_id": series_id,
        "tag_id": GAME_LINES_TAG,
        "limit": str(limit),
        "offset": str(offset),
        "order": "endDate",
        "ascending": "false",
    }
    data = get_json(session, GAMMA_EVENTS, params=params)
    if not isinstance(data, list):
        raise TypeError(f"Expected list from Gamma events, got {type(data)}")
    return data


def clob_history_unbounded(
    session: requests.Session,
    asset_id: str,
    limiter: SlidingWindowRateLimiter,
) -> List[dict[str, Any]]:
    """Fallback: hourly history without start/end (often sparse for old markets)."""
    limiter.acquire()
    params = {"market": asset_id, "interval": "1h", "fidelity": "500"}
    try:
        data = get_json(session, CLOB_PRICES_HISTORY, params=params, limiter=None)
    except requests.RequestException:
        return []
    hist = data.get("history") if isinstance(data, dict) else None
    return hist if isinstance(hist, list) else []


def fetch_clob_history_window(
    session: requests.Session,
    asset_id: str,
    start_ts: int,
    end_ts: int,
    limiter: SlidingWindowRateLimiter,
    *,
    interval: str = "1h",
) -> List[dict[str, Any]]:
    """Polymarket expects ``market=<asset_id>`` (Yes outcome token id)."""
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
        LOG.warning("CLOB history error %s: %s", asset_id[:20], exc)
        return []
    hist = data.get("history") if isinstance(data, dict) else None
    if not isinstance(hist, list):
        return []
    return hist


def _ingest_history_points(all_rows: Dict[int, float], hist: List[dict[str, Any]]) -> None:
    for pt in hist:
        if not isinstance(pt, dict):
            continue
        t, p = pt.get("t"), pt.get("p")
        if t is None or p is None:
            continue
        try:
            all_rows[int(t)] = float(p)
        except (TypeError, ValueError):
            continue


def merge_clob_history(
    session: requests.Session,
    asset_id: str,
    end_dt: pd.Timestamp,
    limiter: SlidingWindowRateLimiter,
    *,
    lookback_hours: Optional[float] = None,
    intervals: Tuple[str, ...] = ("1h",),
) -> pd.DataFrame:
    """
    Bounded window before resolution, then optional unbounded fill.
    Multiple ``interval`` values merge into one series (later passes can refine timestamps).
    """
    all_rows: Dict[int, float] = {}
    end_ts = int(end_dt.timestamp()) + 3600
    if lookback_hours is not None:
        start_ts = int((end_dt - pd.Timedelta(hours=float(lookback_hours))).timestamp())
    else:
        start_ts = int((end_dt - pd.Timedelta(days=CLOB_LOOKBACK_DAYS)).timestamp())

    for iv in intervals:
        hist = fetch_clob_history_window(session, asset_id, start_ts, end_ts, limiter, interval=iv)
        _ingest_history_points(all_rows, hist)

    if len(all_rows) < 2:
        _ingest_history_points(all_rows, clob_history_unbounded(session, asset_id, limiter))

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "price_yes"])
    ts = sorted(all_rows.keys())
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="s", utc=True), "price_yes": [all_rows[k] for k in ts]})
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df


def primary_outcome_label(outcomes: Any) -> str:
    if not isinstance(outcomes, list) or not outcomes:
        return ""
    i = primary_token_index_for_market(outcomes)
    if 0 <= i < len(outcomes):
        return str(outcomes[i]).strip()
    return ""


def gamma_trade_context(m: dict[str, Any]) -> dict[str, Any]:
    """Snapshot fields from Gamma (mostly static per market; repeated on each raw row)."""

    def _num(key: str) -> float:
        v = m.get(key)
        if v is None or v == "":
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "volume_num": _num("volumeNum"),
        "liquidity_num": _num("liquidityNum"),
        "liquidity": _num("liquidity"),
        "liquidity_clob": _num("liquidityClob"),
        "liquidity_amm": _num("liquidityAmm"),
        "volume": _num("volume"),
        "volume_clob": _num("volumeClob"),
        "volume_1wk_clob": _num("volume1wkClob"),
        "best_bid": m.get("bestBid") if m.get("bestBid") not in (None, "") else "",
        "best_ask": m.get("bestAsk") if m.get("bestAsk") not in (None, "") else "",
    }


def fetch_hourly_clob_before_gamma_end(
    session: requests.Session,
    asset_id: str,
    gamma_end: pd.Timestamp,
    lookback_hours: float,
    limiter: SlidingWindowRateLimiter,
) -> pd.DataFrame:
    """Hourly CLOB only, up to ``gamma_end`` (no end buffer) — infer game segment from this."""
    gamma_end = pd.Timestamp(gamma_end).tz_convert("UTC")
    win_start = gamma_end - pd.Timedelta(hours=float(lookback_hours))
    raw_pts = fetch_clob_window_chunked(
        session,
        asset_id,
        win_start,
        gamma_end,
        limiter,
        interval="1h",
        end_buffer_sec=0,
    )
    acc: Dict[int, float] = {}
    _ingest_history_points(acc, raw_pts)
    return history_points_to_dataframe(acc)


def detect_skew_segment_bounds(
    hourly_df: pd.DataFrame,
    *,
    gamma_end: pd.Timestamp,
    segment_hours: float,
    skew_high: float,
    skew_low: float,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], str]:
    """
    Resample to 1h (last print per bucket, forward-filled), find the first two consecutive
    hours where the primary token is skewed (``>= skew_high`` or ``<= skew_low``). The
    **first** hour of the pair is inferred **game end**; **game start** is ``segment_hours``
    earlier (typically 2h for the 1m pull).
    """
    gamma_end = pd.Timestamp(gamma_end).tz_convert("UTC")
    if hourly_df.empty or len(hourly_df) < 2:
        return None, None, "insufficient_hourly_clob"
    h = hourly_df.loc[hourly_df["timestamp"] <= gamma_end].copy()
    if h.empty:
        return None, None, "no_hourly_before_gamma_end"
    s = (
        h.set_index("timestamp")["price_yes"]
        .sort_index()
        .astype(float)
        .clip(0.0, 1.0)
        .resample("1h")
        .last()
        .ffill()
    )
    if s.index.tz is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")
    s = s[s.index <= gamma_end]
    s = s.dropna()
    if len(s) < 2:
        return None, None, "sparse_hourly_resample"

    def skewed(v: float) -> bool:
        if pd.isna(v):
            return False
        fv = float(v)
        return fv >= float(skew_high) or fv <= float(skew_low)

    for i in range(len(s) - 1):
        if skewed(float(s.iloc[i])) and skewed(float(s.iloc[i + 1])):
            game_end = pd.Timestamp(s.index[i]).tz_convert("UTC")
            game_start = game_end - pd.Timedelta(hours=float(segment_hours))
            return game_start, game_end, "ok"

    return None, None, "no_skew_streak"


def fetch_1m_clob_segment_only(
    session: requests.Session,
    asset_id: str,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    limiter: SlidingWindowRateLimiter,
    *,
    end_buffer_sec: int = SEGMENT_1M_END_BUFFER_SEC,
) -> pd.DataFrame:
    """``interval=1m`` only, after hourly skew detection (no prior 1m fetch)."""
    raw_pts = fetch_clob_window_chunked(
        session,
        asset_id,
        seg_start,
        seg_end,
        limiter,
        interval="1m",
        end_buffer_sec=end_buffer_sec,
    )
    acc: Dict[int, float] = {}
    _ingest_history_points(acc, raw_pts)
    return history_points_to_dataframe(acc)


def fetch_clob_window_chunked(
    session: requests.Session,
    asset_id: str,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    limiter: SlidingWindowRateLimiter,
    *,
    interval: str,
    chunk_hours: float = CLOB_GAME_CHUNK_HOURS,
    end_buffer_sec: int = GAME_END_CLOB_BUFFER_SEC,
) -> List[dict[str, Any]]:
    """Query prices-history in time chunks (helps long ``1m`` windows)."""
    win_start = pd.Timestamp(win_start).tz_convert("UTC")
    win_end = pd.Timestamp(win_end).tz_convert("UTC")
    start_i = int(win_start.timestamp())
    end_i = int(win_end.timestamp()) + int(end_buffer_sec)
    chunk_sec = int(max(3600.0, float(chunk_hours) * 3600.0))
    out: List[dict[str, Any]] = []
    cur = start_i
    while cur < end_i:
        nxt = min(end_i, cur + chunk_sec)
        out.extend(fetch_clob_history_window(session, asset_id, cur, nxt, limiter, interval=interval))
        cur = nxt
    return out


def history_points_to_dataframe(all_rows: Dict[int, float]) -> pd.DataFrame:
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "price_yes"])
    ts = sorted(all_rows.keys())
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(ts, unit="s", utc=True), "price_yes": [all_rows[k] for k in ts]}
    ).sort_values("timestamp")


def apply_minute_grid_ffill(hist: pd.DataFrame, win_start: pd.Timestamp, win_end: pd.Timestamp) -> pd.DataFrame:
    """Regular 1-minute UTC index from first to last observation in-window (forward-filled)."""
    if hist.empty:
        return hist
    win_start = pd.Timestamp(win_start).tz_convert("UTC")
    win_end = pd.Timestamp(win_end).tz_convert("UTC")
    h = hist.loc[(hist["timestamp"] >= win_start) & (hist["timestamp"] <= win_end)].copy()
    if h.empty:
        return h
    s = h.set_index("timestamp")["price_yes"].sort_index().astype(float)
    idx = pd.date_range(s.index.min().floor("min"), s.index.max().ceil("min"), freq="1min", tz="UTC")
    s = s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
    out = s.reset_index()
    out.columns = ["timestamp", "price_yes"]
    return out.dropna(subset=["price_yes"])


def build_metadata_row(sport: str, m: dict[str, Any]) -> dict[str, Any]:
    outcomes = parse_gamma_json_field(m.get("outcomes"))
    toks = parse_gamma_json_field(m.get("clobTokenIds"))
    cat = m.get("category")
    if cat is None or (isinstance(cat, str) and not cat.strip()):
        cat = sport
    return {
        "id": str(m.get("id", "")),
        "question": m.get("question"),
        "category": cat,
        "volumeNum": m.get("volumeNum"),
        "liquidityNum": m.get("liquidityNum"),
        "startDate": m.get("startDate"),
        "endDate": m.get("endDate"),
        "clobTokenIds": json.dumps(toks) if toks is not None else None,
        "sport_bucket": sport,
    }


def compute_features_for_market(
    hist: pd.DataFrame,
    end_dt: pd.Timestamp,
    start_dt: pd.Timestamp,
    volume_num: float,
) -> Optional[dict[str, float]]:
    if hist.empty:
        return None

    end_dt = pd.Timestamp(end_dt).tz_convert("UTC")
    start_dt = pd.Timestamp(start_dt).tz_convert("UTC")

    h = hist.loc[hist["timestamp"] <= end_dt].copy()
    if h.empty:
        return None

    # Hourly series: resample irregular CLOB prints, then align to the analysis grid
    grid_start = end_dt - pd.Timedelta(hours=PRE_RES_HOURS + 24)
    grid_end = end_dt
    grid_index = pd.date_range(grid_start, grid_end, freq="1h", tz="UTC")
    ser_raw = (
        h.set_index("timestamp")["price_yes"]
        .sort_index()
        .astype(float)
        .dropna()
        .resample("1h")
        .ffill()
    )
    ser = ser_raw.reindex(grid_index).ffill().bfill()
    if ser.isna().all() or not np.isfinite(ser.to_numpy(dtype=float)).any():
        return None

    L = pd.Series(logit_price(ser.values), index=ser.index)
    L_end = float(L.iloc[-1])

    def L_at_hours_before(delta_h: float) -> Optional[float]:
        ts_cut = end_dt - pd.Timedelta(hours=delta_h)
        sub = L[L.index <= ts_cut]
        if sub.empty:
            return None
        return float(sub.iloc[-1])

    l1 = L_at_hours_before(1.0)
    l6 = L_at_hours_before(6.0)
    l24 = L_at_hours_before(24.0)
    mom_1h = (L_end - l1) if l1 is not None else np.nan
    mom_6h = (L_end - l6) if l6 is not None else np.nan
    mom_24h = (L_end - l24) if l24 is not None else np.nan

    Lw = L.loc[(L.index >= end_dt - pd.Timedelta(hours=PRE_RES_HOURS)) & (L.index <= end_dt)]
    if Lw.empty:
        return None
    roll = Lw.rolling(window=24, min_periods=2).std()
    vol_24h = float(np.nanmean(roll.values)) if len(roll) else np.nan

    p48 = ser.loc[(ser.index >= end_dt - pd.Timedelta(hours=PRE_RES_HOURS)) & (ser.index <= end_dt)]
    raw = p48.astype(float)
    pmax, pmin = float(raw.max()), float(raw.min())
    span = max(pmax - pmin, 1e-6)
    eff = float(volume_num) / span if volume_num is not None and not pd.isna(volume_num) else np.nan

    if pd.isna(start_dt):
        start_dt = end_dt - pd.Timedelta(days=1)
    ripeness = abs((end_dt - start_dt).total_seconds()) / 3600.0
    ripeness = max(ripeness, 1e-6)

    return {
        "momentum_1h": mom_1h,
        "momentum_6h": mom_6h,
        "momentum_24h": mom_24h,
        "volatility_24h": vol_24h,
        "efficiency_ratio": eff,
        "ripeness_hours": ripeness,
    }


def extract_feature_matrix(
    session: requests.Session,
    limiter: SlidingWindowRateLimiter,
    per_sport: int,
    *,
    max_offset_cap: int = 2500,
    fetch_lookback_hours: float = 72.0,
    raw_minute_grid: bool = False,
    market_end_within_last_days: float = DEFAULT_MARKET_END_WITHIN_LAST_DAYS,
    market_end_after: Optional[pd.Timestamp] = None,
    market_end_before: Optional[pd.Timestamp] = None,
    skew_lookback_hours: float = DEFAULT_SKEW_LOOKBACK_HOURS,
    skew_high: float = DEFAULT_SKEW_HIGH,
    skew_low: float = DEFAULT_SKEW_LOW,
    raw_segment_hours: float = DEFAULT_RAW_SEGMENT_HOURS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per sport: first ``per_sport`` markets passing filters (Gamma ``endDate`` desc).
    PCA features use a long lookback. Raw CSV: hourly CLOB → skew streak → 1m only
    on the ``raw_segment_hours`` window ending at inferred game end.
    """
    meta_rows: List[dict[str, Any]] = []
    feat_rows: List[dict[str, Any]] = []
    raw_rows: List[dict[str, Any]] = []

    page_limit = 200
    max_offset = max_offset_cap
    fetch_lb = max(float(fetch_lookback_hours), float(CLOB_LOOKBACK_DAYS) * 24.0)

    for sport, tag_id, series_ids in SPORT_SPECS:
        seen_ids: set[str] = set()
        pending: List[SelectedMarket] = []

        def try_select_market(m: dict[str, Any], ev_title: Optional[str] = None) -> None:
            if len(pending) >= per_sport:
                return
            mid = str(m.get("id", "")).strip()
            if not mid or mid in seen_ids:
                return
            smt = (m.get("sportsMarketType") or "").strip().lower()
            if smt in {"spreads", "spread", "totals", "total"}:
                return
            q = (m.get("question") or "").strip() or (ev_title or "").strip()
            if not is_head_to_head_game_question(q):
                return
            outcomes = parse_gamma_json_field(m.get("outcomes"))
            if not is_two_team_moneyline(outcomes):
                return
            lab = resolved_primary_label(outcomes, m.get("outcomePrices"))
            if lab is None:
                return
            toks = parse_gamma_json_field(m.get("clobTokenIds"))
            if not isinstance(toks, list) or len(toks) < 2:
                return

            tok_idx = primary_token_index_for_market(outcomes)
            yes_tok = str(toks[tok_idx])

            end_dt = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")
            start_dt = pd.to_datetime(m.get("startDate"), utc=True, errors="coerce")
            if pd.isna(end_dt) or pd.isna(start_dt):
                return
            now_utc = pd.Timestamp.now(tz="UTC")
            if end_dt >= now_utc:
                return
            if market_end_within_last_days > 0:
                lo = now_utc - pd.Timedelta(days=float(market_end_within_last_days))
                if end_dt < lo or end_dt > now_utc:
                    return
            if market_end_after is not None and end_dt < market_end_after:
                return
            if market_end_before is not None and end_dt >= market_end_before:
                return

            voln = m.get("volumeNum")
            try:
                voln_f = float(voln) if voln is not None else float("nan")
            except (TypeError, ValueError):
                voln_f = float("nan")

            seen_ids.add(mid)
            pending.append(
                SelectedMarket(
                    sport=sport,
                    market=m,
                    question=q,
                    mid=mid,
                    yes_tok=yes_tok,
                    end_dt=end_dt,
                    start_dt=start_dt,
                    resolved_primary=int(lab),
                    voln_f=voln_f,
                )
            )

        for series_id in series_ids:
            if len(pending) >= per_sport:
                break
            off_ev = 0
            while len(pending) < per_sport and off_ev < max_offset:
                evs = fetch_closed_game_events_page(session, series_id, offset=off_ev, limit=80)
                if not evs:
                    break
                off_ev += len(evs)
                for ev in evs:
                    title = (ev.get("title") or "").strip()
                    for m in ev.get("markets") or []:
                        if isinstance(m, dict):
                            try_select_market(m, title or None)
                            if len(pending) >= per_sport:
                                break
                    if len(pending) >= per_sport:
                        break

        offset = 0
        while len(pending) < per_sport and offset < max_offset:
            batch = fetch_closed_markets_page(session, tag_id, offset=offset, limit=page_limit)
            if not batch:
                break
            offset += len(batch)
            for m in batch:
                try_select_market(m)
                if len(pending) >= per_sport:
                    break

        n_kept = len(pending)
        if n_kept < per_sport:
            LOG.warning(
                "%s: only %d / %d head-to-head markets (tag=%s series=%s); raise --max-scan-offset?",
                sport,
                n_kept,
                per_sport,
                tag_id,
                ",".join(series_ids),
            )
        else:
            LOG.info("%s: kept %d head-to-head markets (tag_id=%s)", sport, n_kept, tag_id)

        for sm in pending:
            outcomes = parse_gamma_json_field(sm.market.get("outcomes"))
            pol = primary_outcome_label(outcomes)
            hist = merge_clob_history(
                session,
                sm.yes_tok,
                sm.end_dt,
                limiter,
                lookback_hours=fetch_lb,
                intervals=("1h", "1m"),
            )
            feats = compute_features_for_market(hist, sm.end_dt, sm.start_dt, sm.voln_f)
            meta = build_metadata_row(sm.sport, sm.market)
            if not (meta.get("question") or "").strip():
                meta["question"] = sm.question

            hourly_det = fetch_hourly_clob_before_gamma_end(
                session, sm.yes_tok, sm.end_dt, skew_lookback_hours, limiter
            )
            win_s, win_e, det_note = detect_skew_segment_bounds(
                hourly_det,
                gamma_end=sm.end_dt,
                segment_hours=raw_segment_hours,
                skew_high=skew_high,
                skew_low=skew_low,
            )
            meta["skew_segment_status"] = det_note
            if win_s is not None and win_e is not None:
                meta["inferred_game_start_iso"] = pd.Timestamp(win_s).tz_convert("UTC").isoformat()
                meta["inferred_game_end_iso"] = pd.Timestamp(win_e).tz_convert("UTC").isoformat()
            else:
                meta["inferred_game_start_iso"] = ""
                meta["inferred_game_end_iso"] = ""
            meta_rows.append(meta)

            feat_row: dict[str, Any] = {
                "market_id": sm.mid,
                "sport": sm.sport,
                "resolved_yes": sm.resolved_primary,
                **EMPTY_FEATURE_ROW.copy(),
            }
            if feats is not None:
                feat_row.update(feats)
            feat_rows.append(feat_row)

            list_start = pd.Timestamp(sm.start_dt).tz_convert("UTC")
            list_end = pd.Timestamp(sm.end_dt).tz_convert("UTC")
            win_s_utc: Optional[pd.Timestamp] = None
            win_e_utc: Optional[pd.Timestamp] = None
            if win_s is None or win_e is None:
                hist_game = pd.DataFrame()
                clob_note = "no_1m_segment"
            else:
                win_s_utc = pd.Timestamp(win_s).tz_convert("UTC")
                win_e_utc = pd.Timestamp(win_e).tz_convert("UTC")
                hist_game = fetch_1m_clob_segment_only(session, sm.yes_tok, win_s_utc, win_e_utc, limiter)
                hist_game = hist_game.loc[
                    (hist_game["timestamp"] >= win_s_utc) & (hist_game["timestamp"] <= win_e_utc)
                ].copy()
                clob_note = "1m_after_skew_detect"
                if raw_minute_grid and not hist_game.empty:
                    hist_game = apply_minute_grid_ffill(hist_game, win_s_utc, win_e_utc)

            gctx = gamma_trade_context(sm.market)
            base_raw: dict[str, Any] = {
                "market_id": sm.mid,
                "sport": sm.sport,
                "question": sm.question,
                "primary_outcome": pol,
                "gamma_listing_start_iso": list_start.isoformat(),
                "gamma_listing_end_iso": list_end.isoformat(),
                "inferred_game_start_iso": meta.get("inferred_game_start_iso") or "",
                "inferred_game_end_iso": meta.get("inferred_game_end_iso") or "",
                "skew_segment_status": det_note,
                "clob_sources": clob_note,
                "minute_grid_ffill": bool(raw_minute_grid),
                **gctx,
            }

            if hist_game.empty:
                raw_rows.append(
                    {
                        **base_raw,
                        "timestamp_iso": "",
                        "unix_ts": "",
                        "price_primary_token": float("nan"),
                        "seconds_since_window_start": float("nan"),
                        "seconds_until_window_end": float("nan"),
                        "note": det_note if win_s_utc is None else "no_1m_points_in_segment",
                    }
                )
            else:
                for _, r in hist_game.iterrows():
                    ts = pd.Timestamp(r["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    sec_since = (ts - win_s_utc).total_seconds()
                    sec_until = (win_e_utc - ts).total_seconds()
                    raw_rows.append(
                        {
                            **base_raw,
                            "timestamp_iso": ts.isoformat(),
                            "unix_ts": int(ts.timestamp()),
                            "price_primary_token": float(r["price_yes"]),
                            "seconds_since_window_start": float(sec_since),
                            "seconds_until_window_end": float(sec_until),
                            "note": "",
                        }
                    )

    meta_df = pd.DataFrame(meta_rows)
    feat_df = pd.DataFrame(feat_rows)
    raw_df = pd.DataFrame(raw_rows)
    return feat_df, meta_df, raw_df


def zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = list(FEATURE_NUMERIC_KEYS)
    out = df.copy()
    if len(out) < 2:
        LOG.warning(
            "Only %d row(s): skipping StandardScaler (undefined variance). Writing raw numeric features.",
            len(out),
        )
        return out
    X = out[numeric_cols].astype(float)
    X_imp = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    out[numeric_cols] = scaler.fit_transform(X_imp)
    return out


def atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    """Write CSV via a temp file + os.replace so a crash mid-write does not truncate outputs."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:  python %(prog)s 20\n          python %(prog)s --markets-per-sport 15",
    )
    p.add_argument(
        "markets_per_sport_pos",
        nargs="?",
        type=int,
        default=None,
        metavar="N",
        help=f"How many head-to-head markets to keep per sport (positional). Default: {DEFAULT_PER_SPORT}.",
    )
    p.add_argument(
        "--markets-per-sport",
        type=int,
        default=None,
        dest="markets_per_sport_opt",
        help=f"Same as positional N (markets per sport). Default: {DEFAULT_PER_SPORT}.",
    )
    p.add_argument(
        "--per-sport",
        type=int,
        default=None,
        dest="per_sport_legacy",
        help="Deprecated alias for --markets-per-sport.",
    )
    p.add_argument("--features-out", default=DEFAULT_FEATURES, help="Scaled feature matrix CSV path.")
    p.add_argument("--metadata-out", default=DEFAULT_METADATA, help="Metadata sidecar CSV path.")
    p.add_argument(
        "--max-scan-offset",
        type=int,
        default=1200,
        help="Max Gamma offset per sport (markets + events pagination; default 1200).",
    )
    p.add_argument(
        "--raw-timeseries-out",
        default=DEFAULT_RAW_TIMESERIES,
        help="Long-format CLOB prices CSV (one row per timestamp per market).",
    )
    p.add_argument(
        "--market-end-within-last-days",
        type=float,
        default=float(DEFAULT_MARKET_END_WITHIN_LAST_DAYS),
        help="Only include markets whose Gamma endDate is within this many days of now (UTC). "
        "Use 0 to disable (default: 30).",
    )
    p.add_argument(
        "--market-end-after",
        type=str,
        default=None,
        metavar="ISO",
        help="Optional: require endDate >= this instant (ISO-8601, e.g. 2026-04-01).",
    )
    p.add_argument(
        "--market-end-before",
        type=str,
        default=None,
        metavar="ISO",
        help="Optional: require endDate < this instant (e.g. 2026-05-01 for April 2026 only).",
    )
    p.add_argument(
        "--skew-lookback-hours",
        type=float,
        default=DEFAULT_SKEW_LOOKBACK_HOURS,
        help="Hourly CLOB window before Gamma endDate used to find the resolution skew streak (default 336).",
    )
    p.add_argument(
        "--skew-high",
        type=float,
        default=DEFAULT_SKEW_HIGH,
        help="Primary-token price >= this (or <= --skew-low) counts as skewed for streak detection.",
    )
    p.add_argument(
        "--skew-low",
        type=float,
        default=DEFAULT_SKEW_LOW,
        help="Lower skew threshold on primary-token implied probability.",
    )
    p.add_argument(
        "--raw-segment-hours",
        type=float,
        default=DEFAULT_RAW_SEGMENT_HOURS,
        help="After inferred game end, pull 1m CLOB for this many hours backward (default 2).",
    )
    p.add_argument(
        "--raw-minute-grid",
        action="store_true",
        help="Forward-fill to a 1-minute grid between first and last in-window CLOB print (analysis-friendly).",
    )
    p.add_argument(
        "--raw-fetch-lookback-hours",
        type=float,
        default=240.0,
        help="CLOB lookback for PCA feature merge only (default 240h, min 10d).",
    )
    return p.parse_args()


def parse_optional_iso_utc(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None or not str(s).strip():
        return None
    t = pd.to_datetime(str(s).strip(), utc=True, errors="coerce")
    if pd.isna(t):
        raise SystemExit(f"Invalid ISO date/time: {s!r}")
    return pd.Timestamp(t).tz_convert("UTC")


def _markets_per_sport_from_args(args: argparse.Namespace) -> int:
    n = args.per_sport_legacy
    if n is None:
        n = args.markets_per_sport_opt
    if n is None:
        n = args.markets_per_sport_pos
    if n is None:
        return DEFAULT_PER_SPORT
    if n < 1:
        raise SystemExit("markets per sport must be >= 1")
    return n


def main() -> int:
    args = parse_args()
    n_per = _markets_per_sport_from_args(args)
    session = http_session()
    limiter = SlidingWindowRateLimiter(CLOB_MAX_REQUESTS, CLOB_WINDOW_SEC)

    feat_df, meta_df, raw_df = extract_feature_matrix(
        session,
        limiter,
        n_per,
        max_offset_cap=args.max_scan_offset,
        fetch_lookback_hours=args.raw_fetch_lookback_hours,
        raw_minute_grid=args.raw_minute_grid,
        market_end_within_last_days=args.market_end_within_last_days,
        market_end_after=parse_optional_iso_utc(args.market_end_after),
        market_end_before=parse_optional_iso_utc(args.market_end_before),
        skew_lookback_hours=args.skew_lookback_hours,
        skew_high=args.skew_high,
        skew_low=args.skew_low,
        raw_segment_hours=args.raw_segment_hours,
    )
    if meta_df.empty:
        LOG.error("No head-to-head closed markets matched filters. Nothing to write.")
        return 1

    if len(feat_df) != len(meta_df):
        LOG.error("Internal error: feature rows (%d) != metadata rows (%d)", len(feat_df), len(meta_df))
        return 1

    scaled = zscore_features(feat_df)
    atomic_write_csv(args.features_out, scaled)
    atomic_write_csv(args.metadata_out, meta_df)
    atomic_write_csv(args.raw_timeseries_out, raw_df)
    LOG.info(
        "Wrote %d rows -> %s (features); %d -> %s (metadata); %d -> %s (raw timeseries).",
        len(scaled),
        args.features_out,
        len(meta_df),
        args.metadata_out,
        len(raw_df),
        args.raw_timeseries_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
