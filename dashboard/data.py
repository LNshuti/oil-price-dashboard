"""
EIA weekly retail fuel prices: download, cache and aggregation.

EIA publishes every measured market for a fuel in one spreadsheet, so a
refresh is two keyless downloads rather than dozens of rate-limited API
calls. Prices are cached in long format (date, fuel, area, price).
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "data" / "processed" / "eia_weekly_retail.csv.gz"
MAX_AGE_SECONDS = 24 * 60 * 60
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Fuel:
    key: str
    label: str
    url: str
    sheet: str


FUELS = {
    "diesel": Fuel(
        "diesel",
        "Diesel (No. 2)",
        "https://www.eia.gov/petroleum/gasdiesel/xls/psw18vwall.xls",
        "Data 1",
    ),
    "gasoline": Fuel(
        "gasoline",
        "Regular gasoline",
        "https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls",
        "Data 3",
    ),
}


@dataclass(frozen=True)
class DataStatus:
    ok: bool
    as_of: pd.Timestamp | None
    checked_at: datetime | None
    message: str


_lock = threading.Lock()
_last_error: str | None = None


def parse_eia_sheet(content: bytes, fuel: Fuel) -> pd.DataFrame:
    """Parse one EIA weekly spreadsheet into long format.

    Row 2 holds series keys such as ``EMD_EPD2D_PTE_R30_DPG``; the fourth
    token is the EIA area code. Data starts on row 4.
    """
    raw = pd.read_excel(BytesIO(content), sheet_name=fuel.sheet, header=None)
    keys = raw.iloc[1, 1:]
    areas = [str(k).split("_")[3] for k in keys]
    body = raw.iloc[3:].copy()
    body.columns = ["date"] + areas
    long = body.melt(id_vars="date", var_name="area", value_name="price")
    long["date"] = pd.to_datetime(long["date"])
    long["price"] = pd.to_numeric(long["price"], errors="coerce")
    long = long.dropna(subset=["price"])
    long.insert(1, "fuel", fuel.key)
    return long.sort_values(["area", "date"]).reset_index(drop=True)


def download_all() -> pd.DataFrame:
    """Download and parse every fuel. Raises on any failure."""
    frames = []
    for fuel in FUELS.values():
        response = requests.get(fuel.url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        frames.append(parse_eia_sheet(response.content, fuel))
    prices = pd.concat(frames, ignore_index=True)
    if prices.empty:
        raise ValueError("EIA returned no prices")
    return prices


def _cache_age() -> float | None:
    if not CACHE_FILE.exists():
        return None
    return time.time() - CACHE_FILE.stat().st_mtime


def refresh(force: bool = False) -> DataStatus:
    """Fetch new data when the cache is stale (or ``force``).

    A failed fetch keeps the cached data and is reported, never hidden.
    """
    global _last_error
    with _lock:
        age = _cache_age()
        if force or age is None or age > MAX_AGE_SECONDS:
            try:
                prices = download_all()
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                prices.to_csv(CACHE_FILE, index=False)
                load_prices.cache_clear()
                _last_error = None
            except Exception as exc:  # network, HTTP or parse failure
                logger.warning("EIA refresh failed: %s", exc)
                _last_error = str(exc) or type(exc).__name__
    return status()


def status() -> DataStatus:
    if not CACHE_FILE.exists():
        return DataStatus(False, None, None, _last_error or "No data yet")
    as_of = load_prices()["date"].max()
    checked = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
    return DataStatus(_last_error is None, as_of, checked, _last_error or "")


def refresh_in_background_if_stale() -> None:
    age = _cache_age()
    if age is None or age > MAX_AGE_SECONDS:
        if not _lock.locked():
            threading.Thread(target=refresh, daemon=True).start()


def _memoize(fn):
    """Single-slot cache that ``refresh`` can clear."""
    box = {}

    def wrapper():
        if "value" not in box:
            box["value"] = fn()
        return box["value"]

    wrapper.cache_clear = box.clear
    return wrapper


@_memoize
def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    return prices


def weekly(fuel: str, area: str) -> pd.Series:
    """Weekly price series for one fuel and market, indexed by date."""
    prices = load_prices()
    rows = prices[(prices["fuel"] == fuel) & (prices["area"] == area)]
    return rows.set_index("date")["price"].sort_index()


def areas_for(fuel: str) -> list[str]:
    prices = load_prices()
    return sorted(prices.loc[prices["fuel"] == fuel, "area"].unique())


def monthly(series: pd.Series) -> pd.Series:
    """Monthly mean of a weekly series, keeping complete months only.

    The latest month is dropped while it is still in progress, so a
    two-week partial average never trains the models or skews backtests.
    """
    if series.empty:
        return series
    means = series.groupby(series.index.to_period("M")).mean()
    last = series.index.max()
    next_obs = last + pd.Timedelta(days=7)
    if next_obs.to_period("M") == last.to_period("M"):
        means = means.iloc[:-1]
    means.index = means.index.to_timestamp()
    return means


def latest_snapshot(fuel: str) -> pd.DataFrame:
    """Latest price per market with 4-week and 52-week changes."""
    prices = load_prices()
    rows = prices[prices["fuel"] == fuel]
    table = rows.pivot(index="date", columns="area", values="price")
    table = table.sort_index()
    latest = table.iloc[-1]
    snapshot = pd.DataFrame({
        "price": latest,
        "change_4w": latest - table.iloc[-5],
        "change_52w": latest - table.iloc[-53],
    })
    return snapshot.dropna(subset=["price"])
