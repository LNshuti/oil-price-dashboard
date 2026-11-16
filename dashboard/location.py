"""
Markets EIA measures, and mapping a ZIP code to the nearest one.

EIA prices fuel for the nation, PADD regions, nine states and ten metro
areas; nothing finer. A ZIP code is resolved to the most local of those
that the selected fuel actually covers, and the result says so plainly.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ZIP_FILE = (Path(__file__).resolve().parent.parent
            / "data" / "reference" / "us_zip_codes.csv.gz")
CITY_RADIUS_KM = 60


@dataclass(frozen=True)
class Market:
    code: str
    name: str
    kind: str  # national, region, subregion, state, city
    parent: str | None = None
    lat: float | None = None
    lon: float | None = None


_MARKETS = [
    Market("NUS", "U.S. average", "national"),
    Market("R10", "East Coast (PADD 1)", "region", "NUS"),
    Market("R1X", "New England (PADD 1A)", "subregion", "R10"),
    Market("R1Y", "Central Atlantic (PADD 1B)", "subregion", "R10"),
    Market("R1Z", "Lower Atlantic (PADD 1C)", "subregion", "R10"),
    Market("R20", "Midwest (PADD 2)", "region", "NUS"),
    Market("R30", "Gulf Coast (PADD 3)", "region", "NUS"),
    Market("R40", "Rocky Mountain (PADD 4)", "region", "NUS"),
    Market("R50", "West Coast (PADD 5)", "region", "NUS"),
    Market("R5XCA", "West Coast except California", "subregion", "R50"),
    Market("SCA", "California", "state", "R50"),
    Market("SCO", "Colorado", "state", "R40"),
    Market("SFL", "Florida", "state", "R1Z"),
    Market("SMA", "Massachusetts", "state", "R1X"),
    Market("SMN", "Minnesota", "state", "R20"),
    Market("SNY", "New York", "state", "R1Y"),
    Market("SOH", "Ohio", "state", "R20"),
    Market("STX", "Texas", "state", "R30"),
    Market("SWA", "Washington", "state", "R5XCA"),
    Market("YBOS", "Boston", "city", "SMA", 42.3601, -71.0589),
    Market("YORD", "Chicago", "city", "R20", 41.8781, -87.6298),
    Market("YCLE", "Cleveland", "city", "SOH", 41.4993, -81.6944),
    Market("YDEN", "Denver", "city", "SCO", 39.7392, -104.9903),
    Market("Y44HO", "Houston", "city", "STX", 29.7604, -95.3698),
    Market("Y05LA", "Los Angeles", "city", "SCA", 34.0522, -118.2437),
    Market("YMIA", "Miami", "city", "SFL", 25.7617, -80.1918),
    Market("Y35NY", "New York City", "city", "SNY", 40.7128, -74.0060),
    Market("Y05SF", "San Francisco", "city", "SCA", 37.7749, -122.4194),
    Market("Y48SE", "Seattle", "city", "SWA", 47.6062, -122.3321),
]
MARKETS = {m.code: m for m in _MARKETS}

# EIA's PADD region (and sub-region, where one exists) for each state.
_PADD = {
    "R1X": "CT ME MA NH RI VT",
    "R1Y": "DE DC MD NJ NY PA",
    "R1Z": "FL GA NC SC VA WV",
    "R20": "IL IN IA KS KY MI MN MO NE ND SD OH OK TN WI",
    "R30": "AL AR LA MS NM TX",
    "R40": "CO ID MT UT WY",
    "R5XCA": "AK AZ HI NV OR WA",
    "R50": "CA",
}
STATE_REGION = {s: code for code, states in _PADD.items()
                for s in states.split()}
STATE_MARKET = {"CA": "SCA", "CO": "SCO", "FL": "SFL", "MA": "SMA",
                "MN": "SMN", "NY": "SNY", "OH": "SOH", "TX": "STX",
                "WA": "SWA"}


@dataclass(frozen=True)
class Resolution:
    market: str
    message: str
    found: bool = True


def chain(code: str) -> list[str]:
    """A market followed by its broader parents, ending at U.S."""
    codes = []
    while code is not None:
        codes.append(code)
        code = MARKETS[code].parent
    return codes


def fallback(code: str, available: set[str]) -> str:
    """Most local market in ``code``'s chain that has data."""
    for candidate in chain(code):
        if candidate in available:
            return candidate
    return "NUS"


def market_choices(available: set[str]) -> list[tuple[str, str]]:
    """(label, code) pairs, broadest first, for a dropdown."""
    return [(m.name, m.code) for m in _MARKETS if m.code in available]


@lru_cache(maxsize=1)
def _zip_table() -> pd.DataFrame:
    return pd.read_csv(ZIP_FILE, dtype={"zip": str}).set_index("zip")


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(a))


def county_label(county: str, state: str) -> str:
    if not isinstance(county, str) or not county:
        return ""
    if state == "LA":
        return f"{county} Parish"
    if state in ("AK", "DC") or "city" in county.lower():
        return county
    return f"{county} County"


def _nearest_city(lat: float, lon: float) -> str | None:
    best, best_km = None, CITY_RADIUS_KM
    for m in _MARKETS:
        if m.kind == "city":
            km = _distance_km(lat, lon, m.lat, m.lon)
            if km <= best_km:
                best, best_km = m.code, km
    return best


def local_market(state: str, lat: float, lon: float) -> str:
    """Most local market that covers a point, ignoring data coverage."""
    city = _nearest_city(lat, lon)
    if city is not None:
        return city
    if state in STATE_MARKET:
        return STATE_MARKET[state]
    return STATE_REGION.get(state, "NUS")


def resolve_zip(zip_code: str, available: set[str]) -> Resolution:
    """Map a ZIP code to the nearest market priced for this fuel."""
    zip_code = (zip_code or "").strip()[:5]
    if len(zip_code) != 5 or not zip_code.isdigit():
        return Resolution("NUS", "Enter a 5-digit ZIP code.", False)
    table = _zip_table()
    if zip_code not in table.index:
        return Resolution("NUS", f"ZIP {zip_code} wasn't found.", False)
    row = table.loc[zip_code]
    state = row["state"]
    ideal = local_market(state, row["lat"], row["lon"])
    market = fallback(ideal, available)

    county = county_label(row["county"], state)
    place = ", ".join(p for p in (row["place"], county, state) if p)
    name = MARKETS[market].name
    if market == ideal and MARKETS[market].kind == "city":
        note = "EIA measures this metro area directly."
    elif market == ideal and MARKETS[market].kind == "state":
        note = "EIA measures this state directly."
    elif market == "NUS":
        note = "EIA doesn't price this area separately."
    else:
        note = ("Nearest market EIA measures for this fuel; "
                "local prices can differ by taxes and competition.")
    return Resolution(market, f"**{zip_code}** · {place} → **{name}**. "
                              f"{note}")
