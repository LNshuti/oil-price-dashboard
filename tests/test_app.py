import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app
from dashboard import charts, data, forecast, location, views


@pytest.fixture(scope="module")
def outlook():
    """Outlook on a synthetic seasonal series (fast and deterministic)."""
    months = pd.date_range("2010-01-01", periods=180, freq="MS")
    rng = np.random.default_rng(0)
    prices = (3 + 0.3 * np.sin(np.arange(180) * 2 * np.pi / 12)
              + rng.normal(0, 0.05, 180))
    return forecast.build_outlook(pd.Series(prices, index=months), 6)


# Forecasting ---------------------------------------------------------------

def test_forecast_starts_after_last_complete_month(outlook):
    """P0: the outlook must look forward, not re-predict history."""
    first = outlook.forecast["ds"].iloc[0]
    assert first == outlook.history.index[-1] + pd.offsets.MonthBegin(1)
    assert len(outlook.forecast) == 6


def test_backtest_covers_last_horizon_months(outlook):
    assert list(outlook.backtest["ds"]) == list(outlook.history.index[-6:])


def test_every_model_has_both_intervals(outlook):
    for model in forecast.MODELS:
        for level in forecast.LEVELS:
            lo, hi = outlook.band(model, level)
            assert (lo <= outlook.forecast[model]).all()
            assert (outlook.forecast[model] <= hi).all()


def test_recommended_model_has_lowest_error(outlook):
    best = outlook.metrics.sort_values("MAPE").iloc[0]["Model"]
    assert outlook.recommended == best


def test_horizon_must_be_offered_choice():
    with pytest.raises(ValueError):
        forecast.build_outlook(pd.Series(dtype=float), 52)


# Data ----------------------------------------------------------------------

def test_monthly_drops_month_in_progress():
    weeks = pd.date_range("2026-09-07", "2026-11-09", freq="7D")
    monthly = data.monthly(pd.Series(1.0, index=weeks))
    assert monthly.index[-1] == pd.Timestamp("2026-10-01")


def test_monthly_keeps_month_ending_with_last_week():
    weeks = pd.date_range("2026-09-07", "2026-10-26", freq="7D")
    monthly = data.monthly(pd.Series(1.0, index=weeks))
    assert monthly.index[-1] == pd.Timestamp("2026-10-01")


def test_failed_refresh_is_reported_and_keeps_cache(monkeypatch, tmp_path):
    cache = tmp_path / "prices.csv.gz"
    pd.DataFrame({"date": ["2026-11-09"], "fuel": ["diesel"],
                  "area": ["NUS"], "price": [6.1]}).to_csv(cache, index=False)
    monkeypatch.setattr(data, "CACHE_FILE", cache)
    data.load_prices.cache_clear()

    def boom():
        raise ConnectionError("rate limited")
    monkeypatch.setattr(data, "download_all", boom)

    status = data.refresh(force=True)
    assert not status.ok
    assert "rate limited" in status.message
    assert status.as_of == pd.Timestamp("2026-11-09")
    data._last_error = None
    data.load_prices.cache_clear()


def test_successful_refresh_is_persisted(monkeypatch, tmp_path):
    cache = tmp_path / "prices.csv.gz"
    monkeypatch.setattr(data, "CACHE_FILE", cache)
    fresh = pd.DataFrame({"date": pd.to_datetime(["2026-11-16"]),
                          "fuel": ["diesel"], "area": ["NUS"],
                          "price": [6.0]})
    monkeypatch.setattr(data, "download_all", lambda: fresh)

    status = data.refresh(force=True)
    assert status.ok
    assert status.as_of == pd.Timestamp("2026-11-16")
    assert cache.exists()
    data.load_prices.cache_clear()


def test_parse_eia_sheet_reads_area_codes(tmp_path):
    rows = [
        ["Back to Contents", "Data 1", None],
        ["Sourcekey", "EMD_EPD2D_PTE_NUS_DPG", "EMD_EPD2D_PTE_R30_DPG"],
        ["Date", "U.S.", "Gulf Coast"],
        [pd.Timestamp("2026-11-02"), 6.29, 5.91],
        [pd.Timestamp("2026-11-09"), 6.13, None],
    ]
    path = tmp_path / "sheet.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name="Data 1", header=False,
                                index=False)
    fuel = data.Fuel("diesel", "Diesel", "", "Data 1")
    parsed = data.parse_eia_sheet(path.read_bytes(), fuel)
    assert set(parsed["area"]) == {"NUS", "R30"}
    assert len(parsed) == 3  # the missing Gulf Coast week is dropped


def test_bundled_cache_covers_both_fuels():
    prices = data.load_prices()
    assert {"diesel", "gasoline"} <= set(prices["fuel"])
    assert set(prices["area"]) <= set(location.MARKETS)


# Location ------------------------------------------------------------------

@pytest.mark.parametrize("zip_code, fuel, expected", [
    ("77002", "gasoline", "Y44HO"),   # Houston metro is measured
    ("77002", "diesel", "R30"),       # diesel only down to PADD
    ("14201", "gasoline", "SNY"),     # Buffalo: state, not NYC
    ("70115", "gasoline", "R30"),     # Louisiana -> Gulf Coast
    ("97201", "diesel", "R5XCA"),     # Oregon -> West Coast ex. CA
    ("97201", "gasoline", "R50"),     # no ex-CA series for gasoline
])
def test_zip_resolves_to_most_local_measured_market(zip_code, fuel,
                                                    expected):
    available = set(data.areas_for(fuel))
    assert location.resolve_zip(zip_code, available).market == expected


def test_parishes_are_named_correctly():
    result = location.resolve_zip("70115", set(data.areas_for("diesel")))
    assert "Orleans Parish" in result.message


@pytest.mark.parametrize("bad", ["", "123", "abcde", "00000"])
def test_bad_zip_is_reported(bad):
    result = location.resolve_zip(bad, {"NUS"})
    assert not result.found


def test_fallback_walks_up_to_a_measured_market():
    assert location.fallback("Y44HO", {"NUS", "R30"}) == "R30"
    assert location.fallback("Y44HO", {"NUS"}) == "NUS"


# Views ---------------------------------------------------------------------

def test_delta_uses_words_not_only_color():
    html = views.delta(0.12, "4 weeks ago")
    assert "▲ 12¢" in html and 'aria-label="up 12¢"' in html
    assert "No change" in views.delta(0.001, "x")


def test_budget_totals_forecast(outlook):
    b = views.budget(outlook, gallons=1000, today=3.0)
    expected = outlook.forecast[outlook.recommended].sum() * 1000
    assert b.expected == pytest.approx(expected)
    assert b.low < b.expected < b.high
    assert b.at_today == pytest.approx(3.0 * 6 * 1000)


def test_tables_show_cents_not_four_decimals(outlook):
    amounts = re.findall(r"\$\d+\.(\d+)", views.forecast_table(outlook))
    assert amounts and all(len(cents) == 2 for cents in amounts)


def test_charts_build(outlook):
    assert charts.outlook_figure(outlook).data
    assert charts.backtest_figure(outlook).data
    assert charts.accuracy_figure(outlook).data


# App -----------------------------------------------------------------------

def test_render_returns_every_view_and_unique_downloads():
    first = app.render("diesel", "NUS", "3", 1000)
    second = app.render("diesel", "NUS", "3", 1000)
    assert len(first) == app.VIEW_COUNT
    forecast_file, history_file = first[3], first[12]
    assert Path(forecast_file).exists() and Path(history_file).exists()
    assert forecast_file != second[3]  # no shared /tmp path between users
    assert "Nov" in first[-1] or "through" in first[-1]


def test_partial_zip_does_not_render():
    result = app.on_zip("770", "diesel", "NUS", "3", 1000)
    assert len(result) == 2 + app.VIEW_COUNT


def test_switching_fuel_keeps_zip_location():
    market, note, *_ = app.on_fuel("77002", "gasoline", "R30", "3", 1000)
    assert market["value"] == "Y44HO"
    assert "Houston" in note
