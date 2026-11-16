"""
Fleet Fuel Price Outlook

Weekly EIA retail diesel and gasoline prices for every market EIA
measures, a ZIP-code lookup to the nearest one, and monthly forecasts
sized to a fleet's fuel budget.
"""

import logging
import tempfile
import threading
from pathlib import Path

import gradio as gr
import pandas as pd

from dashboard import charts, data, location, views
from dashboard.forecast import HORIZONS, cached_outlook

logging.basicConfig(level=logging.INFO)

DEFAULT_FUEL = "diesel"
DEFAULT_MARKET = "NUS"
DEFAULT_HORIZON = 6
DEFAULT_GALLONS = 10_000
NAMES = {code: m.name for code, m in location.MARKETS.items()}

# Quiet field labels: plain muted text, not colored chips, so the
# numbers below are the loudest thing on the page.
THEME = gr.themes.Soft(
    primary_hue="blue", neutral_hue="stone",
    font=[gr.themes.Font(f) for f in ("system-ui", "-apple-system",
                                      "Segoe UI", "sans-serif")],
).set(
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    block_title_text_color="*neutral_500",
    block_title_text_color_dark="*neutral_400",
    block_title_padding="0",
    block_title_text_weight="500",
)

CSS = views.STYLES + """
.fo-header h1 { font-size: 28px; font-weight: 600; margin: 0; }
.fo-header p { margin: 4px 0 0; color: var(--body-text-color-subdued); }
.fo-status { font-size: 13px; color: var(--body-text-color-subdued); }
"""


def _available(fuel: str) -> set[str]:
    return set(data.areas_for(fuel))


def _market_update(fuel: str, value: str):
    return gr.update(choices=location.market_choices(_available(fuel)),
                     value=value)


def status_text() -> str:
    s = data.status()
    if s.as_of is None:
        return f"No price data available. {s.message}"
    as_of = f"{s.as_of:%b} {s.as_of.day}, {s.as_of.year}"
    text = (f"EIA weekly retail prices through **{as_of}** · "
            f"last downloaded {s.checked_at:%b} {s.checked_at.day}, "
            f"{s.checked_at:%H:%M}. "
            "EIA publishes new prices each Monday evening.")
    if not s.ok:
        text += (f"  \n⚠️ Couldn't reach EIA ({s.message}); "
                 "showing cached data.")
    return text


def locate(zip_code: str, fuel: str, market: str) -> tuple[str, str]:
    """Market and location note for a ZIP; a bad ZIP keeps ``market``."""
    result = location.resolve_zip(zip_code, _available(fuel))
    return (result.market if result.found else market), result.message


def pick_market(fuel: str, zip_code: str, market: str) -> tuple[str, str]:
    """Keep the location; fall back to a broader market if needed."""
    if (zip_code or "").strip():
        return locate(zip_code, fuel, market)
    chosen = location.fallback(market, _available(fuel))
    note = ""
    if chosen != market:
        note = (f"{NAMES[market]} isn't priced for "
                f"{data.FUELS[fuel].label.lower()}; showing "
                f"**{NAMES[chosen]}**.")
    return chosen, note


def _export(df: pd.DataFrame, stem: str) -> str:
    """Write a CSV to a per-request folder so users never collide."""
    folder = Path(tempfile.mkdtemp(prefix="fuel-outlook-"))
    path = folder / f"{stem}.csv"
    df.to_csv(path, index=False)
    return str(path)


def render(fuel: str, market: str, horizon: str, gallons):
    data.refresh_in_background_if_stale()
    horizon = int(horizon)
    market = market or DEFAULT_MARKET
    fuel_label = data.FUELS[fuel].label
    name = NAMES[market]
    weekly = data.weekly(fuel, market)
    monthly = data.monthly(weekly)
    as_of = weekly.index.max()
    outlook = cached_outlook((fuel, market, horizon, as_of), monthly,
                             horizon)

    headline = views.headline(market=name, fuel=fuel_label, latest=weekly,
                              outlook=outlook, gallons=gallons or 0)
    snapshot = data.latest_snapshot(fuel)
    slug = f"{fuel}-{name.lower().split(' (')[0].replace(' ', '-')}"

    model = outlook.recommended
    lo80, hi80 = outlook.band(model, 80)
    lo95, hi95 = outlook.band(model, 95)
    forecast_csv = pd.DataFrame({
        "month": outlook.forecast["ds"].dt.strftime("%Y-%m"),
        "forecast_usd_per_gal": outlook.forecast[model].round(3),
        "low_80": lo80.round(3), "high_80": hi80.round(3),
        "low_95": lo95.round(3), "high_95": hi95.round(3),
        "model": model, "market": name, "fuel": fuel_label,
    })
    history = weekly.rename("usd_per_gal").reset_index()
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    preview = history.iloc[::-1].rename(
        columns={"date": "Week of", "usd_per_gal": "Price ($/gal)"})

    return (
        headline,
        charts.outlook_figure(outlook),
        views.forecast_table(outlook),
        _export(forecast_csv, f"{slug}-forecast"),
        charts.markets_figure(snapshot, NAMES, market),
        views.markets_table(snapshot, NAMES, market),
        charts.backtest_figure(outlook),
        charts.accuracy_figure(outlook),
        views.metrics_table(outlook),
        views.all_models_table(outlook),
        charts.history_figure(weekly, name),
        preview,
        _export(history, f"{slug}-weekly-history"),
        status_text(),
    )


VIEW_COUNT = 14  # number of values ``render`` returns


def on_zip(zip_code, fuel, market, horizon, gallons):
    """Resolve as soon as five characters are typed; no Enter needed."""
    typed = (zip_code or "").strip()
    if len(typed) < 5:
        note = "" if not typed else gr.skip()
        return (gr.skip(), note) + (gr.skip(),) * VIEW_COUNT
    chosen, note = locate(typed, fuel, market)
    return (_market_update(fuel, chosen), note,
            *render(fuel, chosen, horizon, gallons))


def on_fuel(zip_code, fuel, market, horizon, gallons):
    chosen, note = pick_market(fuel, zip_code, market)
    return (_market_update(fuel, chosen), note,
            *render(fuel, chosen, horizon, gallons))


def on_market(fuel, market, horizon, gallons):
    return ("", *render(fuel, market, horizon, gallons))


def on_refresh(fuel, market, horizon, gallons):
    data.refresh(force=True)
    return render(fuel, market, horizon, gallons)


ABOUT = """
### What this shows
Retail **diesel (No. 2)** and **regular gasoline** prices from the U.S.
Energy Information Administration's weekly survey, for every market EIA
measures: the U.S., PADD regions and sub-regions, nine states and ten
metro areas. Forecasts are monthly averages.

### How the forecast works
Four models are fit to the last 20 years of complete months:
AutoARIMA and AutoETS (StatsForecast), LightGBM and XGBoost
(MLForecast, with lag and rolling-mean features). Each model is first
tested on the most recent months it wasn't trained on, the same number
of months as the forecast horizon. The model with the lowest average
error is recommended, then every model is refit on all the data to
forecast forward. Ranges are 80% and 95% prediction intervals
(conformal intervals for the ML models).

### Budget
The budget multiplies each month's forecast by your monthly gallons.
Its range totals the monthly 80% bounds, which assumes prices stay near
one edge all period, so it is deliberately conservative.

### Limits
EIA does not publish county or ZIP-level prices. A ZIP code maps to the
most local market EIA measures for the selected fuel; actual pump prices
nearby differ with local taxes, station competition and brand.

### Sources
[EIA weekly retail gasoline and diesel prices][eia]
· ZIP codes from [GeoNames](https://www.geonames.org/) (CC BY 4.0).

[eia]: https://www.eia.gov/petroleum/gasdiesel/
"""


def create_app() -> gr.Blocks:
    fuel_choices = [(f.label, f.key) for f in data.FUELS.values()]
    horizon_choices = [(f"{h} months", str(h)) for h in HORIZONS]

    with gr.Blocks(title="Fleet Fuel Price Outlook") as app:
        gr.HTML(
            '<div class="fo-header"><h1>Fleet fuel price outlook</h1>'
            "<p>Retail diesel and gasoline prices and forecasts for fuel "
            "budgeting, by U.S. market.</p></div>"
        )
        with gr.Row(equal_height=True):
            zip_code = gr.Textbox(label="ZIP code", placeholder="e.g. 70115",
                                  max_lines=1, max_length=5, scale=1)
            fuel = gr.Radio(fuel_choices, value=DEFAULT_FUEL, label="Fuel",
                            scale=1)
            market = gr.Dropdown(
                location.market_choices(_available(DEFAULT_FUEL)),
                value=DEFAULT_MARKET, label="Market", scale=2,
            )
            horizon = gr.Dropdown(horizon_choices,
                                  value=str(DEFAULT_HORIZON),
                                  label="Forecast horizon", scale=1)
            gallons = gr.Number(DEFAULT_GALLONS, label="Gallons per month",
                                minimum=0, precision=0, scale=1)
        place = gr.Markdown()
        headline = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Outlook"):
                outlook_plot = gr.Plot(show_label=False)
                forecast_table = gr.HTML()
                with gr.Row():
                    forecast_file = gr.DownloadButton(
                        "Download forecast (CSV)", variant="secondary",
                        size="sm", scale=0, min_width=220)
            with gr.Tab("Markets"):
                gr.Markdown("How the selected market compares with every "
                            "market EIA measures for this fuel.")
                markets_plot = gr.Plot(show_label=False)
                markets_table = gr.HTML()
            with gr.Tab("Model check"):
                gr.Markdown(
                    "Each model forecast the most recent months without "
                    "seeing them. Lower error is better; the best model "
                    "drives the outlook.")
                backtest_plot = gr.Plot(show_label=False)
                accuracy_plot = gr.Plot(show_label=False)
                metrics_table = gr.HTML()
                models_table = gr.HTML()
            with gr.Tab("Data"):
                history_plot = gr.Plot(show_label=False)
                history_table = gr.DataFrame(interactive=False,
                                             max_height=420)
                with gr.Row():
                    history_file = gr.DownloadButton(
                        "Download weekly history (CSV)",
                        variant="secondary", size="sm", scale=0,
                        min_width=240)
            with gr.Tab("About"):
                gr.Markdown(ABOUT)

        with gr.Row():
            status = gr.Markdown(elem_classes="fo-status")
            refresh_btn = gr.Button("Check for new data", size="sm",
                                    variant="secondary", scale=0,
                                    min_width=180)

        # Every handler returns all of its outputs in one step, so two
        # controls changing at once can't interleave and show stale views.
        inputs = [fuel, market, horizon, gallons]
        views_out = [headline, outlook_plot, forecast_table, forecast_file,
                     markets_plot, markets_table, backtest_plot,
                     accuracy_plot, metrics_table, models_table,
                     history_plot, history_table, history_file, status]
        located = [market, place] + views_out

        app.load(render, inputs, views_out)
        zip_code.change(on_zip, [zip_code] + inputs, located,
                        trigger_mode="always_last")
        fuel.change(on_fuel, [zip_code] + inputs, located)
        market.input(on_market, inputs, [place] + views_out)
        horizon.change(render, inputs, views_out)
        gr.on([gallons.submit, gallons.blur], render, inputs, views_out)
        refresh_btn.click(on_refresh, inputs, views_out)

    return app


def _warm() -> None:
    """Fetch fresh data, then precompute the default outlook."""
    data.refresh()
    weekly = data.weekly(DEFAULT_FUEL, DEFAULT_MARKET)
    cached_outlook(
        (DEFAULT_FUEL, DEFAULT_MARKET, DEFAULT_HORIZON, weekly.index.max()),
        data.monthly(weekly), DEFAULT_HORIZON,
    )


if __name__ == "__main__":
    threading.Thread(target=_warm, daemon=True).start()
    create_app().launch(css=CSS, theme=THEME)
