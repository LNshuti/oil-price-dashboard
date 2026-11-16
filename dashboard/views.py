"""
HTML for the headline tiles and tables.

Styling lives in ``STYLES`` and reads Gradio's theme variables, so the
same markup works in light and dark mode.
"""

import html
from dataclasses import dataclass

import pandas as pd

from dashboard.forecast import Outlook

STYLES = """
.fo-tiles { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.fo-tile { background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: 12px; padding: 16px; }
.fo-label { font-size: 13px; color: var(--body-text-color-subdued);
  margin: 0 0 4px; }
.fo-value { font-size: 32px; font-weight: 600; line-height: 1.15;
  color: var(--body-text-color); margin: 0; }
.fo-unit { font-size: 15px; font-weight: 400;
  color: var(--body-text-color-subdued); }
.fo-meta { font-size: 13px; color: var(--body-text-color-subdued);
  margin: 6px 0 0; }
.fo-up { color: #b42318; }
.fo-down { color: #006300; }
.dark .fo-up { color: #f97066; }
.dark .fo-down { color: #0ca30c; }
.fo-note { font-size: 13px; color: var(--body-text-color-subdued);
  margin: 10px 2px 0; }
.fo-table { width: 100%; border-collapse: collapse; font-size: 14px;
  color: var(--body-text-color); border: none !important; }
.fo-scroll { overflow-x: auto; }
.fo-table thead, .fo-table tbody, .fo-table tr { border: none !important; }
.fo-table td:first-child { white-space: nowrap; }
.fo-table th, .fo-table td { border-left: none !important;
  border-right: none !important; border-top: none !important;
  background: transparent !important; }
.fo-table caption { text-align: left; font-weight: 600;
  padding: 0 0 8px; color: var(--body-text-color); }
.fo-table th { text-align: left; font-weight: 600; padding: 8px 12px;
  border-bottom: 1px solid var(--border-color-primary);
  color: var(--body-text-color-subdued); }
.fo-table td { padding: 8px 12px;
  border-bottom: 1px solid var(--border-color-primary); }
.fo-table .num { text-align: right; font-variant-numeric: tabular-nums;
  white-space: nowrap; }
.fo-table tr.fo-selected td { font-weight: 600; }
.fo-badge { font-size: 12px; font-weight: 500; padding: 2px 8px;
  border-radius: 999px; margin-left: 6px;
  background: var(--background-fill-secondary);
  color: var(--body-text-color-subdued); }
"""


def price(value: float) -> str:
    return f"${value:,.2f}"


def dollars(value: float) -> str:
    return f"${value:,.0f}"


def delta(change: float, label: str) -> str:
    """'▲ 12¢ vs 4 weeks ago', worded so color is never the only cue."""
    if pd.isna(change):
        return ""
    cents = round(change * 100)
    if cents == 0:
        return f"No change vs {label}"
    arrow, css, word = (("▲", "fo-up", "up") if cents > 0
                        else ("▼", "fo-down", "down"))
    amount = (f"{abs(cents)}¢" if abs(cents) < 100
              else f"${abs(change):,.2f}")
    return (f'<span class="{css}" aria-label="{word} {amount}">'
            f"{arrow} {amount}</span> vs {label}")


def month(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %Y")


@dataclass(frozen=True)
class Budget:
    expected: float
    low: float
    high: float
    at_today: float


def budget(outlook: Outlook, gallons: float, today: float) -> Budget:
    """Fuel spend over the horizon at ``gallons`` per month.

    The range totals the monthly 80% bounds, i.e. prices staying near the
    low or high edge all period - a conservative planning range.
    """
    model = outlook.recommended
    lo, hi = outlook.band(model, 80)
    return Budget(
        expected=float(outlook.forecast[model].sum() * gallons),
        low=float(lo.sum() * gallons),
        high=float(hi.sum() * gallons),
        at_today=float(today * outlook.horizon * gallons),
    )


def _tile(label: str, value: str, unit: str, meta: list[str]) -> str:
    lines = "".join(f'<p class="fo-meta">{m}</p>' for m in meta if m)
    return (f'<div class="fo-tile"><p class="fo-label">{label}</p>'
            f'<p class="fo-value">{value}<span class="fo-unit">{unit}'
            f"</span></p>{lines}</div>")


def headline(*, market: str, fuel: str, latest: pd.Series,
             outlook: Outlook, gallons: float) -> str:
    """Current price, next forecast month and the fleet budget."""
    now = latest.iloc[-1]
    week = latest.index[-1]
    ago_4 = now - latest.iloc[-5] if len(latest) > 4 else float("nan")
    ago_52 = now - latest.iloc[-53] if len(latest) > 52 else float("nan")

    model = outlook.recommended
    fc = outlook.forecast
    first = fc["ds"].iloc[0]
    lo, hi = outlook.band(model, 80)
    current = _tile(
        f"{html.escape(market)} · {html.escape(fuel)}", price(now), "/gal",
        [f"Week of {week:%b} {week.day}, {week.year}",
         delta(ago_4, "4 weeks ago"),
         delta(ago_52, "a year ago")],
    )
    nxt = _tile(
        f"{month(first)} average (forecast)", price(fc[model].iloc[0]),
        "/gal",
        [f"80% range {price(lo.iloc[0])}–{price(hi.iloc[0])}",
         delta(fc[model].iloc[0] - now, "this week")],
    )
    tiles = [current, nxt]
    if gallons and gallons > 0:
        b = budget(outlook, gallons, now)
        span = f"{month(first)}–{month(fc['ds'].iloc[-1])}"
        diff = b.expected - b.at_today
        word = "more" if diff >= 0 else "less"
        tiles.append(_tile(
            f"{outlook.horizon}-month fuel budget", dollars(b.expected), "",
            [f"{span} at {gallons:,.0f} gal/month",
             f"80% range {dollars(b.low)}–{dollars(b.high)}",
             f"{dollars(abs(diff))} {word} than at this week's price"],
        ))
    mape = outlook.metrics.loc[0, "MAPE"]
    note = (f"Forecast by {model}, the most accurate of four models when "
            f"tested on the last {outlook.horizon} months it hadn't seen "
            f"(average error {mape:.1f}%). See Model check.")
    return (f'<div class="fo-tiles">{"".join(tiles)}</div>'
            f'<p class="fo-note">{note}</p>')


def table(df: pd.DataFrame, numeric: set[str], caption: str = "",
          selected_row: int | None = None) -> str:
    """Plain HTML table; ``numeric`` columns are right-aligned."""
    head = "".join(
        f'<th class="{"num" if c in numeric else ""}">{html.escape(c)}</th>'
        for c in df.columns)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = "".join(
            f'<td class="{"num" if c in numeric else ""}">{row[c]}</td>'
            for c in df.columns)
        css = ' class="fo-selected"' if i == selected_row else ""
        rows.append(f"<tr{css}>{cells}</tr>")
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return (f'<div class="fo-scroll"><table class="fo-table">{cap}'
            f"<thead><tr>{head}</tr></thead>"
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def forecast_table(outlook: Outlook) -> str:
    model = outlook.recommended
    fc = outlook.forecast
    lo80, hi80 = outlook.band(model, 80)
    lo95, hi95 = outlook.band(model, 95)
    df = pd.DataFrame({
        "Month": [month(d) for d in fc["ds"]],
        "Forecast ($/gal)": [price(v) for v in fc[model]],
        "80% range": [f"{price(a)}–{price(b)}" for a, b in zip(lo80, hi80)],
        "95% range": [f"{price(a)}–{price(b)}" for a, b in zip(lo95, hi95)],
    })
    return table(df, {"Forecast ($/gal)", "80% range", "95% range"},
                 caption=f"Monthly forecast · {model}")


def all_models_table(outlook: Outlook) -> str:
    fc = outlook.forecast
    data = {"Month": [month(d) for d in fc["ds"]]}
    for model in outlook.metrics["Model"]:
        data[model] = [price(v) for v in fc[model]]
    df = pd.DataFrame(data)
    return table(df, set(df.columns) - {"Month"},
                 caption="Every model's forecast ($/gal)")


def metrics_table(outlook: Outlook) -> str:
    m = outlook.metrics
    names = [html.escape(n) + ('<span class="fo-badge">Recommended</span>'
                               if i == 0 else "")
             for i, n in enumerate(m["Model"])]
    df = pd.DataFrame({
        "Model": names,
        "Avg error (MAPE)": [f"{v:.1f}%" for v in m["MAPE"]],
        "MAE ($/gal)": [price(v) for v in m["MAE"]],
        "RMSE ($/gal)": [price(v) for v in m["RMSE"]],
    })
    return table(df, {"Avg error (MAPE)", "MAE ($/gal)", "RMSE ($/gal)"},
                 caption=f"Backtest on the last {outlook.horizon} months")


def markets_table(snapshot: pd.DataFrame, names: dict,
                  selected: str) -> str:
    rows = snapshot.sort_values("price", ascending=False)
    us = snapshot.loc["NUS", "price"] if "NUS" in snapshot.index else None

    def cents(v):
        if pd.isna(v):
            return "–"
        return f"{'+' if v > 0 else '−' if v < 0 else ''}{abs(v) * 100:.0f}¢"

    df = pd.DataFrame({
        "Market": [html.escape(names[c]) for c in rows.index],
        "Price ($/gal)": [price(v) for v in rows["price"]],
        "4-week change": [cents(v) for v in rows["change_4w"]],
        "1-year change": [cents(v) for v in rows["change_52w"]],
        "vs U.S.": [cents(v - us) if us is not None else "–"
                    for v in rows["price"]],
    })
    selected_row = list(rows.index).index(selected) \
        if selected in rows.index else None
    return table(df, set(df.columns) - {"Market"},
                 caption="Latest week, every market EIA measures",
                 selected_row=selected_row)
