"""
Plotly figures.

Each model keeps one color on every chart (validated categorical slots
1-4, light steps). Text stays in neutral ink; the legend plus a table
view on every tab keeps identity from relying on color alone.
"""

import pandas as pd
import plotly.graph_objects as go

from dashboard.forecast import MODELS, Outlook

MODEL_COLORS = dict(zip(MODELS, ("#2a78d6", "#eb6834", "#1baf7a",
                                 "#eda100")))
ACTUAL = "#6f6e69"
MUTED = "#898781"
DIMMED = "#c3c2b7"
GRID = "rgba(137, 135, 129, 0.22)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _base(height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=72, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=MUTED),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.08, xanchor="left",
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside",
                     tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickprefix="$",
                     tickformat=".2f")
    return fig


def _end_label(fig, x, y, text, above=False):
    """Direct label at a line's end; ``above`` keeps it off a line that
    continues to the right."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                       xanchor="right" if above else "left",
                       yanchor="bottom" if above else "middle",
                       xshift=0 if above else 6, yshift=8 if above else 0,
                       font=dict(size=12, color=MUTED))


def _range_buttons(years: tuple[int, ...]) -> dict:
    buttons = [dict(count=y, label=f"{y}Y", step="year",
                    stepmode="backward") for y in years]
    buttons.append(dict(step="all", label="All"))
    return dict(buttons=buttons, x=1, xanchor="right", y=1.08,
                bgcolor="rgba(137, 135, 129, 0.12)",
                activecolor="rgba(137, 135, 129, 0.35)",
                font=dict(color=MUTED))


def outlook_figure(outlook: Outlook) -> go.Figure:
    """Monthly history plus the recommended model's forecast and bands."""
    model = outlook.recommended
    color = MODEL_COLORS[model]
    hist = outlook.history
    fc = outlook.forecast
    # Start the forecast trace at the last actual month so the lines meet.
    x_fc = [hist.index[-1]] + list(fc["ds"])
    y_fc = [hist.iloc[-1]] + list(fc[model])

    fig = _base()
    for level, alpha in ((95, 0.12), (80, 0.24)):
        lo, hi = outlook.band(model, level)
        fig.add_trace(go.Scatter(
            x=list(fc["ds"]) + list(fc["ds"])[::-1],
            y=list(hi) + list(lo)[::-1],
            mode="lines", fill="toself", fillcolor=_rgba(color, alpha),
            line=dict(width=0), hoverinfo="skip",
            name=f"{level}% range",
        ))
        fig.add_trace(go.Scatter(
            x=fc["ds"], y=lo, mode="lines", line=dict(width=0),
            showlegend=False, name=f"{level}% low",
            hovertemplate="$%{y:.2f}",
        ))
        fig.add_trace(go.Scatter(
            x=fc["ds"], y=hi, mode="lines", line=dict(width=0),
            showlegend=False, name=f"{level}% high",
            hovertemplate="$%{y:.2f}",
        ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist, mode="lines", name="Actual (monthly avg)",
        line=dict(color=ACTUAL, width=2), hovertemplate="$%{y:.2f}",
    ))
    fig.add_trace(go.Scatter(
        x=x_fc, y=y_fc, mode="lines", name=f"Forecast ({model})",
        line=dict(color=color, width=2, dash="dot"),
        hovertemplate="$%{y:.2f}",
    ))

    end = fc["ds"].iloc[-1]
    _end_label(fig, hist.index[-1], hist.iloc[-1],
               f"{hist.index[-1]:%b} ${hist.iloc[-1]:.2f}", above=True)
    _end_label(fig, end, fc[model].iloc[-1], f"${fc[model].iloc[-1]:.2f}")
    start = hist.index[-1] - pd.DateOffset(years=3)
    fig.update_xaxes(range=[start, end + pd.DateOffset(days=20)],
                     rangeselector=_range_buttons((3, 10)))
    fig.update_xaxes(hoverformat="%b %Y")
    return fig


def _spread_labels(values: dict[str, float], min_gap: float) -> dict:
    """Nudge end-label positions apart so they never overlap."""
    placed = {}
    last = None
    for name, y in sorted(values.items(), key=lambda kv: kv[1]):
        if last is not None and y - last < min_gap:
            y = last + min_gap
        placed[name] = last = y
    return placed


def backtest_figure(outlook: Outlook) -> go.Figure:
    """Each model's predictions for months it never saw, vs. actual."""
    bt = outlook.backtest
    context = outlook.history[outlook.history.index < bt["ds"].iloc[0]]
    context = context.iloc[-12:]

    fig = _base()
    fig.add_trace(go.Scatter(
        x=list(context.index) + list(bt["ds"]),
        y=list(context) + list(bt["actual"]),
        mode="lines", name="Actual",
        line=dict(color=ACTUAL, width=2), hovertemplate="$%{y:.2f}",
    ))
    ends = {}
    for model in MODELS:
        fig.add_trace(go.Scatter(
            x=bt["ds"], y=bt[model], mode="lines+markers", name=model,
            line=dict(color=MODEL_COLORS[model], width=2),
            marker=dict(size=8, line=dict(width=2, color="white")),
            hovertemplate="$%{y:.2f}",
        ))
        ends[model] = bt[model].iloc[-1]

    values = list(bt[list(MODELS)].to_numpy().ravel()) + list(context)
    gap = (max(values) - min(values)) * 0.06
    for model, y in _spread_labels(ends, gap).items():
        _end_label(fig, bt["ds"].iloc[-1], y, model)

    cutoff = bt["ds"].iloc[0] - pd.DateOffset(days=15)
    fig.add_vline(x=cutoff, line=dict(color=DIMMED, width=1))
    fig.add_annotation(x=cutoff, y=1, yref="paper", yanchor="bottom",
                       text="Models trained up to here", showarrow=False,
                       xanchor="right", xshift=-4,
                       font=dict(size=12, color=MUTED))
    fig.update_xaxes(hoverformat="%b %Y")
    fig.update_layout(margin=dict(r=96))
    return fig


def accuracy_figure(outlook: Outlook) -> go.Figure:
    """Backtest error (MAPE) per model, lowest first."""
    metrics = outlook.metrics.iloc[::-1]
    fig = _base(height=220)
    fig.add_trace(go.Bar(
        x=metrics["MAPE"], y=metrics["Model"], orientation="h",
        marker=dict(color=[MODEL_COLORS[m] for m in metrics["Model"]],
                    cornerradius=4),
        text=[f"{v:.1f}%" for v in metrics["MAPE"]],
        textposition="outside", cliponaxis=False,
        textfont=dict(color=MUTED),
        hovertemplate="%{y}: %{x:.1f}% average error<extra></extra>",
        width=0.55,
    ))
    fig.update_layout(hovermode="closest", showlegend=False,
                      margin=dict(l=8, r=48, t=8, b=8))
    fig.update_xaxes(showticklabels=False, showline=False, ticks="")
    fig.update_yaxes(tickprefix="", gridcolor="rgba(0,0,0,0)")
    return fig


def markets_figure(snapshot: pd.DataFrame, names: dict,
                   selected: str) -> go.Figure:
    """Latest price per market; the selected market is highlighted."""
    rows = snapshot.sort_values("price")
    labels = [names[c] for c in rows.index]
    colors = [MODEL_COLORS["AutoARIMA"] if c == selected else DIMMED
              for c in rows.index]
    text = [f"${p:.2f}" if c in (selected, "NUS") else ""
            for c, p in rows["price"].items()]
    fig = _base(height=max(260, 26 * len(rows) + 40))
    fig.add_trace(go.Bar(
        x=rows["price"], y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=text, textposition="outside", cliponaxis=False,
        textfont=dict(color=MUTED),
        hovertemplate="%{y}: $%{x:.2f}/gal<extra></extra>",
        width=0.62,
    ))
    fig.update_layout(hovermode="closest", showlegend=False,
                      margin=dict(l=8, r=56, t=8, b=8))
    fig.update_xaxes(tickprefix="$", tickformat=".2f", showgrid=True,
                     gridcolor=GRID)
    fig.update_yaxes(tickprefix="", gridcolor="rgba(0,0,0,0)")
    return fig


def history_figure(series: pd.Series, name: str) -> go.Figure:
    """Weekly price history for one market."""
    fig = _base(height=360)
    fig.add_trace(go.Scatter(
        x=series.index, y=series, mode="lines", name=name,
        line=dict(color=MODEL_COLORS["AutoARIMA"], width=2),
        hovertemplate="$%{y:.3f}",
    ))
    start = series.index[-1] - pd.DateOffset(years=5)
    fig.update_xaxes(range=[start, series.index[-1]],
                     rangeselector=_range_buttons((1, 5)),
                     hoverformat="Week of %b %d, %Y")
    fig.update_layout(showlegend=False)
    return fig
