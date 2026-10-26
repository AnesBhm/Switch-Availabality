"""
app.py — Switch Availability Dash Dashboard
============================================
A premium, dark-themed Plotly Dash dashboard for the Switch Silence Detection Engine.

Tabs:
  1. Overview     — KPI cards (uptime %, gap count, MTTR, transactions)
  2. Daily Report — Interactive SLA summary table
  3. Gap Log      — Filterable table of all detected silent gaps
  4. Timeline     — Per-date silence-gap vs. threshold chart
  5. Profile      — Hourly Z-score & floor calibration chart

Run:  python app.py
Open: http://127.0.0.1:8050
"""

import os
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

# ── Google Fonts CDN ────────────────────────────────────────────────────────
EXTERNAL_STYLESHEETS = [
    dbc.themes.CYBORG,
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap",
]

app = Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    title="Switch Availability Dashboard",
    suppress_callback_exceptions=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR & STYLE TOKENS
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    bg         = "#05080f",
    surface    = "#0d1117",
    card       = "#111827",
    card2      = "#1a2236",
    border     = "#1e2d4a",
    accent     = "#3b82f6",
    accent2    = "#8b5cf6",
    green      = "#22c55e",
    yellow     = "#f59e0b",
    red        = "#ef4444",
    text       = "#f1f5f9",
    muted      = "#94a3b8",
    line_gap   = "#60a5fa",
    line_thr   = "#34d399",
    dot_outlier= "#ef4444",
)

FONT = "'Outfit', 'Inter', sans-serif"

PLOTLY_LAYOUT = dict(
    font_family   = FONT,
    paper_bgcolor = C["card"],
    plot_bgcolor  = C["surface"],
    font_color    = C["text"],
    margin=dict(l=50, r=20, t=50, b=50),
    # xaxis, yaxis, legend are set per-chart to avoid Plotly 6.x duplicate-keyword errors
)

# Shared axis / legend defaults applied via separate update_* calls
_AXIS_STYLE  = dict(gridcolor=C["border"], linecolor=C["border"])
_LEGEND_STYLE = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"])


def _apply_layout(fig, **extra):
    """
    Apply PLOTLY_LAYOUT to a figure safely.
    Handles Plotly 6.x's strict 'no duplicate kwargs' rule by splitting
    axis/legend styling into separate update_*axes() calls.
    Pass any extra kwargs (title, height, showlegend, etc.) via **extra.
    """
    # Pop keys that conflict with PLOTLY_LAYOUT or each other
    legend_kw  = extra.pop("legend",  None)
    xaxis_kw   = extra.pop("xaxis",   None)
    yaxis_kw   = extra.pop("yaxis",   None)

    fig.update_layout(**PLOTLY_LAYOUT, **extra)

    # Axis defaults (merged with any per-chart overrides)
    xa = dict(**_AXIS_STYLE)
    if xaxis_kw:
        xa.update(xaxis_kw)
    ya = dict(**_AXIS_STYLE)
    if yaxis_kw:
        ya.update(yaxis_kw)
    fig.update_xaxes(**xa)
    fig.update_yaxes(**ya)

    # Legend default (merged with any per-chart overrides)
    lg = dict(**_LEGEND_STYLE)
    if legend_kw:
        lg.update(legend_kw)
    fig.update_layout(legend=lg)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — runs once at startup (heavy, ~10-30 s on 554 MB CSV)
# ─────────────────────────────────────────────────────────────────────────────
_data = {}          # filled by _load_pipeline()
_loading_error = [] # populated on failure

def _load_pipeline():
    try:
        from engine_api import run_pipeline, profile_to_df
        result = run_pipeline()
        _data.update(result)
        _data['profile_df'] = profile_to_df(result['profile'])
        # Pre-compute reference profile for mentor comparison
        ref = result['config'].mentor_reference
        _data['ref_profile'] = ref
        # 'date' column contains Python date objects (not datetime64), so no .dt accessor
        _data['available_dates'] = sorted(_data['scored_df']['date'].unique())
        _data['loaded'] = True
    except Exception as exc:
        import traceback
        _loading_error.append(traceback.format_exc())
        _data['loaded'] = False
        print("PIPELINE ERROR:", exc)

# Start loading in a background thread so the UI renders immediately
_thread = threading.Thread(target=_load_pipeline, daemon=True)
_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(title, value, subtitle="", color=C["accent"], icon=""):
    return html.Div(
        className="kpi-card glass-panel",
        style={
            "padding": "24px 20px",
            "flex": "1",
            "minWidth": "180px",
            "position": "relative",
            "overflow": "hidden",
        },
        children=[
            html.Div(icon, style={"fontSize": "28px", "marginBottom": "8px"}),
            html.Div(title, style={
                "fontSize": "11px", "fontWeight": "600", "letterSpacing": "1px",
                "textTransform": "uppercase", "color": C["muted"], "marginBottom": "6px"
            }),
            html.Div(value, style={
                "fontSize": "32px", "fontWeight": "700",
                "color": color, "lineHeight": "1.1",
            }),
            html.Div(subtitle, style={"fontSize": "12px", "color": C["muted"], "marginTop": "4px"}),
            # Glow accent bar
            html.Div(style={
                "position": "absolute", "bottom": "0", "left": "0", "right": "0",
                "height": "3px",
                "background": f"linear-gradient(90deg, {color}, transparent)",
            }),
        ]
    )


def section_header(title, subtitle=""):
    return html.Div([
        html.H3(title, style={"margin": "0", "fontWeight": "600", "fontSize": "20px", "color": C["text"]}),
        html.P(subtitle, style={"margin": "4px 0 0", "color": C["muted"], "fontSize": "13px"}),
    ], style={"marginBottom": "20px"})


def styled_table(df, table_id, page_size=15, color_uptime=False):
    """Render a DataTable with the dark premium theme."""
    columns = [{"name": c, "id": c} for c in df.columns]

    style_data_conditional = [
        {"if": {"row_index": "odd"}, "backgroundColor": C["card2"]},
    ]
    if color_uptime and "Uptime (%)" in df.columns:
        style_data_conditional += [
            {"if": {"filter_query": "{Uptime (%)} >= 99.9", "column_id": "Uptime (%)"},
             "color": C["green"], "fontWeight": "700"},
            {"if": {"filter_query": "{Uptime (%)} < 99.9 && {Uptime (%)} >= 99",
                    "column_id": "Uptime (%)"},
             "color": C["yellow"], "fontWeight": "700"},
            {"if": {"filter_query": "{Uptime (%)} < 99", "column_id": "Uptime (%)"},
             "color": C["red"], "fontWeight": "700"},
        ]

    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=df.to_dict("records"),
        page_size=page_size,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto", "borderRadius": "12px", "overflow": "hidden"},
        style_header={
            "backgroundColor": C["card2"],
            "color": C["accent"],
            "fontWeight": "700",
            "fontSize": "12px",
            "letterSpacing": "0.5px",
            "textTransform": "uppercase",
            "border": f"1px solid {C['border']}",
        },
        style_cell={
            "backgroundColor": C["card"],
            "color": C["text"],
            "border": f"1px solid {C['border']}",
            "padding": "10px 14px",
            "fontFamily": FONT,
            "fontSize": "13px",
        },
        style_data_conditional=style_data_conditional,
    )

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_STYLE = {
    "fontFamily": FONT,
    "minHeight": "100vh",
    "color": C["text"],
}

HEADER = html.Div(
    className="glass-header",
    style={
        "padding": "20px 36px",
        "display": "flex",
        "alignItems": "center",
        "gap": "16px",
    },
    children=[
        html.Div("⚡", style={"fontSize": "36px"}),
        html.Div([
            html.H1("Switch Availability Dashboard",
                    style={"margin": "0", "fontSize": "22px", "fontWeight": "700",
                           "background": f"linear-gradient(90deg, {C['accent']}, {C['accent2']})",
                           "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"}),
            html.P("Half-Normal Silence Detection Engine · Hourly Calibration",
                   style={"margin": "2px 0 0", "color": C["muted"], "fontSize": "12px"}),
        ]),
        html.Div(style={"flex": "1"}),
        # Status badge — updated by interval
        html.Div(id="status-badge", children=[
            html.Span("⏳ Loading data…", style={"color": C["yellow"]})
        ]),
    ]
)

ENABLE_LIVE_MONITOR = False  # Set to True to enable the Live Monitor tab

TABS = dcc.Tabs(
    id="main-tabs",
    value="tab-overview",
    className="custom-tabs-container",
    children=[
        dcc.Tab(label="🏠  Overview",      value="tab-overview",  className="custom-tab", selected_className="custom-tab-selected"),
        dcc.Tab(label="📅  Daily Report",  value="tab-daily",     className="custom-tab", selected_className="custom-tab-selected"),
        dcc.Tab(label="🔕  Gap Log",       value="tab-gaps",      className="custom-tab", selected_className="custom-tab-selected"),
        dcc.Tab(label="📈  Timeline",      value="tab-timeline",  className="custom-tab", selected_className="custom-tab-selected"),
        dcc.Tab(label="🎛️  Profile",       value="tab-profile",   className="custom-tab", selected_className="custom-tab-selected"),
    ] + ([
        dcc.Tab(label="🔴  Live Monitor",  value="tab-live",      className="custom-tab", selected_className="custom-tab-selected", style={"color": C["red"]}),
    ] if ENABLE_LIVE_MONITOR else [])
)

app.layout = html.Div(style=GLOBAL_STYLE, children=[
    HEADER,
    TABS,
    html.Div(id="tab-content", style={"padding": "28px 36px"}),
    # Polling interval: check every 2s until data is loaded
    dcc.Interval(id="load-interval", interval=2_000, n_intervals=0, max_intervals=60),
    # Live monitor streaming — 1-second tick, only active on live tab
    dcc.Interval(id="live-interval", interval=1_000, n_intervals=0, disabled=True),
    dcc.Store(id="live-store", data=None),
])

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK — tab router + loading guard
# ─────────────────────────────────────────────────────────────────────────────

LOADING_SCREEN = html.Div([
    html.Div("⚙️", style={"fontSize": "60px", "textAlign": "center"}),
    html.H3("Running pipeline…", style={"textAlign": "center", "color": C["accent"]}),
    html.P("Loading ~554 MB dataset, calibrating hourly profile, and scoring all target dates.",
           style={"textAlign": "center", "color": C["muted"]}),
    html.P("This takes about 10–30 seconds on first launch.",
           style={"textAlign": "center", "color": C["muted"], "fontSize": "12px"}),
    dcc.Loading(type="circle", color=C["accent"],
                children=html.Div(id="loader-placeholder", style={"height": "60px"})),
], style={"maxWidth": "600px", "margin": "80px auto", "textAlign": "center"})

ERROR_SCREEN = lambda msg: html.Div([
    html.Div("❌", style={"fontSize": "60px", "textAlign": "center"}),
    html.H3("Pipeline Error", style={"textAlign": "center", "color": C["red"]}),
    html.Pre(msg, style={
        "background": C["card"], "padding": "16px", "borderRadius": "8px",
        "fontSize": "12px", "color": C["red"], "overflowX": "auto",
    }),
], style={"maxWidth": "800px", "margin": "60px auto"})


@app.callback(
    Output("tab-content",  "children"),
    Output("status-badge", "children"),
    Output("load-interval", "max_intervals"),
    Input("load-interval", "n_intervals"),
    Input("main-tabs",     "value"),
)
def render_tab(n, tab):
    if _loading_error:
        return ERROR_SCREEN(_loading_error[0]), html.Span("❌ Error", style={"color": C["red"]}), 0

    if not _data.get("loaded"):
        return LOADING_SCREEN, html.Span("⏳ Loading…", style={"color": C["yellow"]}), 60

    # Data is ready — stop the interval
    badge = html.Span([
        html.Span("●", style={"color": C["green"], "marginRight": "6px"}),
        html.Span("Live", style={"color": C["green"], "fontWeight": "600"}),
    ])

    if tab == "tab-overview":
        content = build_overview()
    elif tab == "tab-daily":
        content = build_daily()
    elif tab == "tab-gaps":
        content = build_gaps()
    elif tab == "tab-timeline":
        content = build_timeline_tab()
    elif tab == "tab-profile":
        content = build_profile()
    elif tab == "tab-live":
        content = build_live_monitor()
    else:
        content = html.Div("Select a tab above.")

    return content, badge, 0  # 0 = stop polling


# Enable/disable live interval based on active tab
@app.callback(
    Output("live-interval", "disabled"),
    Input("main-tabs", "value"),
)
def toggle_live_interval(tab):
    return tab != "tab-live"


# ─────────────────────────────────────────────────────────────────────────────
# TAB BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_overview():
    df = _data["summary_df"]
    eval_df = df[df["Type"] == "EVALUATION"]

    avg_uptime  = eval_df["Uptime (%)"].mean()
    total_gaps  = eval_df["Silent Gaps"].sum()
    avg_mttr    = eval_df["MTTR (min)"].mean()
    total_txns  = df["Txn Count"].sum()
    fp          = _data["fp_baseline"]

    def uptime_color(u):
        if u >= 99.9: return C["green"]
        if u >= 99.0: return C["yellow"]
        return C["red"]

    kpis = html.Div(
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "28px"},
        children=[
            kpi_card("Avg Uptime", f"{avg_uptime:.3f}%",
                     "Evaluation dates", uptime_color(avg_uptime), "🟢"),
            kpi_card("Total Gaps Detected", f"{int(total_gaps)}",
                     "Across all eval days", C["accent2"], "🔕"),
            kpi_card("Avg MTTR", f"{avg_mttr:.2f} min",
                     "Mean time to recover", C["yellow"], "⏱️"),
            kpi_card("Total Transactions", f"{int(total_txns):,}",
                     "All dates combined", C["accent"], "📊"),
            kpi_card("Baseline False Positives", f"{fp}",
                     f"On {str(_data['config'].real_date)}", C["green"] if fp == 0 else C["red"], "✅"),
        ]
    )

    # Uptime bar chart per day
    fig_uptime = go.Figure()
    colors_bar = [uptime_color(u) for u in df["Uptime (%)"]]
    fig_uptime.add_trace(go.Bar(
        x=df["Date"].astype(str),
        y=df["Uptime (%)"],
        marker_color=colors_bar,
        text=[f"{u:.3f}%" for u in df["Uptime (%)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Uptime: %{y:.4f}%<extra></extra>",
    ))
    fig_uptime.add_hline(y=99.9, line_dash="dash", line_color=C["muted"],
                         annotation_text="99.9% SLA", annotation_font_color=C["muted"])
    _apply_layout(fig_uptime,
        title="Daily Uptime (%)",
        yaxis_range=[min(df["Uptime (%)"].min() - 0.05, 99.8), 100.01],
        showlegend=False,
        height=350,
    )

    # Gaps + Downtime dual-axis
    fig_gaps = make_subplots(specs=[[{"secondary_y": True}]])
    fig_gaps.add_trace(go.Bar(
        x=df["Date"].astype(str), y=df["Silent Gaps"],
        name="Silent Gaps", marker_color=C["accent2"],
        hovertemplate="<b>%{x}</b><br>Gaps: %{y}<extra></extra>",
    ), secondary_y=False)
    fig_gaps.add_trace(go.Scatter(
        x=df["Date"].astype(str), y=df["Downtime (min)"],
        name="Downtime (min)", mode="lines+markers",
        line=dict(color=C["red"], width=2),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Downtime: %{y:.2f} min<extra></extra>",
    ), secondary_y=True)
    _apply_layout(fig_gaps,
        title="Silent Gaps & Downtime per Day",
        height=350,
        legend=dict(orientation="h", y=-0.15),
    )
    fig_gaps.update_yaxes(title_text="Gap Count",    secondary_y=False, gridcolor=C["border"], color=C["text"])
    fig_gaps.update_yaxes(title_text="Downtime (min)", secondary_y=True,  color=C["red"])

    return html.Div([
        section_header("Executive Overview", "Summary KPIs across all target dates"),
        kpis,
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            html.Div(dcc.Graph(figure=fig_uptime, config={"displayModeBar": False}),
                     className="glass-panel", style={"padding": "8px"}),
            html.Div(dcc.Graph(figure=fig_gaps, config={"displayModeBar": False}),
                     className="glass-panel", style={"padding": "8px"}),
        ]),
    ])


def build_daily():
    df = _data["summary_df"].copy()
    df["Date"] = df["Date"].astype(str)
    return html.Div([
        section_header("Daily SLA Report", "Per-day availability and reliability metrics"),
        styled_table(df, "daily-table", page_size=15, color_uptime=True),
    ])


def build_gaps():
    df = _data["gaps_df"].copy()
    if len(df) == 0:
        return html.Div([
            section_header("Silent Gap Log", "All detected silence anomalies"),
            html.P("✅ No silent gaps detected across all evaluation dates.",
                   style={"color": C["green"], "fontSize": "18px", "textAlign": "center", "marginTop": "60px"}),
        ])

    df["date"]           = df["date"].astype(str)
    df["gap_start"]      = pd.to_datetime(df["gap_start"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["gap_end"]        = pd.to_datetime(df["gap_end"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["delta_t"]        = df["delta_t"].round(2)
    df["switch_threshold"] = df["switch_threshold"].round(2)
    df["duration_minutes"] = df["duration_minutes"].round(3)

    return html.Div([
        section_header("Silent Gap Log",
                       f"{len(df)} detected gap events · sorted by duration (longest first)"),
        styled_table(df, "gaps-table", page_size=20),
    ])


def build_timeline_tab():
    dates = _data["available_dates"]
    date_options = [{"label": str(d), "value": str(d)} for d in dates]
    default = str(dates[0]) if dates else None

    return html.Div([
        section_header("Silence Gap Timeline",
                       "Transaction arrival gaps vs. dynamic silence threshold per day"),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px", "marginBottom": "20px"}, children=[
            html.Label("Select Date:", style={"color": C["muted"], "fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="timeline-date-dropdown",
                options=date_options,
                value=default,
                clearable=False,
                style={"width": "200px", "fontFamily": FONT, "color": C["bg"]},
            ),
        ]),
        dcc.Loading(
            type="dot",
            color=C["accent"],
            children=dcc.Graph(id="timeline-chart", config={"displayModeBar": True}),
        ),
    ])


@app.callback(
    Output("timeline-chart", "figure"),
    Input("timeline-date-dropdown", "value"),
)
def update_timeline(selected_date):
    if not selected_date or not _data.get("loaded"):
        return go.Figure()

    scored_df = _data["scored_df"]
    # 'date' column contains Python date objects, convert directly with astype(str)
    day_df = scored_df[scored_df["date"].astype(str) == selected_date].copy()

    if len(day_df) == 0:
        fig = go.Figure()
        _apply_layout(fig, title=f"No data for {selected_date}")
        return fig

    ts = day_df["server_timestamp"]
    outliers = day_df[day_df["is_silent_outlier"]]
    cfg = _data["config"]
    is_base = selected_date == str(cfg.real_date)
    tag = "CALIBRATION BASELINE" if is_base else "EVALUATION DATE"

    fig = go.Figure()

    # Silence gap trace
    fig.add_trace(go.Scatter(
        x=ts, y=day_df["delta_t"],
        mode="lines",
        name="Silence Gap (Δt)",
        line=dict(color=C["line_gap"], width=1.2),
        opacity=0.75,
        hovertemplate="<b>%{x|%H:%M:%S}</b><br>Δt: %{y:.2f}s<extra></extra>",
    ))

    # Threshold line
    fig.add_trace(go.Scatter(
        x=ts, y=day_df["switch_threshold"],
        mode="lines",
        name="Switch Threshold",
        line=dict(color=C["line_thr"], width=2.5, dash="dash"),
        hovertemplate="Threshold: %{y:.2f}s<extra></extra>",
    ))

    # Outlier markers
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(
            x=outliers["server_timestamp"],
            y=outliers["delta_t"],
            mode="markers",
            name=f"Detected Unavailability ({len(outliers)})",
            marker=dict(
                color=C["dot_outlier"], size=9, symbol="circle",
                line=dict(color="white", width=1),
            ),
            hovertemplate="<b>%{x|%H:%M:%S}</b><br>Gap: %{y:.2f}s<extra></extra>",
        ))

    _apply_layout(fig,
        title=f"Date: {selected_date}  [{tag}]",
        xaxis_title="Time of Day",
        yaxis_title="Silence Gap (seconds)",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
    )
    fig.update_xaxes(tickformat="%H:%M", dtick=3_600_000)

    return fig


def build_profile():
    profile_df = _data["profile_df"]
    ref = _data["ref_profile"]

    hours = list(range(24))
    ref_z     = [ref.get(h, (np.nan, np.nan))[0] for h in hours]
    ref_floor = [ref.get(h, (np.nan, np.nan))[1] for h in hours]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Hourly Z-Score Multiplier (Zₕ)", "Minimum Physical Floor — min_thresh_sec (log scale)"],
        vertical_spacing=0.16,
    )

    # Z-score
    fig.add_trace(go.Scatter(
        x=profile_df["label"], y=profile_df["z_score"],
        mode="lines+markers", name="Calibrated Z-Score",
        line=dict(color=C["accent"], width=2.5),
        marker=dict(size=8),
        hovertemplate="Hour %{x}<br>Z: %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=profile_df["label"], y=ref_z,
        mode="lines+markers", name="Mentor Reference Z",
        line=dict(color=C["muted"], width=1.5, dash="dash"),
        marker=dict(size=6, symbol="square"),
        hovertemplate="Hour %{x}<br>Ref Z: %{y}<extra></extra>",
    ), row=1, col=1)

    # Floor
    fig.add_trace(go.Scatter(
        x=profile_df["label"], y=profile_df["min_thresh_sec"],
        mode="lines+markers", name="Calibrated Floor",
        line=dict(color=C["red"], width=2.5),
        marker=dict(size=8),
        hovertemplate="Hour %{x}<br>Floor: %{y:.2f}s<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=profile_df["label"], y=ref_floor,
        mode="lines+markers", name="Mentor Reference Floor",
        line=dict(color=C["muted"], width=1.5, dash="dash"),
        marker=dict(size=6, symbol="square"),
        hovertemplate="Hour %{x}<br>Ref Floor: %{y}s<extra></extra>",
    ), row=2, col=1)

    _apply_layout(fig,
        height=660,
        title="Hourly Calibration Profile vs. Mentor Reference",
    )
    fig.update_yaxes(type="log", row=2, col=1, gridcolor=C["border"])
    for row in [1, 2]:
        fig.update_xaxes(tickangle=-45, row=row, col=1, gridcolor=C["border"])
        fig.update_yaxes(gridcolor=C["border"], row=row, col=1)

    # Profile details table
    tbl_df = profile_df[["label", "z_score", "min_thresh_sec", "sigma_half", "p99_gap", "n_transactions"]].copy()
    tbl_df.columns = ["Hour", "Z-Score", "Floor (s)", "σ_half", "P99 Gap (s)", "Transactions"]
    tbl_df = tbl_df.round(3)

    return html.Div([
        section_header("Calibration Profile",
                       "Hourly Z-scores and physical floor thresholds vs. mentor reference baseline"),
        html.Div(dcc.Graph(figure=fig, config={"displayModeBar": True}),
                 className="glass-panel", style={"padding": "8px", "marginBottom": "24px"}),
        section_header("Profile Data Table", "Raw calibration values per hour"),
        styled_table(tbl_df, "profile-table", page_size=24),
    ])



# =============================================================================
# LIVE MONITOR — Streaming Simulation
# =============================================================================
from datetime import datetime as _dt, timedelta as _td

# Simulation constants
_LIVE_MAX_PTS  = 300     # rolling window length on chart
_LIVE_SPEED    = 45      # simulated seconds generated per 1 real second
_OUTAGE_PROB   = 0.004   # probability of synthetic outage per transaction
_OUTAGE_MULT   = (2.5, 7.0)  # range of threshold multiplier for injected outages


def _compute_threshold(h: int, profile: dict, cfg) -> float:
    """Reconstruct the switch threshold exactly as the scoring engine does."""
    p = profile[h]
    return max(
        p['z_score'] * p['sigma_half'] + cfg.constant_buffer_sec,
        p['min_thresh_sec'],
    )


def _gen_batch(sim_ms: int, profile: dict, cfg, rng, n_sim_sec: int):
    """
    Generate synthetic transactions for n_sim_sec of simulated time.
    Inter-arrival times ~ |N(0, sigma_half)| (half-normal), matching Oct-31.
    Returns (list_of_records, new_sim_ms).
    """
    records = []
    cur_ms  = sim_ms
    end_ms  = sim_ms + n_sim_sec * 1_000

    while cur_ms < end_ms and len(records) < 80:
        h       = int((cur_ms // 3_600_000) % 24)
        p       = profile[h]
        sigma   = p['sigma_half']
        thresh  = _compute_threshold(h, profile, cfg)

        # Sample inter-arrival time
        if sigma > 0:
            delta = abs(float(rng.normal(0, sigma)))
            delta = max(delta, 1e-4)
        else:
            delta = 1.0

        # Occasionally inject a synthetic outage
        is_outage = rng.random() < _OUTAGE_PROB
        if is_outage:
            delta = float(rng.uniform(thresh * _OUTAGE_MULT[0],
                                      thresh * _OUTAGE_MULT[1]))

        cur_ms += int(delta * 1_000)
        records.append({
            'sim_ms':     cur_ms,
            'delta_t':    round(delta, 3),
            'threshold':  round(thresh, 3),
            'hour':       h,
            'is_outlier': int(delta > thresh),
        })

    return records, cur_ms


def build_live_monitor():
    """Static shell for the Live Monitor tab (dynamic parts filled by callback)."""
    return html.Div([
        # ── Status banner ────────────────────────────────────────────────────
        html.Div(id='live-status-banner', children=[
            html.Span("Waiting for first tick…", style={"color": C["muted"]})
        ], style={
            "padding":       "14px 24px",
            "borderRadius":  "14px",
            "marginBottom":  "18px",
            "textAlign":     "center",
            "fontSize":      "16px",
            "fontWeight":    "700",
            "letterSpacing": "1px",
            "transition":    "background 0.4s",
            "border":        f"1px solid {C['border']}",
            "background":    C["card"],
        }),

        # ── KPI strip ────────────────────────────────────────────────────────
        html.Div(id='live-kpi-strip', style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "14px",
            "marginBottom": "20px",
        }),

        # ── Live chart ───────────────────────────────────────────────────────
        html.Div(
            dcc.Graph(
                id='live-chart',
                config={"displayModeBar": False},
                style={"height": "430px"},
            ),
            style={
                "background":    C["card"],
                "borderRadius":  "16px",
                "border":        f"1px solid {C['border']}",
                "padding":       "8px",
                "marginBottom":  "20px",
            },
        ),

        # ── Detected outage log ───────────────────────────────────────────────
        html.Div([
            section_header("Detected Silence Events",
                           "Transactions that exceeded the hourly threshold"),
            html.Div(id='live-outage-log'),
        ]),
    ])


# ── Streaming callback ────────────────────────────────────────────────────────
@app.callback(
    Output('live-store',         'data'),
    Output('live-chart',         'figure'),
    Output('live-status-banner', 'children'),
    Output('live-status-banner', 'style'),
    Output('live-kpi-strip',     'children'),
    Output('live-outage-log',    'children'),
    Input('live-interval',       'n_intervals'),
    State('live-store',          'data'),
    prevent_initial_call=True,
)
def update_live(n_intervals, store):
    """Generate a batch of synthetic transactions and refresh the live chart."""

    # Guard: data must be loaded
    if not _data.get('loaded'):
        empty_fig = go.Figure()
        _apply_layout(empty_fig, title="Waiting for pipeline...")
        banner_style = {
            "padding": "14px 24px", "borderRadius": "14px",
            "marginBottom": "18px", "textAlign": "center",
            "fontSize": "16px", "fontWeight": "700", "letterSpacing": "1px",
            "background": C["card"], "border": f"1px solid {C['border']}",
        }
        return store, empty_fig, "Loading pipeline…", banner_style, [], []

    profile = _data['profile']
    cfg     = _data['config']
    rng     = np.random.default_rng(n_intervals * 7919)  # deterministic per tick

    # ── Initialise store on first tick ────────────────────────────────────────
    if store is None:
        store = {
            'sim_ms':         0,
            'ts_ms_list':     [],
            'dt_list':        [],
            'thr_list':       [],
            'flag_list':      [],
            'hour_list':      [],
            'total_txn':      0,
            'total_outliers': 0,
            'outage_events':  [],   # list of dicts for the log table
        }

    # ── Generate batch ────────────────────────────────────────────────────────
    records, new_sim_ms = _gen_batch(
        store['sim_ms'], profile, cfg, rng, _LIVE_SPEED
    )
    store['sim_ms'] = new_sim_ms % (86_400 * 1_000)   # wrap every simulated day

    for r in records:
        store['ts_ms_list'].append(r['sim_ms'])
        store['dt_list'].append(r['delta_t'])
        store['thr_list'].append(r['threshold'])
        store['flag_list'].append(r['is_outlier'])
        store['hour_list'].append(r['hour'])
        store['total_txn'] += 1
        if r['is_outlier']:
            store['total_outliers'] += 1
            sim_h  = r['hour']
            sim_ts = _dt(2023, 10, 31) + _td(milliseconds=r['sim_ms'])
            store['outage_events'].append({
                'time':   sim_ts.strftime('%H:%M:%S'),
                'hour':   sim_h,
                'gap_s':  r['delta_t'],
                'thresh': r['threshold'],
                'x_over': round(r['delta_t'] / r['threshold'], 2),
            })

    # Trim rolling window
    for key in ['ts_ms_list', 'dt_list', 'thr_list', 'flag_list', 'hour_list']:
        store[key] = store[key][-_LIVE_MAX_PTS:]

    # Keep last 50 outage events in log
    store['outage_events'] = store['outage_events'][-50:]

    # ── Build x-axis (simulated timestamps) ───────────────────────────────────
    base   = _dt(2023, 10, 31, 0, 0, 0)
    ts_dts = [base + _td(milliseconds=ms) for ms in store['ts_ms_list']]
    ts_str = [t.strftime('%H:%M:%S') for t in ts_dts]

    dt_arr   = store['dt_list']
    thr_arr  = store['thr_list']
    flag_arr = store['flag_list']

    # Outlier mask
    out_x  = [ts_str[i] for i, f in enumerate(flag_arr) if f]
    out_y  = [dt_arr[i] for i, f in enumerate(flag_arr) if f]
    norm_x = [ts_str[i] for i, f in enumerate(flag_arr) if not f]
    norm_y = [dt_arr[i] for i, f in enumerate(flag_arr) if not f]

    # ── Y-axis cap: keep the threshold always visible ─────────────────────────
    max_thr = max(thr_arr) if thr_arr else 30.0
    y_cap   = max_thr * 2.8   # Increased cap slightly to fit text labels

    # Clip outlier y-values to cap for display; store true value in customdata
    out_y_display = [min(y, y_cap * 0.90) for y in out_y]
    out_y_true    = out_y

    # ── Plotly figure ─────────────────────────────────────────────────────────
    fig = go.Figure()

    # ① Danger-zone shading above threshold (rendered first → background)
    if thr_arr:
        fig.add_trace(go.Scatter(
            x=ts_str + ts_str[::-1],
            y=[y_cap] * len(ts_str) + thr_arr[::-1],
            fill='toself',
            fillcolor='rgba(239,68,68,0.10)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='_danger_zone',
        ))

    # ② The Gap Line — ALL points so it visually spikes through the threshold
    fig.add_trace(go.Scatter(
        x=ts_str, y=dt_arr,
        mode='lines',
        name='Silence Gap (s)',
        line=dict(color=C['accent'], width=2),
        fill='tozeroy',
        fillcolor='rgba(96,165,250,0.18)',
        hovertemplate='<b>%{x}</b><br>Gap: %{y:.2f}s<extra></extra>',
    ))

    # ③ Threshold line — bold, labelled
    fig.add_trace(go.Scatter(
        x=ts_str, y=thr_arr,
        mode='lines',
        name='Hourly Threshold',
        line=dict(color='#34d399', width=2.5, dash='dot'),
        hovertemplate='Threshold: %{y:.2f}s<extra></extra>',
    ))

    # ④ Outlier markers — clipped to cap, real value in hover
    if out_x:
        fig.add_trace(go.Scatter(
            x=out_x,
            y=out_y_display,
            customdata=[[v] for v in out_y_true],
            mode='markers+text',
            name='SILENT GAP ⚠',
            marker=dict(
                size=16,
                color=C['red'],
                symbol='triangle-up',
                line=dict(color='white', width=2),
            ),
            text=[f'{v:.0f}s' for v in out_y_true],
            textposition='top center',
            textfont=dict(color=C['red'], size=12, family=FONT, weight='bold'),
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Gap: <b>%{customdata[0]:.2f}s</b><br>'
                'Threshold exceeded!<extra></extra>'
            ),
        ))

    # ── Title with current simulated time ─────────────────────────────────────
    sim_cur      = _dt(2023, 10, 31) + _td(milliseconds=store['sim_ms'])
    sim_time_str = sim_cur.strftime('%H:%M:%S')

    _apply_layout(fig,
        title=dict(
            text=(
                f'Live Switch Monitor  |  Simulated: <b>{sim_time_str}</b>  '
                f'|  Threshold: <b>{max_thr:.1f}s</b>  |  (45× speed)'
            ),
            font=dict(size=13),
        ),
        xaxis_title='Simulated Timestamp (HH:MM:SS)',
        yaxis_title='Silence Gap (seconds)',
        height=440,
        hovermode='x unified',
        showlegend=True,
        yaxis=dict(range=[0, y_cap]),
        legend=dict(orientation='h', y=-0.20),
    )
    # X-axis: thin out tick labels so they don't crowd
    fig.update_xaxes(nticks=12)


    # ── Status banner ─────────────────────────────────────────────────────────
    is_down = bool(flag_arr and flag_arr[-1])   # last transaction was an outlier
    if is_down:
        banner_text  = "SILENT GAP DETECTED  |  Switch may be DOWN"
        banner_color = f"linear-gradient(135deg, {C['red']}33, {C['card']})"
        text_color   = C['red']
        icon         = "🔴"
    else:
        banner_text  = "OPERATIONAL  |  All transactions within threshold"
        banner_color = f"linear-gradient(135deg, {C['green']}22, {C['card']})"
        text_color   = C['green']
        icon         = "🟢"

    banner_style = {
        "padding": "14px 24px", "borderRadius": "14px",
        "marginBottom": "18px", "textAlign": "center",
        "fontSize": "16px", "fontWeight": "700", "letterSpacing": "1px",
        "background": banner_color,
        "border":    f"1px solid {text_color}44",
        "color":     text_color,
        "transition": "all 0.4s ease",
    }
    banner_children = [
        html.Span(icon + "  " + banner_text),
    ]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    cur_h      = (store['sim_ms'] // 3_600_000) % 24
    cur_thresh = _compute_threshold(cur_h, profile, cfg)
    last_gap   = dt_arr[-1] if dt_arr else 0.0
    uptime_pct = 100.0 * (1 - store['total_outliers'] / max(store['total_txn'], 1))

    def kpi_card(label, value, color, icon):
        return html.Div([
            html.Div(icon, style={"fontSize": "22px", "marginBottom": "6px"}),
            html.Div(label, style={
                "fontSize": "10px", "textTransform": "uppercase",
                "letterSpacing": "1px", "color": C["muted"], "marginBottom": "4px",
            }),
            html.Div(value, style={
                "fontSize": "22px", "fontWeight": "800", "color": color,
            }),
        ], style={
            "background": C["card2"], "borderRadius": "12px",
            "padding": "14px 16px", "textAlign": "center",
            "border": f"1px solid {C['border']}",
        })

    kpis = [
        kpi_card("Simulated Time",    sim_time_str,
                 C["accent"], "⏱️"),
        kpi_card("Last Gap (s)",      f"{last_gap:.2f}s",
                 C["red"] if is_down else C["text"], "📶"),
        kpi_card("Hour Threshold",    f"{cur_thresh:.2f}s",
                 C["green"], "📏"),
        kpi_card("Session Uptime",    f"{uptime_pct:.3f}%",
                 C["green"] if uptime_pct >= 99.9 else C["yellow"], "📊"),
    ]

    # ── Outage log table ──────────────────────────────────────────────────────
    events = list(reversed(store['outage_events']))   # newest first
    if events:
        log_rows = []
        for ev in events:
            sev_color = C['red'] if ev['x_over'] >= 3 else C['yellow']
            log_rows.append(html.Tr([
                html.Td(ev['time'],            style={"padding": "8px 12px", "color": C["muted"]}),
                html.Td(f"Hour {ev['hour']:02d}",  style={"padding": "8px 12px"}),
                html.Td(f"{ev['gap_s']:.2f}s",     style={"padding": "8px 12px", "color": C["red"], "fontWeight": "700"}),
                html.Td(f"{ev['thresh']:.2f}s",    style={"padding": "8px 12px", "color": C["green"]}),
                html.Td(f"{ev['x_over']}x",        style={"padding": "8px 12px", "color": sev_color, "fontWeight": "700"}),
            ], style={"borderBottom": f"1px solid {C['border']}"}))

        log_table = html.Table([
            html.Thead(html.Tr([
                html.Th(col, style={
                    "padding": "10px 12px", "textAlign": "left",
                    "color": C["accent"], "fontSize": "11px",
                    "textTransform": "uppercase", "letterSpacing": "0.5px",
                    "background": C["card2"],
                })
                for col in ["Sim Time", "Hour", "Gap", "Threshold", "Excess"]
            ])),
            html.Tbody(log_rows),
        ], style={
            "width": "100%", "borderCollapse": "collapse",
            "background": C["card"], "borderRadius": "12px",
            "overflow": "hidden",
        })
    else:
        log_table = html.P("No silent gaps detected yet in this session.",
                           style={"color": C["muted"], "fontStyle": "italic"})

    return store, fig, banner_children, banner_style, kpis, log_table


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Switch Availability Dashboard")
    print("  http://127.0.0.1:8050")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=8050)
