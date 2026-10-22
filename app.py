# =============================================================================
# SmartEnergy - Grid planning dashboard
#   run:  streamlit run app.py
# =============================================================================
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="SmartEnergy", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"

# ---------- palette ----------------------------------------------------------
NAVY, INK, MUTED, LINE, BG, PANEL = "#111936", "#151b33", "#667085", "#e3e6f0", "#eef0f7", "#ffffff"
C_ENERGY, C_PEAK = "#4338ca", "#f59e0b"
C_MODEL, C_PERSIST, C_ACTUAL = "#0d9488", "#9aa3b8", "#2b3350"
C_CARBON, C_BAD, C_SEL = "#16a34a", "#dc2626", "#f43f5e"
C_UP, C_DOWN = "#ea580c", "#2563eb"
DISPLAY = "Bricolage Grotesque, Segoe UI, sans-serif"
BODY = "DM Sans, Segoe UI, Arial, sans-serif"
PLOT_CFG = {"displayModeBar": False}
PAGES = ["Overview", "Forecast", "Reserve and carbon", "Explainability", "About"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, p, div, span, label, button, li, td, th {{ font-family:{BODY}; }}
.stApp {{ background:{BG}; color:{INK}; }}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"], footer, #MainMenu {{ display:none !important; }}
.block-container {{ max-width:1400px; padding:0 2.2rem 3rem 2.2rem; }}
div[data-testid="stVerticalBlock"] {{ gap:1.5rem; }}
label, [data-testid="stWidgetLabel"] p {{ color:{MUTED} !important; font-size:13px !important; font-weight:500; }}

/* ===== navigation bar ===== */
[class*="st-key-navbar"] {{ background:{NAVY}; margin:0 -2.2rem; padding:14px 2.2rem; }}
[class*="st-key-navbar"] div[data-testid="stVerticalBlock"] {{ gap:0; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.brand .mark {{ width:40px; height:40px; border-radius:11px; background:{C_PEAK}; color:{NAVY};
  display:flex; align-items:center; justify-content:center; font-size:21px; font-weight:800; }}
.brand .name {{ font-family:{DISPLAY}; font-size:23px; font-weight:800; color:#fff; letter-spacing:-.5px; }}
[class*="st-key-navbar"] [data-testid="stRadio"] > div {{ gap:4px; justify-content:flex-end; flex-wrap:nowrap; }}
[class*="st-key-navbar"] [data-testid="stRadio"] label {{ padding:9px 16px; border-radius:9px; cursor:pointer; margin:0; }}
[class*="st-key-navbar"] [data-testid="stRadio"] label > div:first-child {{ display:none; }}
[class*="st-key-navbar"] [data-testid="stRadio"] label p {{ color:#aab3d1 !important; font-size:15px !important; font-weight:600; }}
[class*="st-key-navbar"] [data-testid="stRadio"] label:hover p {{ color:#fff !important; }}
[class*="st-key-navbar"] [data-testid="stRadio"] label:has(input:checked) {{ background:rgba(255,255,255,.12); }}
[class*="st-key-navbar"] [data-testid="stRadio"] label:has(input:checked) p {{ color:#fff !important; }}

/* ===== filter bar ===== */
[class*="st-key-filters"] {{ background:{PANEL}; margin:0 -2.2rem; padding:12px 2.2rem 14px 2.2rem;
  border-bottom:1px solid {LINE}; }}

/* ===== hero ===== */
.hero {{ background:{PANEL}; border:1px solid {LINE}; border-radius:20px; padding:34px 38px; height:100%; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:18px; }}
.chip {{ background:{BG}; border-radius:999px; padding:6px 14px; font-size:13.5px; font-weight:600; color:{INK}; }}
.chip.go {{ background:#d9f5ef; color:#0b6b61; }}
.chip.no {{ background:#fde4e4; color:#a61b1b; }}
.hero h1 {{ font-family:{DISPLAY}; font-size:46px; line-height:1.08; font-weight:800; letter-spacing:-1.6px; margin:0 0 16px 0; }}
.hero p {{ font-size:17px; line-height:1.6; color:{MUTED}; margin:0; max-width:560px; }}
.hero p b {{ color:{INK}; font-weight:700; }}
[class*="st-key-herochart"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:20px; padding:22px 26px 8px 26px; }}

/* ===== KPI tiles ===== */
.strip {{ display:grid; grid-template-columns:repeat(4,1fr); gap:18px; }}
.cell {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:20px 24px; }}
.cell .bar {{ width:34px; height:5px; border-radius:3px; background:var(--c); margin-bottom:14px; }}
.kl {{ font-size:14px; font-weight:600; color:{MUTED}; }}
.kv {{ font-family:{DISPLAY}; font-size:40px; font-weight:800; line-height:1.1; margin-top:6px; letter-spacing:-1.2px; font-variant-numeric:tabular-nums; }}
.ku {{ font-family:{BODY}; font-size:15px; font-weight:500; color:{MUTED}; letter-spacing:0; }}
.ks {{ font-size:13.5px; color:{MUTED}; margin-top:8px; line-height:1.55; }}
.ok {{ color:{C_CARBON}; font-weight:700; }} .bad {{ color:{C_BAD}; font-weight:700; }}

/* ===== page headings and cards ===== */
.ph {{ font-family:{DISPLAY}; font-size:34px; font-weight:800; letter-spacing:-1.1px; margin:10px 0 0 0; }}
.pd {{ font-size:16px; color:{MUTED}; margin:4px 0 0 0; max-width:760px; line-height:1.55; }}
[class*="st-key-card_"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:22px 26px 10px 26px; }}
.pt {{ font-family:{DISPLAY}; font-size:20px; font-weight:700; margin:0; letter-spacing:-.4px; }}
.ps {{ font-size:13.5px; color:{MUTED}; margin:3px 0 10px 0; }}

/* ===== about ===== */
.steps {{ display:grid; grid-template-columns:repeat(4,1fr); gap:18px; }}
.step {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:24px; }}
.step h4 {{ font-family:{DISPLAY}; font-size:19px; font-weight:700; margin:0 0 8px 0; letter-spacing:-.3px; }}
.step p {{ font-size:14.5px; line-height:1.6; color:{MUTED}; margin:0; }}
.note {{ background:#fff7e6; border:1px solid #fde0a6; border-radius:14px; padding:18px 24px; font-size:15px; line-height:1.6; max-width:900px; }}

@media (max-width:1000px) {{ .strip, .steps {{ grid-template-columns:repeat(2,1fr); }} .hero h1 {{ font-size:34px; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- pre-flight -------------------------------------------------------
def check_files():
    need = [OUT / f for f in [
        'phase1/xgb_energy.json', 'phase1/xgb_peak.json',
        'phase1/xgb_energy_features.json', 'phase1/xgb_peak_features.json',
        'phase2/quantile_energy_q10.json', 'phase2/quantile_energy_q50.json',
        'phase2/quantile_energy_q90.json', 'phase2/quantile_peak_q10.json',
        'phase2/quantile_peak_q50.json', 'phase2/quantile_peak_q90.json',
        'phase1/df_state.parquet', 'phase2/conformal_calibration.json',
        'phase3/carbon_intensity_hourly.csv', 'phase3/phase3_meta.json',
        'phase4/carbon_savings_per_day.csv', 'phase4/phase4_meta.json']]
    return [str(p.relative_to(ROOT)) for p in need if not p.exists()]


missing = check_files()
if missing:
    st.error("### Missing files\n\nThe dashboard cannot start. Place these inside "
             "your `outputs/` folder:\n\n- " + "\n- ".join(missing))
    st.stop()


# ---------- loaders ----------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and data...")
def load_artifacts():
    art = {}
    for target in ['energy', 'peak']:
        m = xgb.XGBRegressor()
        m.load_model(str(OUT / f'phase1/xgb_{target}.json'))
        with open(OUT / f'phase1/xgb_{target}_features.json') as f:
            feats = json.load(f)
        art[f'xgb_{target}'] = {'model': m, 'features': feats}
    for target in ['energy', 'peak']:
        models = {}
        for q, tag in [(0.1, 10), (0.5, 50), (0.9, 90)]:
            m = xgb.XGBRegressor()
            m.load_model(str(OUT / f'phase2/quantile_{target}_q{tag}.json'))
            models[q] = m
        art[f'q_{target}'] = {'models': models}
    with open(OUT / 'phase2/conformal_calibration.json') as f:
        art['cal'] = json.load(f)
    art['df'] = pd.read_parquet(OUT / 'phase1/df_state.parquet')
    art['FEATURES'] = art['xgb_energy']['features']
    art['ci'] = pd.read_csv(OUT / 'phase3/carbon_intensity_hourly.csv')
    with open(OUT / 'phase3/phase3_meta.json') as f:
        art['p3_meta'] = json.load(f)
    art['p4'] = pd.read_csv(OUT / 'phase4/carbon_savings_per_day.csv', parse_dates=['date'])
    with open(OUT / 'phase4/phase4_meta.json') as f:
        art['p4_meta'] = json.load(f)
    return art


@st.cache_resource(show_spinner=False)
def get_explainer(target):
    return shap.TreeExplainer(load_artifacts()[f'xgb_{target}']['model'])


art = load_artifacts()
df = art['df']
FEATURES = art['FEATURES']
P4META = art['p4_meta']
WIN0, WIN1 = P4META['reserve_window_hours'][0], P4META['reserve_window_hours'][1]
STATES = sorted(df['state'].unique())
TEST_DATES = sorted(df[df['split'] == 'test']['date'].dt.date.unique())
ANCHOR = {'energy': 'energy_lag1', 'peak': 'peak_lag1'}
ACTUAL = {'energy': 'energy_MU', 'peak': 'peak_MW'}


def month_to_season(m):
    if m in (12, 1, 2): return 'winter'
    if m in (3, 4, 5): return 'summer'
    if m in (6, 7, 8, 9): return 'monsoon'
    return 'post-monsoon'


def _row(state, date):
    r = df[(df['state'] == state) & (df['date'] == date)]
    return r.iloc[0] if len(r) else None


def predict_target(state, date, target):
    row = _row(state, date)
    if row is None: return None
    anchor = row[ANCHOR[target]]
    if pd.isna(anchor) or anchor <= 0: return None
    X = row[FEATURES].astype('float32').to_frame().T
    qb = art['q_energy'] if target == 'energy' else art['q_peak']
    preds = {}
    for q, key in [(0.1, 'p10'), (0.5, 'p50'), (0.9, 'p90')]:
        preds[key] = float(anchor * np.exp(qb['models'][q].predict(X)[0]))
    cal = art['cal'].get(target, {}).get(state, [0.0, 0.0])
    preds['p10_cal'] = preds['p10'] - cal[0]
    preds['p90_cal'] = preds['p90'] + cal[1]
    preds['width'] = preds['p90_cal'] - preds['p10_cal']
    a = row[ACTUAL[target]]
    return preds, (float(a) if not pd.isna(a) else None)


@st.cache_data(show_spinner=False)
def forecast_window(state, end_date, target, n_days=60):
    g = (df[(df['state'] == state) & (df['date'] <= end_date)].sort_values('date').tail(n_days))
    g = g[g[ANCHOR[target]].notna() & (g[ANCHOR[target]] > 0)]
    if g.empty: return pd.DataFrame()
    X = g[FEATURES].astype('float32')
    qb = art['q_energy'] if target == 'energy' else art['q_peak']
    anchor = g[ANCHOR[target]].values
    out = pd.DataFrame({'date': g['date'].values})
    for q, key in [(0.1, 'p10'), (0.5, 'p50'), (0.9, 'p90')]:
        out[key] = anchor * np.exp(qb['models'][q].predict(X))
    cal = art['cal'].get(target, {}).get(state, [0.0, 0.0])
    out['p10_cal'] = out['p10'] - cal[0]
    out['p90_cal'] = out['p90'] + cal[1]
    out['actual'] = g[ACTUAL[target]].values
    return out


def shap_attribution(state, date, target):
    row = _row(state, date)
    if row is None: return None
    X = row[FEATURES].astype('float32').to_frame().T
    sv = get_explainer(target).shap_values(X)
    if isinstance(sv, list): sv = sv[0]
    s = pd.Series(sv[0], index=FEATURES)
    return s.reindex(s.abs().sort_values(ascending=False).index)


NAMES = {
    '2m_temperature_min': 'Temp min', '2m_temperature_mean': 'Temp mean',
    '2m_temperature_max': 'Temp max', '2m_dewpoint_temperature_mean': 'Dew point',
    'wind_speed': 'Wind speed', 'surface_solar_radiation_downwards_mean': 'Solar radiation',
    'total_cloud_cover_mean': 'Cloud cover', 'utci_mean': 'UTCI', 'CDD': 'Cooling deg-days',
    'HDD': 'Heating deg-days', 'diurnal_range': 'Temp range', 'RH': 'Rel. humidity',
    't_roll7': 'Temp 7d avg', 't_anom30': 'Temp anomaly 30d', 'cdd_roll7': 'CDD 7d avg',
    'dow': 'Day of week', 'is_weekend': 'Weekend', 'month': 'Month',
    'doy_sin': 'Day-of-year (sin)', 'doy_cos': 'Day-of-year (cos)',
    'dow_sin': 'Day-of-week (sin)', 'dow_cos': 'Day-of-week (cos)', 'is_holiday': 'Public holiday',
}


def pretty(f):
    if f in NAMES: return NAMES[f]
    m = re.fullmatch(r'(energy|peak|unmet)_lag(\d+)', f)
    if m: return f"{m.group(1).capitalize()} lag {m.group(2)}d"
    return f.replace('_', ' ').strip().capitalize()


def rgba(hex_color, a):
    h = hex_color.lstrip('#')
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


# ---------- navigation + filters ---------------------------------------------
with st.container(key="navbar"):
    n1, n2 = st.columns([1.3, 3], gap="large", vertical_alignment="center")
    n1.markdown('<div class="brand"><div class="mark">⚡</div><div class="name">SmartEnergy</div></div>',
                unsafe_allow_html=True)
    page = n2.radio("Page", PAGES, horizontal=True, label_visibility="collapsed", key="page")

default_state = 'Tamil Nadu' if 'Tamil Nadu' in STATES else STATES[0]
if page != "About":
    with st.container(key="filters"):
        f1, f2 = st.columns([1, 3], gap="large")
        state = f1.selectbox("State", STATES, index=STATES.index(default_state), key="state")
        sel_date = f2.select_slider("Date in the test period", options=TEST_DATES,
                                    value=TEST_DATES[len(TEST_DATES) // 2], key="date")
else:
    state, sel_date = default_state, TEST_DATES[len(TEST_DATES) // 2]

sel_date = pd.Timestamp(sel_date)
season = month_to_season(sel_date.month)

# ---------- data for the selected day ----------------------------------------
e = predict_target(state, sel_date, 'energy')
p = predict_target(state, sel_date, 'peak')
p4 = art['p4']
r4 = p4[(p4['state'] == state) & (p4['date'] == sel_date)]
r4 = r4.iloc[0] if len(r4) else None
recent = p4[(p4['state'] == state) & (p4['date'] <= sel_date)].sort_values('date').tail(30)


# ---------- helpers ----------------------------------------------------------
def page_head(title, desc):
    st.markdown(f'<div class="ph">{title}</div><div class="pd">{desc}</div>', unsafe_allow_html=True)


def panel_title(title, sub=""):
    st.markdown(f'<div class="pt">{title}</div><div class="ps">{sub}</div>', unsafe_allow_html=True)


def cell(label, value, unit, sub, color):
    return (f'<div class="cell" style="--c:{color}"><div class="bar"></div><div class="kl">{label}</div>'
            f'<div class="kv">{value}<span class="ku"> {unit}</span></div><div class="ks">{sub}</div></div>')


def stat_strip(items):
    st.markdown(f'<div class="strip">{"".join(cell(*i) for i in items)}</div>', unsafe_allow_html=True)


def forecast_sub(preds, actual, fmt):
    sub = f"80% interval {fmt.format(preds['p10_cal'])} to {fmt.format(preds['p90_cal'])}"
    if actual is not None:
        inside = preds['p10_cal'] <= actual <= preds['p90_cal']
        mark = '<span class="ok">inside ✓</span>' if inside else '<span class="bad">outside ✗</span>'
        sub += f"<br>Actual {fmt.format(actual)}, {mark}"
    return sub


def style(fig, height, legend=True):
    fig.update_layout(
        height=height, template="plotly_white", hovermode="x unified",
        font=dict(family=BODY, size=13, color=INK), margin=dict(l=4, r=4, t=6, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LINE, font=dict(color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12.5)))
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(size=12, color=MUTED))
    fig.update_yaxes(gridcolor="#eceff6", zeroline=False, tickfont=dict(size=12, color=MUTED))
    return fig


def forecast_figure(win, color, ylabel, height=340):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win['date'], y=win['p90_cal'], line=dict(width=0),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=win['date'], y=win['p10_cal'], fill='tonexty',
                             fillcolor=rgba(color, 0.16), line=dict(width=0),
                             name='80% interval', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=win['date'], y=win['actual'], name='Actual', mode='lines',
                             line=dict(color=C_ACTUAL, width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=win['date'], y=win['p50'], name='P50 forecast',
                             line=dict(color=color, width=3)))
    sel = win[win['date'] == sel_date]
    if len(sel):
        fig.add_vline(x=sel_date.to_pydatetime(), line=dict(color=C_SEL, width=1.5, dash='dot'))
        fig.add_trace(go.Scatter(x=sel['date'], y=sel['p50'], mode='markers', name='Selected day',
                                 marker=dict(size=12, color=C_SEL, symbol='diamond',
                                             line=dict(color='#ffffff', width=2))))
    style(fig, height)
    fig.update_yaxes(title_text=ylabel, title_font=dict(size=12, color=MUTED))
    return fig


def reserve_figure(height=340):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['date'], y=recent['reserve_persist_MW'],
                             name='Persistence', line=dict(color=C_PERSIST, width=2.5)))
    fig.add_trace(go.Scatter(x=recent['date'], y=recent['reserve_model_MW'],
                             name='Model', line=dict(color=C_MODEL, width=3),
                             fill='tonexty', fillcolor=rgba(C_MODEL, 0.14)))
    style(fig, height)
    fig.update_yaxes(title_text="Reserve (MW)", title_font=dict(size=12, color=MUTED))
    return fig


def carbon_figure(height=340):
    fig = go.Figure(go.Bar(
        x=recent['date'], y=recent['carbon_saved_t'],
        marker_color=[C_CARBON if v >= 0 else C_BAD for v in recent['carbon_saved_t']],
        name='CO₂ avoided'))
    style(fig, height, legend=False)
    fig.update_layout(bargap=0.25)
    fig.update_yaxes(title_text="t CO₂ per day", title_font=dict(size=12, color=MUTED))
    return fig


def shap_figure(c):
    top = c.head(8)[::-1]
    m = float(np.abs(top.values).max()) * 1.5 or 1.0
    fig = go.Figure(go.Bar(
        x=top.values, y=[pretty(f) for f in top.index], orientation='h',
        marker_color=[C_UP if v > 0 else C_DOWN for v in top.values],
        text=[f"{v:+.3f}" for v in top.values], textposition='outside',
        cliponaxis=False, textfont=dict(size=12), showlegend=False,
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:.5f}<extra></extra>'))
    fig.update_xaxes(range=[-m, m], zeroline=True, zerolinecolor="#94a3b8", zerolinewidth=1.5,
                     showgrid=False, showticklabels=False,
                     title_text="Contribution to log-ratio vs previous day",
                     title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(tickfont=dict(size=12.5, color=INK))
    style(fig, 400, legend=False)
    fig.update_layout(hovermode='closest', margin=dict(l=4, r=40, t=6, b=4))
    return fig


def inside_pct(win):
    w = win.dropna(subset=['actual'])
    if w.empty: return float('nan')
    return float(((w['actual'] >= w['p10_cal']) & (w['actual'] <= w['p90_cal'])).mean() * 100)


def show(fig):
    st.plotly_chart(fig, width='stretch', theme=None, config=PLOT_CFG)


def kpi_items():
    items = [
        ("Energy forecast (P50)", f"{e[0]['p50']:.1f}", "MU", forecast_sub(e[0], e[1], "{:.1f}"), C_ENERGY)
        if e else ("Energy forecast (P50)", "-", "", "No forecast for this date", C_ENERGY),
        ("Peak forecast (P50)", f"{p[0]['p50']:,.0f}", "MW", forecast_sub(p[0], p[1], "{:,.0f}"), C_PEAK)
        if p else ("Peak forecast (P50)", "-", "", "No forecast for this date", C_PEAK),
    ]
    if r4 is not None:
        rm, rp = r4['reserve_model_MW'], r4['reserve_persist_MW']
        pct = (rp - rm) / rp * 100
        items.append(("Evening reserve (model)", f"{rm:,.0f}", "MW",
                      f"Persistence {rp:,.0f} MW<br>{'−' if pct >= 0 else '+'}{abs(pct):.0f}% vs persistence", C_MODEL))
        items.append(("CO₂ avoided (scenario)", f"{r4['carbon_saved_t']:,.1f}", "t",
                      f"Carbon intensity {r4['ci_reserve_kg_per_kwh']:.3f} kg/kWh<br>at hours {WIN0}-{WIN1}", C_CARBON))
    else:
        items.append(("Evening reserve (model)", "-", "", "No reserve record", C_MODEL))
        items.append(("CO₂ avoided (scenario)", "-", "", "No reserve record", C_CARBON))
    return items


# =============================================================================
# PAGES
# =============================================================================
def page_overview():
    hero_l, hero_r = st.columns([5, 6], gap="medium")
    with hero_l:
        chips = (f'<span class="chip">📍 {state}</span><span class="chip">📅 {sel_date:%d %b %Y}</span>'
                 f'<span class="chip">{season.title()}</span>')
        if r4 is not None:
            wm, wp = r4['width_model_MW'], r4['width_persist_MW']
            if wm < wp:
                chips += '<span class="chip go">Model is tighter</span>'
                head = f'Schedule {r4["reserve_model_MW"]:,.0f} MW of evening reserve.'
                body = (f'Cover hours {WIN0}-{WIN1}. The 80% interval is <b>{wp - wm:,.0f} MW narrower</b> '
                        f'than persistence, which avoids about <b>{r4["carbon_saved_t"]:,.1f} t of CO₂</b> today.')
            else:
                chips += '<span class="chip no">Review this date</span>'
                head = f'Fall back to {r4["reserve_persist_MW"]:,.0f} MW persistence reserve.'
                body = (f'The model interval is <b>{wm - wp:,.0f} MW wider</b> than persistence on this date, '
                        f'so the model reserve is not the safer choice.')
        else:
            head, body = "No reserve plan for this date.", "Pick another date in the test period."
        st.markdown(f'<div class="hero"><div class="chips">{chips}</div><h1>{head}</h1><p>{body}</p></div>',
                    unsafe_allow_html=True)
    with hero_r:
        with st.container(key="herochart"):
            w = forecast_window(state, sel_date, 'energy')
            if w.empty:
                st.warning("No forecast window for this date.")
            else:
                panel_title("Energy forecast, last 60 days",
                            f"{inside_pct(w):.0f}% of actual values fell inside the 80% interval")
                show(forecast_figure(w, C_ENERGY, "Energy (MU)", height=290))

    stat_strip(kpi_items())

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(key="card_peak_home"):
            w = forecast_window(state, sel_date, 'peak')
            if w.empty:
                st.warning("No peak forecast window.")
            else:
                panel_title("Peak demand forecast (MW)", "Last 60 days")
                show(forecast_figure(w, C_PEAK, "Peak (MW)", height=320))
    with c2:
        with st.container(key="card_reserve_home"):
            if recent.empty:
                st.warning("No reserve records.")
            else:
                red = (1 - recent['reserve_model_MW'].mean() / recent['reserve_persist_MW'].mean()) * 100
                panel_title("Evening reserve (MW)", f"Last 30 days, model needs {red:.0f}% less than persistence")
                show(reserve_figure(height=320))


def page_forecast():
    page_head("Demand forecast",
              "Day-ahead energy and peak demand for the selected state. The shaded band is the calibrated "
              "80% interval, so about 8 in 10 actual values should land inside it.")
    for key, target, color, title, ylab in [
        ("card_f_energy", 'energy', C_ENERGY, "Energy demand (MU)", "Energy (MU)"),
        ("card_f_peak", 'peak', C_PEAK, "Peak demand (MW)", "Peak (MW)")]:
        with st.container(key=key):
            w = forecast_window(state, sel_date, target)
            if w.empty:
                st.warning(f"No {target} forecast window for this date.")
            else:
                panel_title(title, f"Last 60 days, {inside_pct(w):.0f}% of actual values inside the interval")
                show(forecast_figure(w, color, ylab, height=380))
    stat_strip(kpi_items()[:2])


def page_reserve():
    page_head("Reserve and carbon",
              f"How much evening spinning reserve (hours {WIN0}-{WIN1}) the model needs compared with the "
              "persistence baseline, and the CO₂ this avoids. Shows the 30 days up to the selected date.")
    if recent.empty:
        st.warning("No reserve records for this state and date.")
        return
    red = (1 - recent['reserve_model_MW'].mean() / recent['reserve_persist_MW'].mean()) * 100
    beat = (recent['reserve_model_MW'] < recent['reserve_persist_MW']).mean() * 100
    stat_strip([
        ("Average reserve cut", f"{red:.0f}", "%", "Model vs persistence, 30 days", C_MODEL),
        ("Days model was lower", f"{beat:.0f}", "%", f"{int(round(beat / 100 * len(recent)))} of {len(recent)} days", C_ENERGY),
        ("CO₂ avoided, total", f"{recent['carbon_saved_t'].sum():,.0f}", "t", "Scenario estimate", C_CARBON),
        ("CO₂ avoided, best day", f"{recent['carbon_saved_t'].max():,.1f}", "t",
         f"On {recent.loc[recent['carbon_saved_t'].idxmax(), 'date']:%d %b %Y}", C_PEAK),
    ])
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(key="card_r_reserve"):
            panel_title("Evening reserve margin (MW)", "Lower is better, as long as the interval still covers demand")
            show(reserve_figure())
    with c2:
        with st.container(key="card_r_carbon"):
            panel_title("CO₂ avoided per day (t)", "Red bars are days where the model needed more reserve")
            show(carbon_figure())
    with st.container(key="card_r_table"):
        panel_title("Daily records", "Most recent first")
        tbl = recent.sort_values('date', ascending=False)[[
            'date', 'reserve_model_MW', 'reserve_persist_MW', 'width_model_MW',
            'width_persist_MW', 'ci_reserve_kg_per_kwh', 'carbon_saved_t']]
        st.dataframe(tbl, hide_index=True, width='stretch', height=360, column_config={
            'date': st.column_config.DatetimeColumn("Date", format="DD MMM YYYY"),
            'reserve_model_MW': st.column_config.NumberColumn("Model reserve (MW)", format="%.0f"),
            'reserve_persist_MW': st.column_config.NumberColumn("Persistence reserve (MW)", format="%.0f"),
            'width_model_MW': st.column_config.NumberColumn("Model interval (MW)", format="%.0f"),
            'width_persist_MW': st.column_config.NumberColumn("Persistence interval (MW)", format="%.0f"),
            'ci_reserve_kg_per_kwh': st.column_config.NumberColumn("Carbon intensity (kg/kWh)", format="%.3f"),
            'carbon_saved_t': st.column_config.NumberColumn("CO₂ avoided (t)", format="%.1f")})


def page_explain():
    page_head("Explainability",
              f"The 8 features that moved the forecast most on {sel_date:%d %b %Y}. "
              "Orange bars push the forecast up, blue bars push it down, relative to the previous day.")
    s1, s2 = st.columns(2, gap="medium")
    for col, key, target, title in [(s1, "card_s_energy", 'energy', "Energy forecast"),
                                    (s2, "card_s_peak", 'peak', "Peak forecast")]:
        with col:
            with st.container(key=key):
                try:
                    c = shap_attribution(state, sel_date, target)
                except Exception:
                    c = None
                panel_title(title, "SHAP contribution per feature")
                if c is None:
                    st.warning("No SHAP attribution available for this date.")
                else:
                    show(shap_figure(c))


def page_about():
    page_head("About SmartEnergy",
              "A four-stage pipeline that turns weather and calendar data into a day-ahead reserve decision.")
    st.markdown(
        '<div class="steps">'
        '<div class="step"><h4>Point forecast</h4><p>XGBoost models predict energy (MU) and peak demand (MW) '
        'for each state as a ratio to the previous day.</p></div>'
        '<div class="step"><h4>Uncertainty</h4><p>Quantile models give P10, P50 and P90. A per-state conformal '
        'adjustment calibrates them into an 80% interval.</p></div>'
        '<div class="step"><h4>Carbon intensity</h4><p>Hourly grid carbon intensity is estimated for the '
        'evening window so reserve can be priced in CO₂.</p></div>'
        '<div class="step"><h4>Reserve decision</h4><p>The model interval sets the spinning reserve and is '
        'compared with a persistence baseline, day by day.</p></div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="note"><b>Read the carbon numbers as scenarios.</b> They rest on spinning-reserve and '
        'grid-average carbon-intensity assumptions. They are not measured emission reductions.</div>',
        unsafe_allow_html=True)


{"Overview": page_overview, "Forecast": page_forecast, "Reserve and carbon": page_reserve,
 "Explainability": page_explain, "About": page_about}[page]()