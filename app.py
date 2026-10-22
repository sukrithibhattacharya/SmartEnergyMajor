# =============================================================================
# SmartEnergy - Dashboard (website-style layout)
#   normal mode : streamlit run app.py
#   paper mode  : http://localhost:8501/?paper=1&state=Tamil%20Nadu&date=2024-03-15
#                 (no controls, no tabs: every section stacked, for screenshots)
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

# ---------- palette (light, one accent per meaning) --------------------------
INK, MUTED, LINE, BG, PANEL = "#0f1f2e", "#5b6b7a", "#e1e6ec", "#f3f5f8", "#ffffff"
C_ENERGY, C_PEAK = "#0e9384", "#4f46e5"
C_ACTUAL, C_PERSIST = "#334155", "#94a3b8"
C_UP, C_DOWN, C_SEL = "#ea580c", "#0284c7", "#f59e0b"
C_RESERVE, C_CARBON, C_BAD = "#0e9384", "#16a34a", "#dc2626"
FONT = "Instrument Sans, Segoe UI, Arial, sans-serif"
PLOT_CFG = {"displayModeBar": False}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, div, span, label, button {{ font-family:'Instrument Sans','Segoe UI',Arial,sans-serif; }}
.stApp {{ background:{BG}; color:{INK}; }}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"], footer, #MainMenu {{ display:none !important; }}
.block-container {{ max-width:1320px; padding:0 2rem 3rem 2rem; }}
div[data-testid="stVerticalBlock"] {{ gap:1.4rem; }}
label, [data-testid="stWidgetLabel"] p {{ color:{MUTED} !important; font-size:13px !important; font-weight:500; }}

/* ---- top bar ---- */
[class*="st-key-topbar"] {{ background:{PANEL}; border-bottom:1px solid {LINE};
  margin:0 -2rem; padding:14px 2rem 10px 2rem; position:sticky; top:0; z-index:100; }}
.brand {{ display:flex; align-items:center; gap:12px; padding-top:6px; }}
.brand .mark {{ width:38px; height:38px; border-radius:10px; background:{INK}; color:#fff;
  display:flex; align-items:center; justify-content:center; font-size:19px; }}
.brand .name {{ font-size:20px; font-weight:700; letter-spacing:-.3px; line-height:1.1; }}
.brand .tag {{ font-size:12.5px; color:{MUTED}; }}

/* ---- hero decision ---- */
.dec {{ background:{PANEL}; border:1px solid {LINE}; border-left-width:6px; border-radius:12px;
  padding:22px 28px; }}
.dec-go {{ border-left-color:{C_ENERGY}; }}
.dec-no {{ border-left-color:{C_BAD}; }}
.dec .kick {{ font-size:13px; font-weight:600; color:{MUTED}; margin-bottom:6px; }}
.dec .big {{ font-size:25px; line-height:1.35; font-weight:600; letter-spacing:-.4px; max-width:1000px; }}
.dec .big b {{ font-weight:700; }}
.dec .note {{ font-size:13.5px; color:{MUTED}; margin-top:8px; }}

/* ---- KPI row ---- */
.strip {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
.cell {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:18px 22px;
  border-top:3px solid var(--c); }}
.kl {{ font-size:13.5px; font-weight:600; color:{MUTED}; }}
.kv {{ font-size:38px; font-weight:700; line-height:1.15; margin-top:6px; font-variant-numeric:tabular-nums; letter-spacing:-1px; }}
.ku {{ font-size:15px; font-weight:500; color:{MUTED}; letter-spacing:0; }}
.ks {{ font-size:13px; color:{MUTED}; margin-top:8px; line-height:1.55; }}
.ok {{ color:{C_CARBON}; font-weight:700; }} .bad {{ color:{C_BAD}; font-weight:700; }}

/* ---- tabs ---- */
[data-testid="stTabs"] [role="tablist"] {{ gap:6px; border-bottom:1px solid {LINE}; }}
[data-testid="stTabs"] button[role="tab"] {{ padding:12px 18px; height:auto; }}
[data-testid="stTabs"] button[role="tab"] p {{ font-size:15.5px; font-weight:600; color:{MUTED}; }}
[data-testid="stTabs"] button[aria-selected="true"] p {{ color:{INK}; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background:{INK}; height:3px; }}

/* ---- panels & section text ---- */
[class*="st-key-card_"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:20px 24px 10px 24px; }}
.sec {{ font-size:22px; font-weight:700; letter-spacing:-.4px; margin:6px 0 0 0; }}
.secsub {{ font-size:14.5px; color:{MUTED}; margin:2px 0 4px 0; max-width:780px; }}
.pt {{ font-size:17px; font-weight:700; margin:0; letter-spacing:-.2px; }}
.ps {{ font-size:13px; color:{MUTED}; margin:2px 0 8px 0; }}
.foot {{ font-size:12.5px; color:{MUTED}; border-top:1px solid {LINE}; padding-top:14px; max-width:900px; }}

@media (max-width:900px) {{ .strip {{ grid-template-columns:repeat(2,1fr); }} .dec .big {{ font-size:20px; }} }}
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
    return 'post_monsoon'


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


# ---------- top bar: brand + controls (controls hidden in paper mode) --------
qp = st.query_params
paper = str(qp.get('paper', '0')) == '1'
default_state = qp.get('state', 'Tamil Nadu')
if default_state not in STATES:
    default_state = 'Tamil Nadu' if 'Tamil Nadu' in STATES else STATES[0]
default_date = TEST_DATES[len(TEST_DATES) // 2]
try:
    qd = pd.Timestamp(qp.get('date')).date()
    if qd in TEST_DATES: default_date = qd
except Exception:
    pass

BRAND = ('<div class="brand"><div class="mark">⚡</div><div><div class="name">SmartEnergy</div>'
         '<div class="tag">Day-ahead demand, evening reserve and carbon impact</div></div></div>')

with st.container(key="topbar"):
    if paper:
        state, sel_date = default_state, default_date
        st.markdown(BRAND, unsafe_allow_html=True)
    else:
        t1, t2, t3 = st.columns([1.6, 1, 2.4], gap="large")
        t1.markdown(BRAND, unsafe_allow_html=True)
        state = t2.selectbox("State", STATES, index=STATES.index(default_state))
        sel_date = t3.select_slider("Date (test period)", options=TEST_DATES, value=default_date)

sel_date = pd.Timestamp(sel_date)
season = month_to_season(sel_date.month)

# ---------- data for the day -------------------------------------------------
e = predict_target(state, sel_date, 'energy')
p = predict_target(state, sel_date, 'peak')
p4 = art['p4']
r4 = p4[(p4['state'] == state) & (p4['date'] == sel_date)]
r4 = r4.iloc[0] if len(r4) else None
recent = p4[(p4['state'] == state) & (p4['date'] <= sel_date)].sort_values('date').tail(30)
ctx = f"{state}, {sel_date:%d %b %Y}, {season.replace('_', ' ')} season"


# ---------- hero: today's decision ------------------------------------------
if r4 is not None:
    wm, wp = r4['width_model_MW'], r4['width_persist_MW']
    if wm < wp:
        st.markdown(
            f'<div class="dec dec-go"><div class="kick">Recommendation for {ctx}</div>'
            f'<div class="big">Schedule <b>{r4["reserve_model_MW"]:,.0f} MW</b> of evening spinning reserve '
            f'for hours {WIN0}-{WIN1}. The 80% interval is <b>{wp - wm:,.0f} MW narrower</b> than persistence, '
            f'avoiding about <b>{r4["carbon_saved_t"]:,.1f} t CO₂</b> today.</div>'
            f'<div class="note">The model reserve is lower than the persistence baseline on this date.</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="dec dec-no"><div class="kick">Recommendation for {ctx}</div>'
            f'<div class="big">Use the persistence reserve of <b>{r4["reserve_persist_MW"]:,.0f} MW</b>. '
            f'The model interval is <b>{wm - wp:,.0f} MW wider</b>, so flag this date for review.</div>'
            f'<div class="note">The model reserve is not tighter than persistence on this date.</div></div>',
            unsafe_allow_html=True)
else:
    st.markdown(f'<div class="dec"><div class="kick">{ctx}</div>'
                f'<div class="big">No reserve record exists for this date.</div></div>', unsafe_allow_html=True)


# ---------- KPI row ----------------------------------------------------------
def cell(label, value, unit, sub, color):
    return (f'<div class="cell" style="--c:{color}"><div class="kl">{label}</div>'
            f'<div class="kv">{value}<span class="ku"> {unit}</span></div>'
            f'<div class="ks">{sub}</div></div>')


def forecast_sub(preds, actual, fmt):
    sub = f"80% interval {fmt.format(preds['p10_cal'])} to {fmt.format(preds['p90_cal'])}"
    if actual is not None:
        inside = preds['p10_cal'] <= actual <= preds['p90_cal']
        mark = '<span class="ok">inside ✓</span>' if inside else '<span class="bad">outside ✗</span>'
        sub += f"<br>Actual {fmt.format(actual)}, {mark}"
    return sub


cells = [
    cell("Energy forecast (P50)", f"{e[0]['p50']:.1f}", "MU", forecast_sub(e[0], e[1], "{:.1f}"), C_ENERGY)
    if e else cell("Energy forecast (P50)", "-", "", "No forecast for this date", C_ENERGY),
    cell("Peak forecast (P50)", f"{p[0]['p50']:,.0f}", "MW", forecast_sub(p[0], p[1], "{:,.0f}"), C_PEAK)
    if p else cell("Peak forecast (P50)", "-", "", "No forecast for this date", C_PEAK),
]
if r4 is not None:
    rm, rp = r4['reserve_model_MW'], r4['reserve_persist_MW']
    pct = (rp - rm) / rp * 100
    cells.append(cell("Evening reserve (model)", f"{rm:,.0f}", "MW",
                      f"Persistence {rp:,.0f} MW<br>{'−' if pct >= 0 else '+'}{abs(pct):.0f}% vs persistence", C_RESERVE))
    cells.append(cell("CO₂ avoided (scenario)", f"{r4['carbon_saved_t']:,.1f}", "t",
                      f"Carbon intensity {r4['ci_reserve_kg_per_kwh']:.3f} kg/kWh<br>at hours {WIN0}-{WIN1}", C_CARBON))
else:
    cells.append(cell("Evening reserve (model)", "-", "", "No reserve record", C_RESERVE))
    cells.append(cell("CO₂ avoided (scenario)", "-", "", "No reserve record", C_CARBON))
st.markdown(f'<div class="strip">{"".join(cells)}</div>', unsafe_allow_html=True)


# ---------- chart helpers ----------------------------------------------------
def section(title, sub):
    st.markdown(f'<div class="sec">{title}</div><div class="secsub">{sub}</div>', unsafe_allow_html=True)


def panel_title(title, sub=""):
    st.markdown(f'<div class="pt">{title}</div><div class="ps">{sub}</div>', unsafe_allow_html=True)


def style(fig, height, legend=True):
    fig.update_layout(
        height=height, template="plotly_white", hovermode="x unified",
        font=dict(family=FONT, size=13, color=INK),
        margin=dict(l=4, r=4, t=6, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LINE, font=dict(color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12.5)))
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(size=12, color=MUTED))
    fig.update_yaxes(gridcolor="#eef1f5", zeroline=False, tickfont=dict(size=12, color=MUTED))
    return fig


def forecast_figure(win, color, ylabel):
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
    style(fig, 340)
    fig.update_yaxes(title_text=ylabel, title_font=dict(size=12, color=MUTED))
    return fig


def shap_figure(c, title):
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
    style(fig, 380, legend=False)
    fig.update_layout(hovermode='closest', margin=dict(l=4, r=40, t=6, b=4))
    return fig


def inside_pct(win):
    w = win.dropna(subset=['actual'])
    if w.empty: return float('nan')
    return float(((w['actual'] >= w['p10_cal']) & (w['actual'] <= w['p90_cal'])).mean() * 100)


# ---------- sections ---------------------------------------------------------
def render_forecast():
    section("Demand forecast",
            "Median forecast with a calibrated 80% interval for the 60 days up to the selected date.")
    a1, a2 = st.columns(2, gap="medium")
    for col, key, target, color, title, ylab in [
        (a1, "card_energy", 'energy', C_ENERGY, "Energy (MU)", "Energy (MU)"),
        (a2, "card_peak", 'peak', C_PEAK, "Peak demand (MW)", "Peak (MW)")]:
        with col:
            with st.container(key=key):
                w = forecast_window(state, sel_date, target)
                if w.empty:
                    st.warning(f"No {target} forecast window for this date.")
                else:
                    panel_title(title, f"{inside_pct(w):.0f}% of actual values fall inside the interval")
                    st.plotly_chart(forecast_figure(w, color, ylab), width='stretch',
                                    theme=None, config=PLOT_CFG)


def render_reserve():
    section("Reserve and carbon",
            f"How much evening spinning reserve (hours {WIN0}-{WIN1}) the model needs compared with "
            "persistence, and the CO₂ this avoids. Last 30 days.")
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        with st.container(key="card_reserve"):
            if recent.empty:
                st.warning("No reserve records.")
            else:
                red = (1 - recent['reserve_model_MW'].mean() / recent['reserve_persist_MW'].mean()) * 100
                panel_title("Evening reserve margin (MW)",
                            f"Model needs {red:.0f}% less reserve than persistence on average")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent['date'], y=recent['reserve_persist_MW'],
                                         name='Persistence', line=dict(color=C_PERSIST, width=2.5)))
                fig.add_trace(go.Scatter(x=recent['date'], y=recent['reserve_model_MW'],
                                         name='Model', line=dict(color=C_RESERVE, width=3),
                                         fill='tonexty', fillcolor=rgba(C_RESERVE, 0.14)))
                style(fig, 340)
                fig.update_yaxes(title_text="Reserve (MW)", title_font=dict(size=12, color=MUTED))
                st.plotly_chart(fig, width='stretch', theme=None, config=PLOT_CFG)
    with b2:
        with st.container(key="card_carbon"):
            if recent.empty:
                st.warning("No carbon records.")
            else:
                panel_title("CO₂ avoided per day (t)",
                            f"{recent['carbon_saved_t'].sum():,.0f} t in total, scenario estimate")
                fig = go.Figure(go.Bar(
                    x=recent['date'], y=recent['carbon_saved_t'],
                    marker_color=[C_CARBON if v >= 0 else C_BAD for v in recent['carbon_saved_t']],
                    name='CO₂ avoided'))
                style(fig, 340, legend=False)
                fig.update_layout(bargap=0.25)
                fig.update_yaxes(title_text="t CO₂ per day", title_font=dict(size=12, color=MUTED))
                st.plotly_chart(fig, width='stretch', theme=None, config=PLOT_CFG)


def render_explain():
    section("Why this forecast?",
            f"The 8 strongest drivers on {sel_date:%d %b %Y}. "
            f"<span style='color:{C_UP}'>■</span> raises the forecast, "
            f"<span style='color:{C_DOWN}'>■</span> lowers it.")
    s1, s2 = st.columns(2, gap="medium")
    for col, key, target, title in [(s1, "card_shap_energy", 'energy', "Energy forecast"),
                                    (s2, "card_shap_peak", 'peak', "Peak forecast")]:
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
                    st.plotly_chart(shap_figure(c, title), width='stretch', theme=None, config=PLOT_CFG)


# ---------- layout: tabs normally, stacked in paper mode ---------------------
if paper:
    render_forecast()
    render_reserve()
    render_explain()
else:
    tab1, tab2, tab3 = st.tabs(["Demand forecast", "Reserve and carbon", "Why this forecast"])
    with tab1:
        render_forecast()
    with tab2:
        render_reserve()
    with tab3:
        render_explain()

st.markdown(
    '<div class="foot">XGBoost forecasts with calibrated 80% intervals. Carbon values are scenario '
    'estimates under spinning-reserve and grid-average carbon-intensity assumptions, '
    'not measured emission reductions.</div>', unsafe_allow_html=True)