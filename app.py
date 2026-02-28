import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from config import PROFILES_PATH, YEARLY_KPI_PATH, DATASET_PATH
from engine.dna import (
    compute_similarity_weights,
    build_pure_dna,
    build_year_dataframe,
    build_dna_layers,
)
from engine.calibration import calibrate_base, build_projections
from views.dashboard import render_dashboard
from views.lab import render_lab
from views.chatbot import render_chatbot
from utils.export import build_excel_report

st.set_page_config(page_title="Tech Strategy Lab", layout="wide", page_icon="🧬")

# ─── CREDENTIALS ────────────────────────────────────────────────────────────────
_VALID_USER = "universitybox_2026"
_VALID_PASS = "99118822"
_SUPPORT_EMAIL = "parsahajiannezhad@universitybox.com"

# ─── AUTH GATE ──────────────────────────────────────────────────────────────────

if "_auth_ok" not in st.session_state:
    st.session_state._auth_ok = False

if not st.session_state._auth_ok:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, * { font-family: 'Inter', sans-serif !important; }

/* Full-page dark background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #050d1a 0%, #0a1628 50%, #0d1f3c 100%) !important;
    min-height: 100vh;
}
[data-testid="stHeader"]          { display: none !important; }
[data-testid="stSidebar"]         { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }
footer                            { display: none !important; }

/* Center the login card */
.block-container {
    max-width: 420px !important;
    padding-top: 6vh !important;
    margin: 0 auto !important;
}

/* Card */
div[data-testid="stForm"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    padding: 36px 40px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 24px 60px rgba(0,0,0,0.5) !important;
}

/* Inputs */
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: #f0f4ff !important;
    padding: 10px 14px !important;
}
div[data-testid="stTextInput"] label {
    color: #a0b0d0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #4a90d9 !important;
    box-shadow: 0 0 0 3px rgba(74,144,217,0.2) !important;
    outline: none !important;
}

/* Sign-in button */
div[data-testid="stForm"] .stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1a1a6b 0%, #2d2db0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
div[data-testid="stForm"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(26,26,107,0.5) !important;
}

/* Error message */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    background: rgba(220,38,38,0.15) !important;
    border: 1px solid rgba(220,38,38,0.35) !important;
    color: #fca5a5 !important;
}
</style>
""", unsafe_allow_html=True)

    # Brand header
    st.markdown("""
<div style="text-align:center;margin-bottom:8px">
  <div style="font-size:3rem;margin-bottom:4px">🧬</div>
  <div style="color:#e0e8ff;font-size:1.5rem;font-weight:700;letter-spacing:-0.02em">
    Tech Strategy Lab
  </div>
  <div style="color:#6080a0;font-size:0.85rem;margin-top:4px;font-weight:400">
    Sign in to access your analytics workspace
  </div>
</div>
""", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        sign_in  = st.form_submit_button("Sign In")

    if sign_in:
        if username == _VALID_USER and password == _VALID_PASS:
            st.session_state._auth_ok = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown(f"""
<div style="text-align:center;margin-top:28px;color:#4a6080;font-size:0.78rem">
  Need help?&nbsp;
  <a href="mailto:{_SUPPORT_EMAIL}" style="color:#4a90d9;text-decoration:none">
    {_SUPPORT_EMAIL}
  </a>
</div>
""", unsafe_allow_html=True)

    st.stop()

# ─── GLOBAL CSS (authenticated view) ────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, * { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
[data-testid="stAppViewContainer"] { background: #f0f4f9 !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 65%, #0a1a2e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p  { color: #a8b8d8 !important; font-size: 0.83rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #7090d0 !important; font-size: 0.88rem !important;
                                text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stSidebar"] .stRadio label  { color: #c0d0ec !important; }
[data-testid="stSidebar"] hr             { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 6px !important;
    color: #e0e8ff !important;
}

/* ── Page title ── */
h1 { color: #0a1628 !important; font-weight: 700 !important; letter-spacing: -0.03em !important; }
h2 { color: #0d1f3c !important; font-weight: 600 !important; }
h3 { color: #1a3060 !important; font-weight: 600 !important; }

/* ── Tab strip ── */
[data-baseweb="tab-list"] {
    background: #dde4ef;
    border-radius: 10px;
    gap: 2px;
    padding: 3px;
}
[data-baseweb="tab"] {
    border-radius: 8px;
    padding: 6px 20px;
    font-weight: 500;
    color: #4a5568;
    font-size: 0.88rem;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, #0a1628 0%, #1a3060 100%) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(10,22,40,0.3) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07) !important;
    border-left: 4px solid #0d1f3c !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12) !important;
}
[data-testid="stMetricLabel"]  { color: #6b7280 !important; font-size: 0.78rem !important;
                                  font-weight: 500 !important; text-transform: uppercase;
                                  letter-spacing: 0.06em; }
[data-testid="stMetricValue"]  { color: #0a1628 !important; font-size: 1.5rem !important;
                                  font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { font-size: 0.82rem !important; font-weight: 500 !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.18s !important;
    border: none !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.18) !important;
}
/* Primary-style buttons in sidebar */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1a3060 0%, #2d50a0 100%) !important;
    color: white !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #0d5c3a 0%, #128a58 100%) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(13,92,58,0.4) !important;
}

/* ── Info/success/warning/error ── */
[data-testid="stAlert"]   { border-radius: 10px !important; font-size: 0.88rem !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #dde4ef !important;
    border-radius: 10px !important;
    background: white !important;
}

/* ── Data tables ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Logout button in sidebar ── */
#logout-btn .stButton > button {
    background: rgba(220,38,38,0.15) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(220,38,38,0.3) !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ──────────────────────────────────────────────────────────────

if "event_log"        not in st.session_state: st.session_state.event_log        = []
if "shock_library"    not in st.session_state: st.session_state.shock_library    = []
if "shift_target_idx" not in st.session_state: st.session_state.shift_target_idx = None
if "tgt_start"        not in st.session_state: st.session_state.tgt_start        = date(2026, 1, 1)
if "tgt_end"          not in st.session_state: st.session_state.tgt_end          = date(2026, 12, 31)
if "target_metric"    not in st.session_state: st.session_state.target_metric    = "Sales"
if "target_val"       not in st.session_state: st.session_state.target_val       = 200_000.0
# Sidebar widget persistence
if "ui_res_level"     not in st.session_state: st.session_state.ui_res_level     = "Monthly"
if "ui_t_start"       not in st.session_state: st.session_state.ui_t_start       = date(2026, 1, 1)
if "ui_t_end"         not in st.session_state: st.session_state.ui_t_end         = date(2026, 1, 31)
if "ui_c_val"         not in st.session_state: st.session_state.ui_c_val         = 5_000.0
if "ui_q_val"         not in st.session_state: st.session_state.ui_q_val         = 250.0
if "ui_s_val"         not in st.session_state: st.session_state.ui_s_val         = 12_500.0
if "ui_adj_c"         not in st.session_state: st.session_state.ui_adj_c         = 0.0
if "ui_adj_q"         not in st.session_state: st.session_state.ui_adj_q         = 0.0
if "ui_adj_s"         not in st.session_state: st.session_state.ui_adj_s         = 0.0
if "ui_sel_brands"    not in st.session_state: st.session_state.ui_sel_brands    = []

# ─── DATA LOADING (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_data():
    profiles    = pd.read_csv(PROFILES_PATH)
    profiles["Year"] = profiles["Year"].astype(str)
    yearly_kpis = pd.read_csv(YEARLY_KPI_PATH)
    df_raw      = pd.read_csv(DATASET_PATH)
    df_raw["Date"]  = pd.to_datetime(df_raw["Date"])
    df_raw["brand"] = df_raw["brand"].str.strip().str.lower()
    return profiles, yearly_kpis, df_raw

profiles, yearly_kpis, df_raw = load_data()

data_years   = sorted([int(y) for y in profiles["Year"].unique() if y != "Overall"])
min_data_yr  = data_years[0]
max_data_yr  = data_years[-1]

# ─── TITLE ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
  <span style="font-size:2.2rem">🧬</span>
  <div>
    <div style="font-size:1.75rem;font-weight:700;color:#0a1628;line-height:1.1;
                letter-spacing:-0.03em">Tech Strategy Lab</div>
    <div style="font-size:0.82rem;color:#6b7280;font-weight:400;margin-top:2px">
      Marketing analytics &amp; campaign simulation platform
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────

# Sidebar logo / brand
st.sidebar.markdown("""
<div style="text-align:center;padding:12px 0 8px">
  <div style="font-size:1.6rem">🧬</div>
  <div style="color:#7090d0;font-weight:700;font-size:0.9rem;letter-spacing:0.04em">
    TECH STRATEGY LAB
  </div>
  <div style="color:#4a6080;font-size:0.72rem;margin-top:2px">Analytics Platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

page = st.sidebar.radio("🧭 Navigation",
                         ["📊 Main Dashboard", "⚡ Event Simulation Lab"])
st.sidebar.markdown("---")

# 1. Market Resolution
st.sidebar.header("🧬 1. Market Resolution")
res_level = st.sidebar.radio(
    "Analysis Granularity", ["Monthly", "Weekly", "Daily"], key="ui_res_level")
time_col  = ("Month" if res_level == "Monthly"
              else "Week" if res_level == "Weekly" else "DayOfYear")

# Brand selection
st.sidebar.markdown("**Select DNA Brands**")
all_brands = sorted(profiles["brand"].unique())
if not st.session_state.ui_sel_brands:
    st.session_state.ui_sel_brands = list(all_brands)
select_all = st.sidebar.checkbox("ALL", value=True)
sel_brands = []
for b in all_brands:
    chk = st.sidebar.checkbox(
        b.title(),
        value=(select_all or b in st.session_state.ui_sel_brands),
        disabled=select_all,
        key=f"chk_{b}",
    )
    if select_all or chk:
        sel_brands.append(b)
st.session_state.ui_sel_brands = sel_brands

if not sel_brands:
    st.warning("Please select at least one brand.")
    st.stop()

# 2. Trial Reality
st.sidebar.markdown("---")
st.sidebar.header("🎯 2. Trial Reality")
t_start = st.sidebar.date_input("Trial Start Date", key="ui_t_start")
t_end   = st.sidebar.date_input("Trial End Date",   key="ui_t_end")
c_val   = st.sidebar.number_input("Total Clicks in Trial",  min_value=0.0, key="ui_c_val")
q_val   = st.sidebar.number_input("Total Qty in Trial",     min_value=0.0, key="ui_q_val")
s_val   = st.sidebar.number_input("Total Sales in Trial",   min_value=0.0, key="ui_s_val")

if t_start.year < min_data_yr or t_end.year > max_data_yr + 2:
    st.sidebar.warning(
        f"Trial dates may be outside data range ({min_data_yr}–{max_data_yr}). "
        "DNA will be extrapolated.")

with st.sidebar.expander("⚙️ Trial Pre-Adjustment"):
    st.caption(
        "**Positive %** = trial was boosted (strip the boost to get organic base).  "
        "**Negative %** = trial was suppressed — add back the missing lift.")
    st.number_input("Clicks Adjustment (%)", -100.0, 500.0, key="ui_adj_c", step=5.0)
    st.number_input("Qty Adjustment (%)",    -100.0, 500.0, key="ui_adj_q", step=5.0)
    st.number_input("Sales Adjustment (%)",  -100.0, 500.0, key="ui_adj_s", step=5.0)

trial_adj_c = st.session_state.ui_adj_c
trial_adj_q = st.session_state.ui_adj_q
trial_adj_s = st.session_state.ui_adj_s

# Adjusted trial values used for calibration
adj_c = c_val / (1 + trial_adj_c / 100) if (1 + trial_adj_c / 100) != 0 else c_val
adj_q = q_val / (1 + trial_adj_q / 100) if (1 + trial_adj_q / 100) != 0 else q_val
adj_s = s_val / (1 + trial_adj_s / 100) if (1 + trial_adj_s / 100) != 0 else s_val

# ─── DYNAMIC SIMILARITY ENGINE ─────────────────────────────────────────────────

proj_year    = str(t_start.year)
norm_weights = compute_similarity_weights(
    profiles, sel_brands, proj_year, t_start, t_end, c_val, q_val, s_val)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ DNA Similarity Engine")
st.sidebar.caption("🎯 **Dynamic Weight Distribution:**")
st.sidebar.caption("• **35.0%** — All-Time Overall Average")
for y, w in norm_weights.items():
    st.sidebar.caption(f"• **{w * 65.0:.1f}%** — {y} Historical Pattern")

# ─── BUILD DNA & YEAR DATAFRAME ────────────────────────────────────────────────

pure_dna       = build_pure_dna(profiles, sel_brands, norm_weights)
df, _full_year = build_year_dataframe(int(proj_year))
build_dna_layers(df, pure_dna, st.session_state.event_log)

# ─── CALIBRATION ───────────────────────────────────────────────────────────────

base_clicks, base_cr, base_aov = calibrate_base(df, t_start, t_end, adj_c, adj_q, adj_s)

if base_clicks is None:
    st.error("Trial date range yields zero DNA sum. Please widen the trial period.")
    st.stop()

# ─── PROJECTIONS ───────────────────────────────────────────────────────────────

build_projections(df, base_clicks, base_cr, base_aov, st.session_state.event_log)

# ─── EXPORT STRATEGY BUTTON ────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("📥 Export")

excel_bytes = build_excel_report(
    df, st.session_state.event_log, sel_brands,
    t_start, t_end, adj_c, adj_q, adj_s,
    base_clicks, base_cr, base_aov,
)
st.sidebar.download_button(
    label="⬇️ Export Strategy (Excel)",
    data=excel_bytes,
    file_name=f"strategy_report_{proj_year}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

# ─── AI CHATBOT ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
render_chatbot()

# ─── SIDEBAR FOOTER ─────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size:0.74rem;color:#3a5070;text-align:center;padding:4px 0">
  💬 Questions or support?<br>
  <a href="mailto:{_SUPPORT_EMAIL}"
     style="color:#4a90d9;text-decoration:none;font-weight:500">
    {_SUPPORT_EMAIL}
  </a>
</div>
""", unsafe_allow_html=True)

# Logout
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown('<div id="logout-btn">', unsafe_allow_html=True)
if st.sidebar.button("🔒 Sign Out", use_container_width=True):
    st.session_state._auth_ok = False
    st.rerun()
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ─── PAGE ROUTING ──────────────────────────────────────────────────────────────

if page == "📊 Main Dashboard":
    render_dashboard(
        df, profiles, yearly_kpis,
        sel_brands, res_level, time_col,
        base_cr, base_aov,
    )

elif page == "⚡ Event Simulation Lab":
    render_lab(
        df, df_raw, sel_brands, res_level, time_col,
        base_clicks, base_cr, base_aov,
        adj_c, adj_q, adj_s,
        t_start, t_end, pure_dna,
    )
