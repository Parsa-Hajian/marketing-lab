import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from config import PROFILES_PATH, YEARLY_KPI_PATH, DATASET_PATH, LOGO_PATH
from engine.dna import (
    compute_similarity_weights,
    build_pure_dna,
    build_year_dataframe,
    build_dna_layers,
)
from engine.calibration import calibrate_base, build_projections
from views.dashboard import render_dashboard
from views.lab import render_lab
from views.brand_add import render_brand_add
from views.brand_update import render_brand_update
from utils.export import build_excel_report

st.set_page_config(page_title="Tech Strategy Lab", layout="wide", page_icon="🧬")

# ─── CONSTANTS ───────────────────────────────────────────────────────────────────
_USER         = "universitybox_2026"
_PASS         = "99118822"
_EMAIL        = "parsa.hajiannejad@universitybox.com"
_ORANGE       = "#F47920"
_BLACK        = "#111111"
_LOGO_EXISTS  = os.path.exists(LOGO_PATH)

def _logo_b64() -> str:
    if not _LOGO_EXISTS:
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()

_LOGO_B64 = _logo_b64()

# ─── SHARED CSS HELPERS ──────────────────────────────────────────────────────────
_BASE_FONTS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, * { font-family: 'Inter', sans-serif !important; }
"""

# ─── AUTH GATE ───────────────────────────────────────────────────────────────────
if "_auth_ok" not in st.session_state:
    st.session_state._auth_ok = False

if not st.session_state._auth_ok:
    st.markdown(f"""
<style>
{_BASE_FONTS}
[data-testid="stAppViewContainer"] {{
    background: {_BLACK} !important;
    min-height: 100vh;
}}
[data-testid="stHeader"], [data-testid="stSidebar"],
[data-testid="stToolbar"], footer {{ display: none !important; }}

.block-container {{
    max-width: 400px !important;
    padding-top: 8vh !important;
    margin: 0 auto !important;
}}
div[data-testid="stForm"] {{
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 12px !important;
    padding: 32px 36px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6) !important;
}}
div[data-testid="stTextInput"] input {{
    background: #222 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #F0F0F0 !important;
    padding: 10px 14px !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: {_ORANGE} !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(244,121,32,0.25) !important;
}}
div[data-testid="stTextInput"] label {{ color: #888 !important; font-size: 0.8rem !important; }}
div[data-testid="stForm"] .stButton > button {{
    width: 100% !important;
    background: {_ORANGE} !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 11px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.18s !important;
}}
div[data-testid="stForm"] .stButton > button:hover {{
    opacity: 0.88 !important;
}}
[data-testid="stAlert"] {{
    background: rgba(220,38,38,0.12) !important;
    border: 1px solid rgba(220,38,38,0.3) !important;
    border-radius: 6px !important;
    color: #FCA5A5 !important;
}}
</style>
""", unsafe_allow_html=True)

    # Logo + title
    logo_col = st.columns([1, 2, 1])[1]
    with logo_col:
        if _LOGO_EXISTS:
            st.image(LOGO_PATH, width=72)
        st.markdown(f"""
<div style='text-align:center;margin:8px 0 20px'>
  <div style='color:#F0F0F0;font-size:1.35rem;font-weight:700;
              letter-spacing:-0.02em'>Tech Strategy Lab</div>
  <div style='color:#555;font-size:0.78rem;margin-top:3px'>
    Sign in to your workspace
  </div>
</div>""", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            sign_in  = st.form_submit_button("Sign In")

        if sign_in:
            if username == _USER and password == _PASS:
                st.session_state._auth_ok = True
                st.rerun()
            else:
                st.error("Invalid credentials.")

        st.markdown(f"""
<div style='text-align:center;margin-top:20px;color:#444;font-size:0.75rem'>
  <a href='mailto:{_EMAIL}' style='color:#F47920;text-decoration:none'>{_EMAIL}</a>
</div>""", unsafe_allow_html=True)

    st.stop()


# ─── GLOBAL CSS (authenticated) ──────────────────────────────────────────────────
st.markdown(f"""
<style>
{_BASE_FONTS}

/* ── Background ── */
[data-testid="stAppViewContainer"] {{ background: #F7F7F7 !important; }}
[data-testid="stHeader"]           {{ background: transparent !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {_BLACK} !important;
    border-right: 1px solid #1E1E1E !important;
}}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p  {{ color: #888 !important; font-size: 0.81rem !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {_ORANGE} !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stRadio label {{ color: #CCCCCC !important; font-size: 0.86rem !important; }}
[data-testid="stSidebar"] hr             {{ border-color: #222 !important; }}
[data-testid="stSidebar"] input          {{
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 5px !important;
    color: #E0E0E0 !important;
}}

/* ── Page typography ── */
h1 {{ color: {_BLACK} !important; font-weight: 700 !important;
      letter-spacing: -0.025em !important; font-size: 1.6rem !important; }}
h2 {{ color: {_BLACK} !important; font-weight: 600 !important; }}
h3 {{ color: #333 !important; font-weight: 600 !important; }}

/* ── Tabs ── */
[data-baseweb="tab-list"] {{
    background: #EBEBEB;
    border-radius: 7px;
    gap: 2px;
    padding: 3px;
}}
[data-baseweb="tab"] {{
    border-radius: 5px;
    padding: 5px 18px;
    font-weight: 500;
    color: #555;
    font-size: 0.86rem;
}}
[aria-selected="true"] {{
    background: {_ORANGE} !important;
    color: white !important;
    box-shadow: 0 2px 6px rgba(244,121,32,0.35) !important;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: white !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    border-left: 3px solid {_ORANGE} !important;
    border-top: none !important;
    transition: transform 0.15s !important;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}}
[data-testid="stMetricLabel"] {{
    color: #888 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
[data-testid="stMetricValue"] {{
    color: {_BLACK} !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    border: none !important;
    transition: all 0.15s !important;
}}
/* Default (non-primary) buttons */
.stButton > button:not([kind="primaryFormSubmit"]) {{
    background: {_BLACK} !important;
    color: white !important;
}}
.stButton > button:not([kind="primaryFormSubmit"]):hover {{
    opacity: 0.82 !important;
}}
/* Primary buttons */
.stButton > button[kind="primaryFormSubmit"],
.stButton > button[data-testid="baseButton-primary"] {{
    background: {_ORANGE} !important;
    color: white !important;
}}
.stButton > button[data-testid="baseButton-primary"]:hover {{
    opacity: 0.88 !important;
    box-shadow: 0 4px 12px rgba(244,121,32,0.35) !important;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {{
    background: {_ORANGE} !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    border: none !important;
    width: 100% !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    opacity: 0.88 !important;
}}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {{
    background: #1A1A1A !important;
    color: #CCC !important;
    border: 1px solid #2A2A2A !important;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    border-radius: 7px !important;
    font-size: 0.86rem !important;
    border: none !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    border: 1px solid #E8E8E8 !important;
    border-radius: 8px !important;
    background: white !important;
}}

/* ── Sidebar sign-out ── */
.signout-btn .stButton > button {{
    background: transparent !important;
    color: #555 !important;
    border: 1px solid #2A2A2A !important;
    font-size: 0.78rem !important;
    padding: 5px 10px !important;
}}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────────
if "event_log"        not in st.session_state: st.session_state.event_log        = []
if "shock_library"    not in st.session_state: st.session_state.shock_library    = []
if "shift_target_idx" not in st.session_state: st.session_state.shift_target_idx = None
if "tgt_start"        not in st.session_state: st.session_state.tgt_start        = date(2026, 1, 1)
if "tgt_end"          not in st.session_state: st.session_state.tgt_end          = date(2026, 12, 31)
if "target_metric"    not in st.session_state: st.session_state.target_metric    = "Sales"
if "target_val"       not in st.session_state: st.session_state.target_val       = 200_000.0
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

# ─── DATA LOADING ─────────────────────────────────────────────────────────────────
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

data_years  = sorted([int(y) for y in profiles["Year"].unique() if y != "Overall"])
min_data_yr = data_years[0]
max_data_yr = data_years[-1]

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────────

# Logo + brand
if _LOGO_EXISTS:
    st.sidebar.image(LOGO_PATH, width=56)
st.sidebar.markdown(f"""
<div style='margin-bottom:2px'>
  <span style='color:#E0E0E0;font-weight:700;font-size:0.9rem;letter-spacing:-0.01em'>
    Tech Strategy Lab
  </span>
</div>
<div style='color:#444;font-size:0.72rem;margin-bottom:8px'>Analytics Platform</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "⚡ Lab", "➕ Add Brand", "✏️ Update Brand"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

# ── Only show analytics controls for Dashboard + Lab ──────────────────────────
_is_analytics_page = page in ("📊 Dashboard", "⚡ Lab")

if _is_analytics_page:
    # 1. Market Resolution
    st.sidebar.header("Market Resolution")
    res_level = st.sidebar.radio(
        "Granularity", ["Monthly", "Weekly", "Daily"], key="ui_res_level")
    time_col = ("Month" if res_level == "Monthly"
                 else "Week" if res_level == "Weekly" else "DayOfYear")

    # Brand selection
    st.sidebar.markdown("<span style='color:#666;font-size:0.78rem;font-weight:600'>DNA BRANDS</span>",
                        unsafe_allow_html=True)
    all_brands = sorted(profiles["brand"].unique())
    if not st.session_state.ui_sel_brands:
        st.session_state.ui_sel_brands = list(all_brands)
    select_all = st.sidebar.checkbox("All brands", value=True)
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
        st.warning("Select at least one brand.")
        st.stop()

    # 2. Trial Reality
    st.sidebar.markdown("---")
    st.sidebar.header("Trial Reality")
    t_start = st.sidebar.date_input("Start Date", key="ui_t_start")
    t_end   = st.sidebar.date_input("End Date",   key="ui_t_end")
    c_val   = st.sidebar.number_input("Clicks",  min_value=0.0, key="ui_c_val")
    q_val   = st.sidebar.number_input("Quantity", min_value=0.0, key="ui_q_val")
    s_val   = st.sidebar.number_input("Sales",   min_value=0.0, key="ui_s_val")

    if t_start.year < min_data_yr or t_end.year > max_data_yr + 2:
        st.sidebar.warning(f"Outside data range ({min_data_yr}–{max_data_yr}).")

    with st.sidebar.expander("Pre-Adjustment"):
        st.caption("+ % = trial boosted (strip boost).  − % = suppressed (add lift back).")
        st.number_input("Clicks adj (%)",   -100.0, 500.0, key="ui_adj_c", step=5.0)
        st.number_input("Quantity adj (%)", -100.0, 500.0, key="ui_adj_q", step=5.0)
        st.number_input("Sales adj (%)",    -100.0, 500.0, key="ui_adj_s", step=5.0)

    adj_c = c_val / (1 + st.session_state.ui_adj_c / 100) \
            if (1 + st.session_state.ui_adj_c / 100) != 0 else c_val
    adj_q = q_val / (1 + st.session_state.ui_adj_q / 100) \
            if (1 + st.session_state.ui_adj_q / 100) != 0 else q_val
    adj_s = s_val / (1 + st.session_state.ui_adj_s / 100) \
            if (1 + st.session_state.ui_adj_s / 100) != 0 else s_val

    # 3. DNA similarity
    proj_year    = str(t_start.year)
    norm_weights = compute_similarity_weights(
        profiles, sel_brands, proj_year, t_start, t_end, c_val, q_val, s_val)

    st.sidebar.markdown("---")
    st.sidebar.header("DNA Weights")
    st.sidebar.caption("• **35%** — All-time overall")
    for y, w in norm_weights.items():
        st.sidebar.caption(f"• **{w * 65.0:.1f}%** — {y}")

    # Engine
    pure_dna       = build_pure_dna(profiles, sel_brands, norm_weights)
    df, _full_year = build_year_dataframe(int(proj_year))
    build_dna_layers(df, pure_dna, st.session_state.event_log)

    base_clicks, base_cr, base_aov = calibrate_base(df, t_start, t_end, adj_c, adj_q, adj_s)

    if base_clicks is None:
        st.error("Trial date range yields zero DNA sum. Please widen the trial period.")
        st.stop()

    build_projections(df, base_clicks, base_cr, base_aov, st.session_state.event_log)

    # Export
    st.sidebar.markdown("---")
    st.sidebar.header("Export")
    excel_bytes = build_excel_report(
        df, st.session_state.event_log, sel_brands,
        t_start, t_end, adj_c, adj_q, adj_s,
        base_clicks, base_cr, base_aov,
    )
    st.sidebar.download_button(
        label="⬇️ Export Strategy",
        data=excel_bytes,
        file_name=f"strategy_{proj_year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ── Sidebar footer ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:0.72rem;color:#444;padding:2px 0 8px'>
  <a href='mailto:{_EMAIL}' style='color:{_ORANGE};text-decoration:none'>{_EMAIL}</a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="signout-btn">', unsafe_allow_html=True)
if st.sidebar.button("Sign Out", use_container_width=True):
    st.session_state._auth_ok = False
    st.rerun()
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ─── PAGE TITLE ──────────────────────────────────────────────────────────────────
_page_labels = {
    "📊 Dashboard":   "Dashboard",
    "⚡ Lab":          "Event Simulation Lab",
    "➕ Add Brand":    "Add Brand",
    "✏️ Update Brand": "Update Brand",
}
_logo_tag = (
    f"<img src='data:image/png;base64,{_LOGO_B64}' "
    "style='height:34px;border-radius:50%;margin-right:2px'>"
    if _LOGO_B64 else "🧬"
)
st.markdown(f"""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:2px'>
  {_logo_tag}
  <span style='font-size:1.4rem;font-weight:700;color:{_BLACK};
               letter-spacing:-0.025em'>{_page_labels.get(page, page)}</span>
</div>
<div style='height:1px;background:#E8E8E8;margin-bottom:16px'></div>
""", unsafe_allow_html=True)

# ─── PAGE ROUTING ─────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    render_dashboard(
        df, profiles, yearly_kpis,
        sel_brands, res_level, time_col,
        base_cr, base_aov,
    )

elif page == "⚡ Lab":
    render_lab(
        df, df_raw, sel_brands, res_level, time_col,
        base_clicks, base_cr, base_aov,
        adj_c, adj_q, adj_s,
        t_start, t_end, pure_dna,
    )

elif page == "➕ Add Brand":
    render_brand_add()

elif page == "✏️ Update Brand":
    render_brand_update()
