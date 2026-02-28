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

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
_USER   = "universitybox_2026"
_PASS   = "99118822"
_EMAIL  = "parsa.hajiannejad@universitybox.com"
_ORANGE = "#F47920"
_BLACK  = "#111111"

_LOGO_EXISTS = os.path.exists(LOGO_PATH)

def _b64_logo() -> str:
    if not _LOGO_EXISTS:
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()

_LOGO_B64 = _b64_logo()

# ─── AUTH GATE ────────────────────────────────────────────────────────────────
if "_auth_ok" not in st.session_state:
    st.session_state._auth_ok = False

if not st.session_state._auth_ok:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] { background: #111111 !important; }
[data-testid="stHeader"]  { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
footer                    { display: none !important; }

.main .block-container {
    max-width: 380px;
    padding-top: 7vh;
    padding-left: 1rem;
    padding-right: 1rem;
    margin: 0 auto;
}
[data-testid="stForm"] {
    background: #1C1C1C;
    border: 1px solid #2C2C2C;
    border-radius: 12px;
    padding: 28px 28px 20px;
}
[data-testid="stTextInput"] > div > div > input {
    background: #252525;
    border: 1px solid #383838;
    color: #F0F0F0;
    border-radius: 6px;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #F47920;
    box-shadow: 0 0 0 2px rgba(244,121,32,0.2);
}
/* Form submit buttons */
[data-testid="baseButton-secondaryFormSubmit"],
[data-testid="baseButton-primaryFormSubmit"] {
    background-color: #F47920 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Login layout ─────────────────────────────────────────────────────────
    if _LOGO_B64:
        st.markdown(
            f"<div style='text-align:center;margin-bottom:4px'>"
            f"<img src='data:image/png;base64,{_LOGO_B64}' "
            f"style='width:64px;border-radius:50%'></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h3 style='text-align:center;color:#F0F0F0;margin:0 0 4px;font-family:Inter,sans-serif'>"
        "Tech Strategy Lab</h3>"
        "<p style='text-align:center;color:#666;font-size:0.82rem;margin-bottom:20px;font-family:Inter,sans-serif'>"
        "Sign in to continue</p>",
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

    if submitted:
        if username == _USER and password == _PASS:
            st.session_state._auth_ok = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown(
        f"<p style='text-align:center;font-size:0.75rem;color:#444;margin-top:16px;font-family:Inter,sans-serif'>"
        f"<a href='mailto:{_EMAIL}' style='color:#F47920;text-decoration:none'>{_EMAIL}</a></p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ─── GLOBAL CSS (authenticated view only) ────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Apply Inter font to headings and text — NOT to inputs ── */
h1, h2, h3, h4, h5, h6,
p, label, .stMarkdown,
[data-baseweb="tab"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {{
    font-family: 'Inter', sans-serif;
}}

/* ── App background ── */
[data-testid="stAppViewContainer"] {{ background: #F5F5F5; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {{
    background-color: {_BLACK};
}}
section[data-testid="stSidebar"] p   {{ color: #888888; }}
section[data-testid="stSidebar"] label {{ color: #CCCCCC; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3  {{
    color: {_ORANGE};
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
}}
section[data-testid="stSidebar"] hr {{ border-color: #2A2A2A; }}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: white;
    border-radius: 10px;
    border-left: 4px solid {_ORANGE};
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    padding: 14px 18px;
}}

/* ── Tab strip ── */
[data-baseweb="tab-list"] {{
    background: #E8E8E8;
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
}}
[data-baseweb="tab"] {{
    border-radius: 6px;
    font-weight: 500;
    color: #555555;
}}
[aria-selected="true"] {{
    background-color: {_ORANGE} !important;
    color: white !important;
}}

/* ── Primary action buttons ── */
[data-testid="baseButton-primary"],
[data-testid="baseButton-primaryFormSubmit"] {{
    background-color: {_ORANGE} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
[data-testid="baseButton-primary"]:hover,
[data-testid="baseButton-primaryFormSubmit"]:hover {{
    opacity: 0.88 !important;
}}

/* ── Secondary buttons — keep Streamlit default, just round them ── */
[data-testid="baseButton-secondary"],
[data-testid="baseButton-secondaryFormSubmit"] {{
    border-radius: 8px !important;
    font-weight: 500 !important;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {{
    background-color: {_ORANGE} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
[data-testid="stDownloadButton"] > button:hover {{ opacity: 0.88 !important; }}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: white;
    border: 1px solid #E8E8E8;
    border-radius: 10px;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{ border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
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

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
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

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

# Logo + name
if _LOGO_EXISTS:
    st.sidebar.image(LOGO_PATH, width=90)
st.sidebar.markdown(
    f"<div style='margin-top:6px'>"
    f"<span style='color:#E0E0E0;font-size:0.92rem;font-weight:700;"
    f"font-family:Inter,sans-serif;letter-spacing:-0.01em'>Tech Strategy Lab</span><br>"
    f"<span style='color:#555;font-size:0.72rem;font-family:Inter,sans-serif'>"
    f"Analytics Platform</span></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "⚡ Lab", "➕ Add Brand", "✏️ Update Brand"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

# Analytics controls only on Dashboard / Lab
_is_analytics_page = page in ("📊 Dashboard", "⚡ Lab")

if _is_analytics_page:
    st.sidebar.header("Market Resolution")
    res_level = st.sidebar.radio(
        "Granularity", ["Monthly", "Weekly", "Daily"], key="ui_res_level")
    time_col = ("Month"     if res_level == "Monthly"
                 else "Week" if res_level == "Weekly" else "DayOfYear")

    st.sidebar.markdown("**Select Brands**")
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

    st.sidebar.markdown("---")
    st.sidebar.header("Trial Reality")
    t_start = st.sidebar.date_input("Start Date", key="ui_t_start")
    t_end   = st.sidebar.date_input("End Date",   key="ui_t_end")
    c_val   = st.sidebar.number_input("Clicks",   min_value=0.0, key="ui_c_val")
    q_val   = st.sidebar.number_input("Quantity", min_value=0.0, key="ui_q_val")
    s_val   = st.sidebar.number_input("Sales",    min_value=0.0, key="ui_s_val")

    if t_start.year < min_data_yr or t_end.year > max_data_yr + 2:
        st.sidebar.warning(f"Outside data range ({min_data_yr}–{max_data_yr}).")

    with st.sidebar.expander("Pre-Adjustment"):
        st.caption("+ % = trial boosted.  − % = trial suppressed.")
        st.number_input("Clicks adj (%)",   -100.0, 500.0, key="ui_adj_c", step=5.0)
        st.number_input("Quantity adj (%)", -100.0, 500.0, key="ui_adj_q", step=5.0)
        st.number_input("Sales adj (%)",    -100.0, 500.0, key="ui_adj_s", step=5.0)

    adj_c = c_val / (1 + st.session_state.ui_adj_c / 100) if (1 + st.session_state.ui_adj_c / 100) != 0 else c_val
    adj_q = q_val / (1 + st.session_state.ui_adj_q / 100) if (1 + st.session_state.ui_adj_q / 100) != 0 else q_val
    adj_s = s_val / (1 + st.session_state.ui_adj_s / 100) if (1 + st.session_state.ui_adj_s / 100) != 0 else s_val

    proj_year    = str(t_start.year)
    norm_weights = compute_similarity_weights(
        profiles, sel_brands, proj_year, t_start, t_end, c_val, q_val, s_val)

    st.sidebar.markdown("---")
    st.sidebar.header("DNA Weights")
    st.sidebar.caption("• **35%** — All-time overall")
    for y, w in norm_weights.items():
        st.sidebar.caption(f"• **{w * 65.0:.1f}%** — {y}")

    pure_dna       = build_pure_dna(profiles, sel_brands, norm_weights)
    df, _full_year = build_year_dataframe(int(proj_year))
    build_dna_layers(df, pure_dna, st.session_state.event_log)

    base_clicks, base_cr, base_aov = calibrate_base(df, t_start, t_end, adj_c, adj_q, adj_s)
    if base_clicks is None:
        st.error("Trial date range yields zero DNA sum. Widen the trial period.")
        st.stop()

    build_projections(df, base_clicks, base_cr, base_aov, st.session_state.event_log)

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

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<p style='font-size:0.72rem;color:#444;font-family:Inter,sans-serif;margin:0'>"
    f"<a href='mailto:{_EMAIL}' style='color:{_ORANGE};text-decoration:none'>{_EMAIL}</a></p>",
    unsafe_allow_html=True,
)
if st.sidebar.button("Sign Out", use_container_width=True):
    st.session_state._auth_ok = False
    st.rerun()

# ─── PAGE HEADER ──────────────────────────────────────────────────────────────
_labels = {
    "📊 Dashboard":   "Dashboard",
    "⚡ Lab":          "Event Simulation Lab",
    "➕ Add Brand":    "Add Brand",
    "✏️ Update Brand": "Update Brand",
}
_img_tag = (
    f"<img src='data:image/png;base64,{_LOGO_B64}' "
    f"style='width:32px;height:32px;border-radius:50%;vertical-align:middle;margin-right:8px'>"
    if _LOGO_B64 else ""
)
st.markdown(
    f"<h2 style='margin:0 0 4px;font-family:Inter,sans-serif;color:{_BLACK};font-weight:700'>"
    f"{_img_tag}{_labels.get(page, page)}</h2>"
    f"<hr style='margin:0 0 16px;border:none;border-top:1px solid #E0E0E0'>",
    unsafe_allow_html=True,
)

# ─── PAGE ROUTING ─────────────────────────────────────────────────────────────
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
