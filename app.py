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
from engine.activity_log import log_login
from engine.settings_store import load_settings
from engine.i18n import t
from views.dashboard import render_dashboard
from views.lab import render_lab
from views.brand_add import render_brand_add
from views.brand_update import render_brand_update
from views.user_log import render_user_log
from views.settings import render_settings
from views.docs import render_docs
from utils.export import build_excel_report

st.set_page_config(
    page_title="Tech Strategy Lab",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
_USER   = "universitybox_2026"
_PASS   = "99118822"
_EMAIL  = "parsa.hajiannejad@universitybox.com"
_ORANGE = "#F47920"
_BLACK  = "#111111"

_LOGO_EXISTS = os.path.exists(LOGO_PATH)
def _b64() -> str:
    if not _LOGO_EXISTS:
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()
_LOGO_B64 = _b64()

# ─── DESIGN TOKENS & FULL CSS ──────────────────────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design tokens ── */
:root {{
    --orange:       {_ORANGE};
    --black:        {_BLACK};
    --bg:           #F8F8F8;
    --surface:      #FFFFFF;
    --border:       #EFEFEF;
    --text-1:       #111111;
    --text-2:       #555555;
    --text-3:       #AAAAAA;
    --radius:       10px;
    --shadow:       0 1px 12px rgba(0,0,0,0.06);
    --shadow-hover: 0 4px 20px rgba(0,0,0,0.10);
}}

/* ── Inter font on text elements only ── */
h1,h2,h3,h4,h5,h6,p,label,
.stMarkdown,.stCaption,
[data-baseweb="tab"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {{
    font-family: 'Inter', sans-serif;
}}

/* ── Strip Streamlit chrome — KEEP the header element in the DOM
       so its embedded sidebar-toggle button keeps working.
       Only hide the content items inside it. ── */
#MainMenu                        {{ display: none !important; }}
footer                           {{ display: none !important; }}
[data-testid="stToolbar"]        {{ display: none !important; }}
[data-testid="stDecoration"]     {{ display: none !important; }}
[data-testid="stStatusWidget"]   {{ display: none !important; }}

/* Shrink the header to a thin invisible bar; toggle button stays accessible */
[data-testid="stHeader"] {{
    background:    var(--bg) !important;
    border-bottom: none !important;
    height:        2.25rem !important;
    min-height:    2.25rem !important;
    padding:       0 !important;
    overflow:      visible !important;
}}

/* ── Global background ── */
[data-testid="stAppViewContainer"] {{ background: var(--bg); }}

/* ── Main content padding ── */
.main .block-container {{
    padding-top:   2rem;
    padding-left:  2.5rem;
    padding-right: 2.5rem;
    max-width:     none;
}}
@media (max-width: 767px) {{
    .main .block-container {{
        padding-top:   3rem;
        padding-left:  1rem;
        padding-right: 1rem;
    }}
}}

/* ────────────────────────────────────────────────
   SIDEBAR
──────────────────────────────────────────────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {{
    background:   var(--surface);
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] p           {{ color: var(--text-2); font-size: 0.82rem; }}
section[data-testid="stSidebar"] label       {{ color: var(--text-1); font-size: 0.85rem; }}
section[data-testid="stSidebar"] .stCaption  {{ color: var(--text-3); font-size: 0.78rem; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color:          var(--text-3) !important;
    font-size:      0.6rem !important;
    font-weight:    700 !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin:         6px 0 2px;
}}
section[data-testid="stSidebar"] hr {{ border-color: var(--border); margin: 8px 0; }}

/* Orange radio indicator for selected nav item */
section[data-testid="stSidebar"] [data-baseweb="radio"] svg {{ fill: var(--orange) !important; }}

/* Hide the collapse «‹» button inside the sidebar so users can't collapse it */
[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}

/* ────────────────────────────────────────────────
   TABS — underline style
──────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{
    background:    transparent !important;
    border-radius: 0 !important;
    gap:           0 !important;
    padding:       0 !important;
    border-bottom: 1px solid var(--border) !important;
}}
[data-baseweb="tab"] {{
    border-radius: 0 !important;
    background:    transparent !important;
    padding:       10px 20px !important;
    color:         var(--text-3) !important;
    font-weight:   500 !important;
    font-size:     0.88rem !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    transition:    color 0.15s !important;
}}
[data-baseweb="tab"]:hover  {{ color: var(--text-2) !important; }}
[aria-selected="true"] {{
    background:    transparent !important;
    color:         var(--orange) !important;
    border-bottom: 2px solid var(--orange) !important;
}}

/* ────────────────────────────────────────────────
   METRIC CARDS
──────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background:    var(--surface);
    border-radius: var(--radius);
    border:        1px solid var(--border);
    border-left:   3px solid var(--orange);
    box-shadow:    var(--shadow);
    padding:       20px 22px;
    transition:    box-shadow 0.2s, transform 0.2s;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: var(--shadow-hover);
    transform:  translateY(-1px);
}}
[data-testid="stMetricLabel"] {{
    color:          var(--text-3) !important;
    font-size:      0.65rem !important;
    font-weight:    700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom:  4px;
}}
[data-testid="stMetricValue"] {{
    color:       var(--black) !important;
    font-size:   1.25rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}}

/* ────────────────────────────────────────────────
   BUTTONS
──────────────────────────────────────────────── */
[data-testid="baseButton-primary"],
[data-testid="baseButton-primaryFormSubmit"] {{
    background:     var(--orange) !important;
    color:          #FFFFFF !important;
    border:         none !important;
    border-radius:  8px !important;
    font-weight:    600 !important;
    letter-spacing: 0.01em !important;
    transition:     opacity 0.15s, transform 0.15s !important;
}}
[data-testid="baseButton-primary"]:hover,
[data-testid="baseButton-primaryFormSubmit"]:hover {{
    opacity:   0.88 !important;
    transform: translateY(-1px) !important;
}}
[data-testid="baseButton-secondary"],
[data-testid="baseButton-secondaryFormSubmit"] {{
    background:    #FFFFFF !important;
    color:         var(--text-1) !important;
    border:        1px solid #DEDEDE !important;
    border-radius: 8px !important;
    font-weight:   500 !important;
}}
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-secondaryFormSubmit"]:hover {{
    border-color: #BBBBBB !important;
    background:   #FAFAFA !important;
}}
[data-testid="stDownloadButton"] > button {{
    background:    var(--orange) !important;
    color:         #FFFFFF !important;
    border:        none !important;
    border-radius: 8px !important;
    font-weight:   600 !important;
}}
[data-testid="stDownloadButton"] > button:hover {{ opacity: 0.88 !important; }}

/* ────────────────────────────────────────────────
   INPUTS — orange focus ring
──────────────────────────────────────────────── */
input[type="text"]:focus,
input[type="number"]:focus,
input[type="password"]:focus,
textarea:focus {{
    border-color: var(--orange) !important;
    box-shadow:   0 0 0 2px rgba(244,121,32,0.15) !important;
    outline:      none !important;
}}
[data-testid="stFileUploadDropzone"] {{
    border:        2px dashed #E0E0E0;
    border-radius: var(--radius);
    background:    #FAFAFA;
    transition:    border-color 0.15s, background 0.15s;
}}
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: var(--orange);
    background:   #FFF8F4;
}}

/* ────────────────────────────────────────────────
   EXPANDERS
──────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background:    var(--surface);
    border:        1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow:    none;
}}

/* ────────────────────────────────────────────────
   ALERTS
──────────────────────────────────────────────── */
[data-testid="stAlert"] {{ border-radius: 8px; }}
</style>
"""

# ─── AUTH GATE ─────────────────────────────────────────────────────────────────
if "_auth_ok"    not in st.session_state: st.session_state._auth_ok    = False
if "_user_name"  not in st.session_state: st.session_state._user_name  = ""
if "_username"   not in st.session_state: st.session_state._username   = ""

if not st.session_state._auth_ok:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
[data-testid="stAppViewContainer"] {{ background: #F8F8F8 !important; }}
[data-testid="stHeader"]  {{ display: none !important; }}
[data-testid="stSidebar"] {{ display: none !important; }}
footer                    {{ display: none !important; }}
.main .block-container {{
    max-width: 420px;
    padding-top: 8vh;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    margin: 0 auto;
}}
[data-testid="stForm"] {{
    background: #FFFFFF;
    border: 1px solid #EFEFEF;
    border-radius: 16px;
    padding: 36px 36px 28px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.07);
}}
[data-testid="stTextInput"] > div > div > input {{
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    color: #111111;
    padding: 10px 14px;
}}
[data-testid="stTextInput"] > div > div > input:focus {{
    border-color: {_ORANGE};
    box-shadow: 0 0 0 2px rgba(244,121,32,0.15);
    outline: none;
}}
[data-testid="stTextInput"] label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    color: #555555;
}}
[data-testid="baseButton-secondaryFormSubmit"],
[data-testid="baseButton-primaryFormSubmit"] {{
    background: {_ORANGE} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s !important;
}}
[data-testid="baseButton-secondaryFormSubmit"]:hover,
[data-testid="baseButton-primaryFormSubmit"]:hover {{ opacity: 0.88 !important; }}
</style>
""", unsafe_allow_html=True)

    # Login layout
    if _LOGO_B64:
        st.markdown(
            f"<div style='text-align:center;margin-bottom:20px'>"
            f"<img src='data:image/png;base64,{_LOGO_B64}' "
            f"style='width:76px;border-radius:50%'></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<h3 style='text-align:center;font-family:Inter,sans-serif;font-weight:700;"
        "color:#111;margin:0 0 6px;font-size:1.35rem'>Tech Strategy Lab</h3>"
        "<p style='text-align:center;font-family:Inter,sans-serif;color:#AAAAAA;"
        "font-size:0.84rem;margin-bottom:24px;font-weight:400'>Sign in to continue</p>",
        unsafe_allow_html=True,
    )
    # Detect language from saved settings for login form
    try:
        from engine.settings_store import load_settings as _ls
        from engine.i18n import t as _t
        _login_lang = _ls().get("language", "en")
    except Exception:
        _login_lang = "en"
        _t = lambda k, l="en": k  # noqa

    with st.form("login_form"):
        full_name = st.text_input(
            _t("full_name", _login_lang),
            placeholder=_t("name_placeholder", _login_lang))
        username  = st.text_input(_t("username", _login_lang))
        password  = st.text_input(_t("password", _login_lang), type="password")
        submitted = st.form_submit_button(
            _t("sign_in", _login_lang), use_container_width=True)

    if submitted:
        if not full_name.strip():
            st.error(_t("name_required", _login_lang))
        elif username == _USER and password == _PASS:
            st.session_state._auth_ok   = True
            st.session_state._user_name = full_name.strip()
            st.session_state._username  = username
            log_login(name=full_name.strip(), username=username)
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown(
        f"<p style='text-align:center;font-size:0.75rem;color:#CCCCCC;"
        f"font-family:Inter,sans-serif;margin-top:20px'>"
        f"<a href='mailto:{_EMAIL}' style='color:{_ORANGE};text-decoration:none'>"
        f"{_EMAIL}</a></p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ─── INJECT GLOBAL CSS ─────────────────────────────────────────────────────────
st.markdown(_CSS, unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
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

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
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

# ─── SETTINGS ──────────────────────────────────────────────────────────────────
_settings = load_settings()
_lang     = _settings.get("language", "en")

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────

# ── Logo + brand centered ─────────────────────────────────────────────────────
logo_html = (
    f"<img src='data:image/png;base64,{_LOGO_B64}' "
    f"style='width:82px;border-radius:50%;display:inline-block'>"
    if _LOGO_B64 else
    "<div style='width:82px;height:82px;border-radius:50%;background:#F8F8F8;"
    "display:inline-flex;align-items:center;justify-content:center;"
    "font-size:2rem'>🧬</div>"
)
st.sidebar.markdown(
    f"<div style='text-align:center;padding:28px 16px 16px'>"
    f"{logo_html}"
    f"<div style='margin-top:10px;font-family:Inter,sans-serif;font-weight:700;"
    f"font-size:0.9rem;color:#111111;letter-spacing:-0.01em'>Tech Strategy Lab</div>"
    f"<div style='font-family:Inter,sans-serif;font-size:0.68rem;color:#AAAAAA;"
    f"margin-top:2px;font-weight:400'>Analytics Platform</div>"
    f"<div style='font-family:Inter,sans-serif;font-size:0.72rem;color:#555555;"
    f"margin-top:6px;font-weight:500'>{st.session_state._user_name}</div>"
    f"</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

# ── Navigation ────────────────────────────────────────────────────────────────
_nav_keys = [
    "nav_dashboard", "nav_sim_lab", "nav_add_brand",
    "nav_update_brand", "nav_settings", "nav_user_log", "nav_docs",
]
_nav_labels = [t(k, _lang) for k in _nav_keys]

if "nav_page_idx" not in st.session_state:
    st.session_state.nav_page_idx = 0

_page_idx = st.sidebar.radio(
    "Navigate",
    options=list(range(len(_nav_labels))),
    format_func=lambda i: _nav_labels[i],
    key="nav_page_idx",
    label_visibility="collapsed",
)
page = _nav_labels[_page_idx]
st.sidebar.divider()

_is_analytics = page in (t("nav_dashboard", _lang), t("nav_sim_lab", _lang))

# ── Analytics-only sidebar controls ──────────────────────────────────────────
if _is_analytics:
    st.sidebar.header("Market Resolution")
    res_level = st.sidebar.radio(
        "Granularity", ["Monthly", "Weekly", "Daily"], key="ui_res_level")
    time_col = ("Month"     if res_level == "Monthly"
                 else "Week" if res_level == "Weekly" else "DayOfYear")

    st.sidebar.markdown(
        "<p style='font-size:0.75rem;font-weight:600;color:#555;margin:8px 0 2px;"
        "font-family:Inter,sans-serif'>DNA Brands</p>",
        unsafe_allow_html=True,
    )
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

    st.sidebar.divider()
    st.sidebar.header("Trial Reality")
    t_start = st.sidebar.date_input("Start Date", key="ui_t_start")
    t_end   = st.sidebar.date_input("End Date",   key="ui_t_end")
    c_val   = st.sidebar.number_input("Clicks",   min_value=0.0, key="ui_c_val")
    q_val   = st.sidebar.number_input("Quantity", min_value=0.0, key="ui_q_val")
    s_val   = st.sidebar.number_input("Sales",    min_value=0.0, key="ui_s_val")

    if t_start.year < min_data_yr or t_end.year > max_data_yr + 2:
        st.sidebar.warning(f"Outside data range ({min_data_yr}–{max_data_yr}).")

    with st.sidebar.expander("Pre-Adjustment"):
        st.caption("+ % = boosted trial (strip lift).  − % = suppressed (add lift back).")
        st.number_input("Clicks adj (%)",   -100.0, 500.0, key="ui_adj_c", step=5.0)
        st.number_input("Quantity adj (%)", -100.0, 500.0, key="ui_adj_q", step=5.0)
        st.number_input("Sales adj (%)",    -100.0, 500.0, key="ui_adj_s", step=5.0)

    adj_c = c_val / (1 + st.session_state.ui_adj_c / 100) if (1 + st.session_state.ui_adj_c / 100) != 0 else c_val
    adj_q = q_val / (1 + st.session_state.ui_adj_q / 100) if (1 + st.session_state.ui_adj_q / 100) != 0 else q_val
    adj_s = s_val / (1 + st.session_state.ui_adj_s / 100) if (1 + st.session_state.ui_adj_s / 100) != 0 else s_val

    proj_year    = str(t_start.year)
    norm_weights = compute_similarity_weights(
        profiles, sel_brands, proj_year, t_start, t_end, c_val, q_val, s_val)

    st.sidebar.divider()
    st.sidebar.header("DNA Weights")
    st.sidebar.caption("35% — All-time overall")
    for y, w in norm_weights.items():
        st.sidebar.caption(f"{w * 65.0:.1f}% — {y}")

    pure_dna       = build_pure_dna(profiles, sel_brands, norm_weights)
    df, _full_year = build_year_dataframe(int(proj_year))
    build_dna_layers(df, pure_dna, st.session_state.event_log)

    base_clicks, base_cr, base_aov = calibrate_base(df, t_start, t_end, adj_c, adj_q, adj_s)
    if base_clicks is None:
        st.error("Trial date range yields zero DNA sum. Widen the trial period.")
        st.stop()

    build_projections(df, base_clicks, base_cr, base_aov, st.session_state.event_log)

    st.sidebar.divider()
    st.sidebar.header("Export")
    excel_bytes = build_excel_report(
        df, st.session_state.event_log, sel_brands,
        t_start, t_end, adj_c, adj_q, adj_s,
        base_clicks, base_cr, base_aov,
    )
    st.sidebar.download_button(
        label="Download Strategy Report",
        data=excel_bytes,
        file_name=f"strategy_{proj_year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ── Sidebar footer ─────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(
    f"<p style='font-size:0.72rem;color:#CCCCCC;font-family:Inter,sans-serif;"
    f"margin:0 0 10px;text-align:center'>"
    f"<a href='mailto:{_EMAIL}' style='color:{_ORANGE};text-decoration:none'>"
    f"{_EMAIL}</a></p>",
    unsafe_allow_html=True,
)
if st.sidebar.button(t("sign_out", _lang), use_container_width=True):
    st.session_state._auth_ok   = False
    st.session_state._user_name = ""
    st.session_state._username  = ""
    st.rerun()

# ─── PAGE HEADER ───────────────────────────────────────────────────────────────
_subtitles = {
    t("nav_dashboard",    _lang): t("sub_dashboard",    _lang),
    t("nav_sim_lab",      _lang): t("sub_sim_lab",      _lang),
    t("nav_add_brand",    _lang): t("sub_add_brand",    _lang),
    t("nav_update_brand", _lang): t("sub_update_brand", _lang),
    t("nav_settings",     _lang): t("sub_settings",     _lang),
    t("nav_user_log",     _lang): t("sub_user_log",     _lang),
    t("nav_docs",         _lang): t("sub_docs",         _lang),
}
st.markdown(
    f"<div style='margin-bottom:24px'>"
    f"<h1 style='font-family:Inter,sans-serif;font-weight:700;font-size:1.6rem;"
    f"color:{_BLACK};margin:0 0 4px;letter-spacing:-0.025em'>{page}</h1>"
    f"<p style='font-family:Inter,sans-serif;font-size:0.84rem;color:#AAAAAA;"
    f"margin:0;font-weight:400'>{_subtitles.get(page,'')}</p>"
    f"<div style='height:1px;background:#EBEBEB;margin-top:16px'></div></div>",
    unsafe_allow_html=True,
)

# ─── PAGE ROUTING ──────────────────────────────────────────────────────────────
if page == t("nav_dashboard", _lang):
    render_dashboard(
        df, profiles, yearly_kpis,
        sel_brands, res_level, time_col,
        base_cr, base_aov,
    )
elif page == t("nav_sim_lab", _lang):
    render_lab(
        df, df_raw, sel_brands, res_level, time_col,
        base_clicks, base_cr, base_aov,
        adj_c, adj_q, adj_s,
        t_start, t_end, pure_dna,
        settings=_settings,
    )
elif page == t("nav_add_brand", _lang):
    render_brand_add()
elif page == t("nav_update_brand", _lang):
    render_brand_update()
elif page == t("nav_settings", _lang):
    render_settings(lang=_lang)
elif page == t("nav_user_log", _lang):
    render_user_log()
elif page == t("nav_docs", _lang):
    render_docs(lang=_lang)
