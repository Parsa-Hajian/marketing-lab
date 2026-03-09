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
from engine.activity_log import log_action
from engine.settings_store import load_settings
from engine.i18n import t
from engine.noise import apply_noise_bands
from views.dashboard import render_goal_tracker
from views.lab import (
    render_brand_select, render_edit_dna, render_trial_data,
    render_campaigns, render_audit, render_download,
)
from views.risk import render_risk
from views.brand_add import render_brand_add
from views.brand_update import render_brand_update
from views.user_log import render_user_log
from views.settings import render_settings
from views.docs import render_docs
from views.brand_forge import render_brand_forge
from views.monitor import render_monitor

st.set_page_config(
    page_title="Tech Strategy Lab",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
_ORANGE = "#F47920"
_BLACK  = "#111111"

_LOGO_EXISTS = os.path.exists(LOGO_PATH)
def _b64() -> str:
    if not _LOGO_EXISTS:
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()
_LOGO_B64 = _b64()

# ─── WORKFLOW STEPS (8-step sequential order) ────────────────────────────────
_WORKFLOW_STEPS = [
    {"key": "nav_brand_select",  "icon": "🏷️", "num": 1},
    {"key": "nav_edit_dna",      "icon": "🧬", "num": 2},
    {"key": "nav_trial_data",    "icon": "📝", "num": 3},
    {"key": "nav_goal_tracker",  "icon": "🎯", "num": 4},
    {"key": "nav_campaigns",     "icon": "🚀", "num": 5},
    {"key": "nav_risk",          "icon": "📉", "num": 6},
    {"key": "nav_audit",         "icon": "📋", "num": 7},
    {"key": "nav_download",      "icon": "📥", "num": 8},
]
_WORKFLOW_KEYS = {s["key"] for s in _WORKFLOW_STEPS}

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

/* ── Strip Streamlit chrome ── */
#MainMenu                        {{ display: none !important; }}
footer                           {{ display: none !important; }}
[data-testid="stToolbar"]        {{ display: none !important; }}
[data-testid="stDecoration"]     {{ display: none !important; }}
[data-testid="stStatusWidget"]   {{ display: none !important; }}

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

/* ── Sidebar nav buttons — flat, left-aligned ── */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {{
    background:    transparent !important;
    border:        none !important;
    box-shadow:    none !important;
    text-align:    left !important;
    padding:       7px 10px !important;
    color:         #555555 !important;
    font-size:     0.84rem !important;
    font-weight:   400 !important;
    font-family:   'Inter', sans-serif !important;
    border-radius: 7px !important;
    width:         100%;
    transition:    background 0.15s, color 0.15s !important;
    margin:        1px 0 !important;
}}
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {{
    background: rgba(26,26,107,0.07) !important;
    color:      #1a1a6b !important;
    border:     none !important;
}}
/* Active nav item rendered as a div */
.nav-active {{
    display:       block;
    background:    rgba(26,26,107,0.09);
    border-left:   3px solid #1a1a6b;
    border-radius: 0 7px 7px 0;
    color:         #1a1a6b;
    font-family:   'Inter', sans-serif;
    font-size:     0.84rem;
    font-weight:   600;
    padding:       7px 10px 7px 7px;
    margin:        1px 0;
    cursor:        default;
}}
/* Locked nav item */
.nav-locked {{
    display:       block;
    background:    transparent;
    border-radius: 7px;
    color:         #CCCCCC;
    font-family:   'Inter', sans-serif;
    font-size:     0.84rem;
    font-weight:   400;
    padding:       7px 10px;
    margin:        1px 0;
    cursor:        not-allowed;
}}
/* Nav section group labels */
.nav-section {{
    display:        block;
    font-family:    'Inter', sans-serif;
    font-size:      0.62rem;
    font-weight:    700;
    color:          #BBBBBB;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding:        12px 10px 5px;
}}

/* Hide the collapse button */
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
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label {{
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

/* ────────────────────────────────────────────────
   STEP PROGRESS BAR
──────────────────────────────────────────────── */
.step-bar {{
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 18px;
    padding: 10px 0;
}}
.step-item {{
    display: flex;
    align-items: center;
    gap: 0;
}}
.step-dot {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
}}
.step-dot.active {{
    background: {_ORANGE};
    color: #FFFFFF;
}}
.step-dot.done {{
    background: #e0f2fe;
    color: #0284c7;
}}
.step-dot.pending {{
    background: #F0F0F0;
    color: #BBBBBB;
}}
.step-line {{
    width: 24px;
    height: 2px;
    flex-shrink: 0;
}}
.step-line.done {{
    background: #0284c7;
}}
.step-line.pending {{
    background: #E0E0E0;
}}
</style>
"""

# ─── SESSION DEFAULTS (auth removed) ──────────────────────────────────────────
if "_user_name"  not in st.session_state: st.session_state._user_name  = "User"
if "_username"   not in st.session_state: st.session_state._username   = ""

# ─── INJECT GLOBAL CSS ─────────────────────────────────────────────────────────
st.markdown(_CSS, unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "event_log"          not in st.session_state: st.session_state.event_log          = []
if "shift_target_idx"   not in st.session_state: st.session_state.shift_target_idx   = None
if "tgt_start"          not in st.session_state: st.session_state.tgt_start          = date(2026, 1, 1)
if "tgt_end"            not in st.session_state: st.session_state.tgt_end            = date(2026, 12, 31)
if "target_metric"      not in st.session_state: st.session_state.target_metric      = "Sales"
if "target_val"         not in st.session_state: st.session_state.target_val         = 200_000.0
if "ui_res_level"       not in st.session_state: st.session_state.ui_res_level       = "Monthly"
if "ui_t_start"         not in st.session_state: st.session_state.ui_t_start         = date(2026, 1, 1)
if "ui_t_end"           not in st.session_state: st.session_state.ui_t_end           = date(2026, 1, 31)
if "ui_c_val"           not in st.session_state: st.session_state.ui_c_val           = 5_000.0
if "ui_q_val"           not in st.session_state: st.session_state.ui_q_val           = 250.0
if "ui_s_val"           not in st.session_state: st.session_state.ui_s_val           = 12_500.0
if "ui_adj_c"           not in st.session_state: st.session_state.ui_adj_c           = 0.0
if "ui_adj_q"           not in st.session_state: st.session_state.ui_adj_q           = 0.0
if "ui_adj_s"           not in st.session_state: st.session_state.ui_adj_s           = 0.0
if "ui_sel_brands"      not in st.session_state: st.session_state.ui_sel_brands      = []

# Per-step completion tracking
if "step_completed" not in st.session_state:
    st.session_state.step_completed = {s["key"]: False for s in _WORKFLOW_STEPS}

# Cached pipeline results
if "pipeline_cache" not in st.session_state:
    st.session_state.pipeline_cache = {
        "pure_dna": None, "pure_dna_weighted": None,
        "df": None, "df_base": None,
        "norm_weights": None,
        "base_clicks": None, "base_cr": None, "base_aov": None,
        "adj_c": None, "adj_q": None, "adj_s": None,
        "proj_year": None,
        "df_raw_mod": None,
        "profiles_mod": None,
    }

# Activity-log change-detection sentinels
if "_prev_page"         not in st.session_state: st.session_state._prev_page         = ""
if "_prev_res_level"    not in st.session_state: st.session_state._prev_res_level    = ""
if "_prev_sel_brands"   not in st.session_state: st.session_state._prev_sel_brands   = []

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    profiles    = pd.read_csv(PROFILES_PATH)
    profiles["Year"] = profiles["Year"].astype(str)
    yearly_kpis = pd.read_csv(YEARLY_KPI_PATH)
    df_raw      = pd.read_csv(DATASET_PATH)
    df_raw["Date"]  = pd.to_datetime(df_raw["Date"], format="mixed", dayfirst=False)
    df_raw["brand"] = df_raw["brand"].str.strip().str.lower()
    return profiles, yearly_kpis, df_raw

profiles, yearly_kpis, df_raw = load_data()

data_years  = sorted([int(y) for y in profiles["Year"].unique() if y != "Overall"])
min_data_yr = data_years[0]
max_data_yr = data_years[-1]

# ─── SETTINGS ──────────────────────────────────────────────────────────────────
_settings = load_settings()
_lang     = _settings.get("language", "en")


# ─── STEP GATING ──────────────────────────────────────────────────────────────
def _can_access_step(step_key):
    """All prior steps must be completed before accessing this step."""
    order = [s["key"] for s in _WORKFLOW_STEPS]
    idx = order.index(step_key) if step_key in order else -1
    if idx <= 0:
        return True  # Step 1 is always accessible
    return all(st.session_state.step_completed.get(order[i], False) for i in range(idx))


# ─── STEP PROGRESS BAR HELPER ─────────────────────────────────────────────────
def _render_step_bar(current_key):
    """Render a horizontal step progress indicator."""
    current_num = 0
    for s in _WORKFLOW_STEPS:
        if s["key"] == current_key:
            current_num = s["num"]
            break

    dots = []
    for i, s in enumerate(_WORKFLOW_STEPS):
        if st.session_state.step_completed.get(s["key"], False):
            cls = "done"
        elif s["num"] == current_num:
            cls = "active"
        else:
            cls = "pending"
        dots.append(f'<span class="step-dot {cls}" title="{t(s["key"], _lang)}">{s["num"]}</span>')
        if i < len(_WORKFLOW_STEPS) - 1:
            line_cls = "done" if st.session_state.step_completed.get(s["key"], False) else "pending"
            dots.append(f'<span class="step-line {line_cls}"></span>')

    st.markdown(f'<div class="step-bar">{"".join(dots)}</div>', unsafe_allow_html=True)


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
    f"</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

# ── Navigation ────────────────────────────────────────────────────────────────
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "nav_brand_select"

def _nav_btn(key: str, icon: str, label: str, numbered: int = 0, locked: bool = False) -> None:
    """Render a sidebar navigation button with lock/active state."""
    display = f"{numbered}.&nbsp;&nbsp;{icon}&nbsp;&nbsp;{label}" if numbered else f"{icon}&nbsp;&nbsp;{label}"

    # Check mark for completed steps
    is_done = st.session_state.step_completed.get(key, False)
    if is_done and numbered:
        display = f"{numbered}.&nbsp;&nbsp;✓&nbsp;&nbsp;{label}"

    if st.session_state.nav_page == key:
        st.sidebar.markdown(
            f"<span class='nav-active'>{display}</span>",
            unsafe_allow_html=True,
        )
    elif locked:
        st.sidebar.markdown(
            f"<span class='nav-locked'>🔒&nbsp;&nbsp;{numbered}.&nbsp;&nbsp;{label}</span>",
            unsafe_allow_html=True,
        )
    else:
        btn_text = f"{numbered}.  {icon}  {label}" if numbered else f"{icon}  {label}"
        if is_done and numbered:
            btn_text = f"{numbered}.  ✓  {label}"
        if st.sidebar.button(btn_text, key=f"nb_{key}", use_container_width=True):
            st.session_state.nav_page = key
            st.rerun()

# Workflow steps
st.sidebar.markdown("<span class='nav-section'>Strategy Workflow</span>", unsafe_allow_html=True)
for step in _WORKFLOW_STEPS:
    locked = not _can_access_step(step["key"])
    _nav_btn(step["key"], step["icon"], t(step["key"], _lang), numbered=step["num"], locked=locked)

# Analytics section
st.sidebar.markdown("<span class='nav-section'>Analytics</span>", unsafe_allow_html=True)
_nav_btn("nav_monitor", "📊", t("nav_monitor", _lang))

# Brands section
st.sidebar.markdown("<span class='nav-section'>Brands</span>", unsafe_allow_html=True)
_nav_btn("nav_forge",        "🔬", t("nav_forge",        _lang))
_nav_btn("nav_add_brand",    "➕", t("nav_add_brand",    _lang))
_nav_btn("nav_update_brand", "✏️",  t("nav_update_brand", _lang))

# System section
st.sidebar.markdown("<span class='nav-section'>System</span>", unsafe_allow_html=True)
_nav_btn("nav_settings", "⚙️",  t("nav_settings", _lang))
_nav_btn("nav_user_log", "📋", t("nav_user_log", _lang))
_nav_btn("nav_docs",     "📖", t("nav_docs",     _lang))

page_key = st.session_state.nav_page
page     = t(page_key, _lang)

# ── Enforce step gating ──────────────────────────────────────────────────────
if page_key in _WORKFLOW_KEYS and not _can_access_step(page_key):
    # Find the first incomplete step
    order = [s["key"] for s in _WORKFLOW_STEPS]
    first_incomplete = order[0]
    for k in order:
        if not st.session_state.step_completed.get(k, False):
            first_incomplete = k
            break
    st.warning(f"Complete earlier steps before accessing **{page}**.")
    st.session_state.nav_page = first_incomplete
    st.rerun()

# ── Log page navigation ───────────────────────────────────────────────────────
_prev_page = st.session_state._prev_page
if _prev_page and page != _prev_page:
    log_action(
        name=st.session_state._user_name,
        username=st.session_state._username,
        action="Page Navigation",
        details=f"From: {_prev_page} | To: {page}",
    )
st.session_state._prev_page = page

st.sidebar.divider()

_is_analytics = page_key in _WORKFLOW_KEYS

# ── Sidebar: Market Resolution (for steps 2+) ──
if _is_analytics and page_key != "nav_brand_select":
    st.sidebar.header("Market Resolution")
    res_level = st.sidebar.radio(
        "Granularity", ["Monthly", "Weekly", "Daily"], key="ui_res_level")
    time_col = ("Month"     if res_level == "Monthly"
                 else "Week" if res_level == "Weekly" else "DayOfYear")
    _prev_res = st.session_state._prev_res_level
    if _prev_res and res_level != _prev_res:
        log_action(
            name=st.session_state._user_name,
            username=st.session_state._username,
            action="Resolution Changed",
            details=f"From: {_prev_res} | To: {res_level}",
        )
    st.session_state._prev_res_level = res_level
else:
    res_level = st.session_state.ui_res_level
    time_col = ("Month"     if res_level == "Monthly"
                 else "Week" if res_level == "Weekly" else "DayOfYear")

# ── Read cached pipeline results ──────────────────────────────────────────────
sel_brands = st.session_state.ui_sel_brands
cache = st.session_state.pipeline_cache
t_start = st.session_state.ui_t_start
t_end   = st.session_state.ui_t_end
proj_year = cache.get("proj_year") or "2026"

# ── Sidebar footer ─────────────────────────────────────────────────────────────
st.sidebar.divider()

# ─── PAGE HEADER + STEP BAR ──────────────────────────────────────────────────
_subtitles = {}
for s in _WORKFLOW_STEPS:
    _subtitles[t(s["key"], _lang)] = t(f"sub_{s['key'][4:]}", _lang)
_subtitles.update({
    t("nav_add_brand",    _lang): t("sub_add_brand",    _lang),
    t("nav_update_brand", _lang): t("sub_update_brand", _lang),
    t("nav_settings",     _lang): t("sub_settings",     _lang),
    t("nav_user_log",     _lang): t("sub_user_log",     _lang),
    t("nav_forge",        _lang): t("sub_forge",        _lang),
    t("nav_docs",         _lang): t("sub_docs",         _lang),
    t("nav_monitor",      _lang): t("sub_monitor",      _lang),
})

# Step bar for workflow pages
if page_key in _WORKFLOW_KEYS:
    _render_step_bar(page_key)

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
if page_key == "nav_brand_select":
    all_brands = sorted(profiles["brand"].unique())
    render_brand_select(profiles, all_brands)

elif page_key == "nav_edit_dna":
    render_edit_dna(profiles, df_raw, sel_brands, res_level, time_col, _settings)

elif page_key == "nav_trial_data":
    all_brands = sorted(profiles["brand"].unique())
    render_trial_data(profiles, all_brands, min_data_yr, max_data_yr, df_raw)

elif page_key == "nav_goal_tracker":
    df = cache.get("df")
    _dr = cache.get("df_raw_mod") if cache.get("df_raw_mod") is not None else df_raw
    if df is not None:
        render_goal_tracker(
            df, _dr, profiles, yearly_kpis, sel_brands,
            res_level, time_col,
            cache.get("base_cr"), cache.get("base_aov"),
            t_start, t_end,
        )
    else:
        st.warning("Complete Steps 1–3 to access Goal Tracker.")

elif page_key == "nav_campaigns":
    df = cache.get("df")
    _dr = cache.get("df_raw_mod") if cache.get("df_raw_mod") is not None else df_raw
    if df is not None:
        render_campaigns(
            df, _dr, sel_brands, t_start,
            cache.get("base_clicks"), cache.get("base_cr"), cache.get("base_aov"),
            _settings, profiles=profiles,
        )
    else:
        st.warning("Complete Steps 1–3 to access Campaigns.")

elif page_key == "nav_risk":
    df = cache.get("df")
    _dr = cache.get("df_raw_mod") if cache.get("df_raw_mod") is not None else df_raw
    if df is not None:
        render_risk(_dr, sel_brands, df)
    else:
        st.warning("Complete Steps 1–3 to access Risk.")

elif page_key == "nav_audit":
    df = cache.get("df")
    if df is not None:
        render_audit(
            df, cache.get("pure_dna_weighted"),
            cache.get("adj_c"), cache.get("adj_q"), cache.get("adj_s"),
            t_start, t_end,
        )
    else:
        st.warning("Complete Steps 1–3 to access Audit.")

elif page_key == "nav_download":
    df = cache.get("df")
    if df is not None:
        render_download(
            df, st.session_state.event_log, sel_brands,
            t_start, t_end,
            cache.get("adj_c"), cache.get("adj_q"), cache.get("adj_s"),
            cache.get("base_clicks"), cache.get("base_cr"), cache.get("base_aov"),
            proj_year,
        )
    else:
        st.warning("Complete Steps 1–3 to access Download.")

elif page_key == "nav_monitor":
    _pc = cache if cache.get("df") is not None else None
    render_monitor(profiles, df_raw, _lang, pipeline_cache=_pc,
                   event_log=st.session_state.event_log,
                   sel_brands=sel_brands)

elif page_key == "nav_forge":
    render_brand_forge(profiles)

elif page_key == "nav_add_brand":
    render_brand_add()

elif page_key == "nav_update_brand":
    render_brand_update()

elif page_key == "nav_settings":
    render_settings(lang=_lang)

elif page_key == "nav_user_log":
    render_user_log()

elif page_key == "nav_docs":
    render_docs(lang=_lang)

# ── Next Step navigation (workflow pages only) ────────────────────────────────
if page_key in _WORKFLOW_KEYS:
    current_idx = next(
        (i for i, s in enumerate(_WORKFLOW_STEPS) if s["key"] == page_key), -1
    )
    if current_idx < len(_WORKFLOW_STEPS) - 1:
        next_step = _WORKFLOW_STEPS[current_idx + 1]
        st.markdown("---")
        col_prev, col_spacer, col_next = st.columns([1, 3, 1])
        if current_idx > 0:
            prev_step = _WORKFLOW_STEPS[current_idx - 1]
            if col_prev.button(
                f"← {t(prev_step['key'], _lang)}",
                key="nav_prev_step",
                use_container_width=True,
            ):
                st.session_state.nav_page = prev_step["key"]
                st.rerun()
        # Only enable Next if current step is complete
        can_next = st.session_state.step_completed.get(page_key, False)
        if can_next:
            if col_next.button(
                f"Next: {t(next_step['key'], _lang)} →",
                key="nav_next_step",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.nav_page = next_step["key"]
                st.rerun()
        else:
            col_next.button(
                f"Next: {t(next_step['key'], _lang)} →",
                key="nav_next_step",
                type="primary",
                use_container_width=True,
                disabled=True,
            )
    elif current_idx > 0:
        st.markdown("---")
        col_prev, _ = st.columns([1, 4])
        prev_step = _WORKFLOW_STEPS[current_idx - 1]
        if col_prev.button(
            f"← {t(prev_step['key'], _lang)}",
            key="nav_prev_step",
            use_container_width=True,
        ):
            st.session_state.nav_page = prev_step["key"]
            st.rerun()
