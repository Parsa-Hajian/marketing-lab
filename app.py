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

st.set_page_config(page_title="Tech Strategy Lab", layout="wide", page_icon="🧬")

# ─── SESSION STATE ──────────────────────────────────────────────────────────────

if "event_log"        not in st.session_state: st.session_state.event_log        = []
if "shock_library"    not in st.session_state: st.session_state.shock_library    = []
if "affect_future"    not in st.session_state: st.session_state.affect_future    = False
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
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw["brand"] = df_raw["brand"].str.strip().str.lower()
    return profiles, yearly_kpis, df_raw

profiles, yearly_kpis, df_raw = load_data()

data_years   = sorted([int(y) for y in profiles["Year"].unique() if y != "Overall"])
min_data_yr  = data_years[0]
max_data_yr  = data_years[-1]

# ─── TITLE ──────────────────────────────────────────────────────────────────────

st.title("🧬 Tech Strategy Lab")

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────

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

st.sidebar.markdown("---")
st.session_state.affect_future = st.sidebar.checkbox(
    "✨ Affect Future (Apply events to projections & Goal Tracker)",
    value=st.session_state.affect_future,
)

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

pure_dna          = build_pure_dna(profiles, sel_brands, norm_weights)
df, _full_year    = build_year_dataframe(int(proj_year))
build_dna_layers(df, pure_dna, st.session_state.event_log)

# ─── CALIBRATION ───────────────────────────────────────────────────────────────

base_clicks, base_cr, base_aov = calibrate_base(df, t_start, t_end, adj_c, adj_q, adj_s)

if base_clicks is None:
    st.error("Trial date range yields zero DNA sum. Please widen the trial period.")
    st.stop()

# ─── PROJECTIONS ───────────────────────────────────────────────────────────────

build_projections(df, base_clicks, base_cr, base_aov, st.session_state.event_log)

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
