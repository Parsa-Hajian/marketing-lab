"""Update Brand page — replace DNA data for an existing brand."""
import pandas as pd
import streamlit as st

from config import PROFILES_PATH, DATASET_PATH
from engine.brand_manager import validate_upload, save_brand


def render_brand_update():
    st.header("Update Brand")
    st.caption("Replace the historical data and recompute DNA for an existing brand.")
    st.markdown("---")

    # Load existing brands
    try:
        profiles = pd.read_csv(PROFILES_PATH)
        profiles["brand"] = profiles["brand"].str.strip().str.lower()
        existing = sorted(profiles["brand"].unique())
    except Exception as e:
        st.error(f"Could not load profiles: {e}")
        return

    if not existing:
        st.info("No brands found in the data.")
        return

    # ── Brand selector ────────────────────────────────────────────────────────
    col_sel, col_info = st.columns([2, 2])
    with col_sel:
        display_name = st.selectbox(
            "Select Brand",
            options=[b.title() for b in existing],
        )
        brand_key = display_name.lower()

    # Current data summary
    brand_profiles = profiles[profiles["brand"] == brand_key]
    levels_present = sorted(brand_profiles["Level"].unique())
    years_present  = sorted([y for y in brand_profiles["Year"].unique() if y != "Overall"])
    with col_info:
        st.markdown(f"""
<div style='background:#F5F5F5;border-radius:8px;padding:12px 16px;margin-top:28px'>
  <div style='font-size:0.78rem;color:#888;font-weight:500'>CURRENT DATA</div>
  <div style='font-size:0.88rem;color:#111;margin-top:4px'>
    <b>Levels:</b> {', '.join(levels_present)}<br>
    <b>Years:</b> {', '.join(years_present) if years_present else '—'}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Granularity ───────────────────────────────────────────────────────────
    st.markdown("**Recompute DNA at**")
    cg1, cg2, cg3 = st.columns(3)
    do_monthly = cg1.checkbox("Monthly", value="Monthly" in levels_present, key="upd_m")
    do_weekly  = cg2.checkbox("Weekly",  value="Weekly"  in levels_present, key="upd_w")
    do_daily   = cg3.checkbox("Daily",   value="Daily"   in levels_present, key="upd_d")

    # ── Current data preview ──────────────────────────────────────────────────
    try:
        dataset = pd.read_csv(DATASET_PATH)
        dataset["brand"] = dataset["brand"].str.strip().str.lower()
        cur = dataset[dataset["brand"] == brand_key].copy()
        if not cur.empty:
            cur["Date"] = pd.to_datetime(cur["Date"])
            with st.expander("Current Data"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Clicks",   f"{cur['clicks'].sum():,.0f}")
                c2.metric("Quantity", f"{cur['quantity'].sum():,.0f}")
                c3.metric("Sales",    f"€{cur['sales'].sum():,.0f}")
                st.dataframe(cur.head(20), use_container_width=True)
    except Exception:
        pass

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("**Upload Replacement Data**")
    uploaded = st.file_uploader(
        "Drag & drop or click — CSV or Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
        help="Required columns: Date, Clicks, Quantity, Sales",
    )

    if uploaded is None:
        st.info("Upload a file to replace **all** existing data for this brand.")
        return

    # Parse
    try:
        raw_df = (
            pd.read_excel(uploaded)
            if uploaded.name.endswith(".xlsx")
            else pd.read_csv(uploaded)
        )
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    ok, msg = validate_upload(raw_df)
    if not ok:
        st.error(msg)
        return

    raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    # ── Preview new data ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("New Rows",      f"{len(raw_df):,}")
    c2.metric("New Sales",     f"€{raw_df['Sales'].sum():,.0f}")
    c3.metric("Date range",    f"{raw_df['Date'].min().date()} → {raw_df['Date'].max().date()}")

    with st.expander("Preview New Data (first 30 rows)"):
        st.dataframe(raw_df.head(30), use_container_width=True)

    # ── Submit ────────────────────────────────────────────────────────────────
    levels = (
        (["Monthly"] if do_monthly else []) +
        (["Weekly"]  if do_weekly  else []) +
        (["Daily"]   if do_daily   else [])
    )
    if not levels:
        st.warning("Select at least one granularity.")
        return

    st.markdown("---")
    st.warning(f"This will permanently replace all data for **{brand_key}**.")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Confirm Update", type="primary", use_container_width=True):
            with st.spinner("Updating brand data…"):
                success, message = save_brand(
                    brand_key, raw_df, levels,
                    PROFILES_PATH, DATASET_PATH,
                    overwrite=True,
                )
            if success:
                st.success(message)
                st.info("Press **R** to reload the app with updated data.")
                st.cache_data.clear()
            else:
                st.error(message)
