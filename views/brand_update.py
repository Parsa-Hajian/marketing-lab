"""Update Brand page — replace or extend DNA data for an existing brand."""
import pandas as pd
import streamlit as st

from config import PROFILES_PATH, DATASET_PATH
from engine.brand_manager import (
    validate_upload,
    save_brand,
    save_brand_append,
    load_raw_for_brand,
)

_MODE_REPLACE = "Replace all data"
_MODE_APPEND  = "Add new records"


def render_brand_update():
    st.header("Update Brand")
    st.caption("Modify the historical data and recompute DNA for an existing brand.")
    st.markdown("---")

    # ── Load brands ───────────────────────────────────────────────────────────
    try:
        profiles = pd.read_csv(PROFILES_PATH)
        profiles["brand"] = profiles["brand"].str.strip().str.lower()
        existing_brands = sorted(profiles["brand"].unique())
    except Exception as e:
        st.error(f"Could not load profiles: {e}")
        return

    if not existing_brands:
        st.info("No brands found in the data.")
        return

    # ── Brand selector + current summary ─────────────────────────────────────
    col_sel, col_info = st.columns([2, 2])
    with col_sel:
        display_name = st.selectbox(
            "Select Brand",
            options=[b.title() for b in existing_brands],
        )
        brand_key = display_name.lower()

    brand_profiles = profiles[profiles["brand"] == brand_key]
    levels_present = sorted(brand_profiles["Level"].unique())
    years_present  = sorted([y for y in brand_profiles["Year"].unique() if y != "Overall"])

    with col_info:
        st.markdown(
            f"<div style='background:#F5F5F5;border-radius:8px;"
            f"padding:12px 16px;margin-top:28px'>"
            f"<div style='font-size:0.75rem;color:#888;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.05em'>Current data</div>"
            f"<div style='font-size:0.88rem;color:#111;margin-top:6px'>"
            f"<b>Levels:</b> {', '.join(levels_present)}<br>"
            f"<b>Years:</b> {', '.join(years_present) if years_present else '—'}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Update mode ───────────────────────────────────────────────────────────
    mode = st.radio(
        "Update mode",
        [_MODE_REPLACE, _MODE_APPEND],
        horizontal=True,
        help=(
            f"**{_MODE_REPLACE}**: removes ALL existing records for this brand "
            f"and replaces them with your upload.\n\n"
            f"**{_MODE_APPEND}**: keeps existing records; new records are added. "
            f"If a date already exists, the new value overwrites the old one."
        ),
    )

    # Show current raw data preview for context
    try:
        existing_raw = load_raw_for_brand(brand_key, DATASET_PATH)
        if not existing_raw.empty:
            with st.expander(f"Current data — {len(existing_raw):,} rows"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Clicks",   f"{existing_raw['Clicks'].sum():,.0f}")
                c2.metric("Quantity", f"{existing_raw['Quantity'].sum():,.0f}")
                c3.metric("Sales",    f"€{existing_raw['Sales'].sum():,.0f}")
                st.dataframe(existing_raw.head(20), use_container_width=True)
    except Exception:
        pass

    # ── Granularity ───────────────────────────────────────────────────────────
    st.markdown("**Recompute DNA at**")
    cg1, cg2, cg3 = st.columns(3)
    do_monthly = cg1.checkbox("Monthly", value="Monthly" in levels_present, key="upd_m")
    do_weekly  = cg2.checkbox("Weekly",  value="Weekly"  in levels_present, key="upd_w")
    do_daily   = cg3.checkbox("Daily",   value="Daily"   in levels_present, key="upd_d")

    # ── File upload ───────────────────────────────────────────────────────────
    lbl = "Replacement data" if mode == _MODE_REPLACE else "New records to add"
    st.markdown(f"**{lbl}**")
    uploaded = st.file_uploader(
        "Drag & drop or click — CSV or Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
        help="Required columns: Date, Clicks, Quantity, Sales",
    )

    if uploaded is None:
        if mode == _MODE_REPLACE:
            st.info("Upload a file to **replace all** existing data for this brand.")
        else:
            st.info(
                "Upload a file with new records. "
                "Existing dates are kept unless overwritten by the new data."
            )
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

    # ── Preview ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in upload", f"{len(raw_df):,}")
    c2.metric("Sales (upload)", f"€{raw_df['Sales'].sum():,.0f}")
    c3.metric(
        "Date range",
        f"{raw_df['Date'].min().date()} → {raw_df['Date'].max().date()}",
    )

    if mode == _MODE_APPEND:
        # Show overlap stats
        try:
            existing_raw = load_raw_for_brand(brand_key, DATASET_PATH)
            existing_dates = set(existing_raw["Date"].dt.date)
            new_dates      = set(raw_df["Date"].dt.date)
            overlap = len(existing_dates & new_dates)
            truly_new = len(new_dates - existing_dates)
            st.info(
                f"**{truly_new:,}** new dates will be added — "
                f"**{overlap:,}** overlapping dates will be overwritten."
            )
        except Exception:
            pass

    with st.expander("Preview upload (first 30 rows)"):
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
    if mode == _MODE_REPLACE:
        st.warning(f"⚠️ This will permanently replace **all** data for **{brand_key}**.")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        label = "Confirm Replace" if mode == _MODE_REPLACE else "Confirm Add Records"
        if st.button(label, type="primary", use_container_width=True):
            with st.spinner("Processing…"):
                if mode == _MODE_REPLACE:
                    success, message = save_brand(
                        brand_key, raw_df, levels,
                        PROFILES_PATH, DATASET_PATH,
                        overwrite=True,
                    )
                else:
                    success, message = save_brand_append(
                        brand_key, raw_df, levels,
                        PROFILES_PATH, DATASET_PATH,
                    )
            if success:
                st.success(message)
                st.info("Press **R** to reload the app with updated data.")
                st.cache_data.clear()
            else:
                st.error(message)
