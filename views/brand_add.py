"""Add Brand page — upload raw data for a new brand."""
import io
import pandas as pd
import streamlit as st

from config import PROFILES_PATH, DATASET_PATH
from engine.brand_manager import validate_upload, save_brand
from engine.activity_log import log_action


_TEMPLATE = pd.DataFrame({
    "Date":     ["2024-01-01", "2024-01-02", "2024-01-03"],
    "Clicks":   [1200,  950, 1100],
    "Quantity": [60,     45,   55],
    "Sales":    [3000.0, 2250.0, 2750.0],
})


def _template_csv() -> bytes:
    buf = io.BytesIO()
    _TEMPLATE.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()


def render_brand_add():
    st.header("Add Brand")
    st.caption(
        "Upload historical data for a new brand. "
        "The app will compute DNA indices automatically."
    )
    st.markdown("---")

    # ── Brand name + template download ────────────────────────────────────────
    col_name, col_tpl = st.columns([3, 1])
    with col_name:
        brand_name = st.text_input("Brand Name", placeholder="e.g. Nike, Zara…")
    with col_tpl:
        st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
        st.download_button(
            "Download CSV Template",
            data=_template_csv(),
            file_name="brand_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Granularity ───────────────────────────────────────────────────────────
    st.markdown("**Compute DNA at**")
    cg1, cg2, cg3 = st.columns(3)
    do_monthly = cg1.checkbox("Monthly", value=True)
    do_weekly  = cg2.checkbox("Weekly",  value=True)
    do_daily   = cg3.checkbox("Daily",   value=False)

    # ── Custom date-range filter ──────────────────────────────────────────────
    with st.expander("Custom Date Range Filter (optional)"):
        st.caption("Restrict DNA computation to a specific window.")
        cd1, cd2 = st.columns(2)
        with cd1:
            filter_start = st.date_input("From", value=None, key="add_fs")
        with cd2:
            filter_end   = st.date_input("To",   value=None, key="add_fe")

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("**Upload Data**")
    uploaded = st.file_uploader(
        "Drag & drop or click — CSV or Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
        help="Required columns: Date, Clicks, Quantity, Sales",
    )

    if uploaded is None:
        st.info("Upload a file with columns: **Date, Clicks, Quantity, Sales**")
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

    # Validate
    ok, msg = validate_upload(raw_df)
    if not ok:
        st.error(msg)
        return

    raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    # Apply date filter
    if filter_start and filter_end:
        raw_df = raw_df[
            (raw_df["Date"] >= pd.Timestamp(filter_start)) &
            (raw_df["Date"] <= pd.Timestamp(filter_end))
        ]
        if raw_df.empty:
            st.warning("No data in the selected date range.")
            return

    # ── Preview ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",        f"{len(raw_df):,}")
    c2.metric("Date range",  f"{raw_df['Date'].min().date()} → {raw_df['Date'].max().date()}")
    c3.metric("Total Sales", f"€{raw_df['Sales'].sum():,.0f}")

    with st.expander("Preview (first 30 rows)"):
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
    if not brand_name.strip():
        st.warning("Enter a brand name.")
        return

    st.markdown("---")
    if st.button("Add Brand", type="primary"):
        with st.spinner("Computing DNA indices and saving…"):
            success, message = save_brand(
                brand_name, raw_df, levels,
                PROFILES_PATH, DATASET_PATH,
                overwrite=False,
            )
        if success:
            log_action(
                name=st.session_state.get("_user_name", "Unknown"),
                username=st.session_state.get("_username", ""),
                action="Add Brand",
                details=f"Brand: {brand_name.strip().lower()} | Rows: {len(raw_df):,} | Levels: {', '.join(levels)}",
            )
            st.success(message)
            st.info("Press **R** to reload the app and see the new brand in the selector.")
            st.cache_data.clear()
        else:
            st.error(message)
