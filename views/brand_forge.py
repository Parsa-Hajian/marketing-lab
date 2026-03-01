"""Brand Forge — synthesise a new brand from existing DNA profiles."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

from engine.brand_manager import save_brand
from engine.activity_log import log_action
from config import PROFILES_PATH, DATASET_PATH

_TEMPLATE = "plotly_white"
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_monthly_dna(profiles: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Return 12-row monthly DNA for brand (Overall first, fallback to all years)."""
    m_dna = profiles[(profiles["brand"] == brand) & (profiles["Level"] == "Monthly")]
    overall = (
        m_dna[m_dna["Year"] == "Overall"]
        .groupby("TimeIdx")
        .agg({"idx_clicks": "median", "idx_cr": "median", "idx_aov": "median"})
        .reset_index()
        .sort_values("TimeIdx")
    )
    if overall.empty:
        overall = (
            m_dna
            .groupby("TimeIdx")
            .agg({"idx_clicks": "median", "idx_cr": "median", "idx_aov": "median"})
            .reset_index()
            .sort_values("TimeIdx")
        )
    if overall.empty:
        overall = pd.DataFrame({
            "TimeIdx":    list(range(1, 13)),
            "idx_clicks": [1.0] * 12,
            "idx_cr":     [1.0] * 12,
            "idx_aov":    [1.0] * 12,
        })
    return overall


def _blend_dna(primary: pd.DataFrame, secondary: pd.DataFrame,
               weight: float) -> pd.DataFrame:
    """Blend two 12-row monthly DNA frames; weight applies to secondary."""
    merged = primary.merge(
        secondary[["TimeIdx", "idx_clicks", "idx_cr", "idx_aov"]],
        on="TimeIdx", suffixes=("_p", "_s"), how="left",
    ).fillna(1.0)
    out = primary[["TimeIdx"]].copy()
    for col in ["idx_clicks", "idx_cr", "idx_aov"]:
        out[col] = merged[f"{col}_p"] * (1 - weight) + merged[f"{col}_s"] * weight
    return out


def _apply_multipliers(dna: pd.DataFrame,
                       mult_c: list, mult_cr: list, mult_aov: list) -> pd.DataFrame:
    """Multiply each month's DNA by user-supplied multipliers."""
    out = dna.copy()
    for idx_row, row in out.iterrows():
        m = int(row["TimeIdx"]) - 1
        if 0 <= m < 12:
            out.loc[idx_row, "idx_clicks"] *= mult_c[m]
            out.loc[idx_row, "idx_cr"]     *= mult_cr[m]
            out.loc[idx_row, "idx_aov"]    *= mult_aov[m]
    return out


def _generate_synthetic(dna: pd.DataFrame,
                        data_start: date, data_end: date,
                        annual_clicks: int, annual_orders: int,
                        annual_revenue: float, noise: float) -> pd.DataFrame:
    """Generate daily synthetic data from monthly DNA and annual targets."""
    dates = pd.date_range(start=data_start, end=data_end)
    df = pd.DataFrame({"Date": dates})
    df["Month"] = df["Date"].dt.month

    df = df.merge(
        dna[["TimeIdx", "idx_clicks", "idx_cr", "idx_aov"]],
        left_on="Month", right_on="TimeIdx", how="left",
    ).fillna(1.0)

    n_years = max((data_end - data_start).days / 365.25, 1 / 365)

    # ── Clicks: distribute annual total by seasonal weight ──
    clicks_sum = df["idx_clicks"].sum()
    df["Clicks"] = (
        annual_clicks * n_years * df["idx_clicks"] / max(clicks_sum, 1e-9)
    ).round().astype(int).clip(0)

    # ── CR and AOV: scale from annual ratios, shaped by index ──
    base_cr  = annual_orders  / max(annual_clicks, 1)
    base_aov = annual_revenue / max(annual_orders, 1)

    df["CR"]  = (base_cr  * df["idx_cr"]  / df["idx_cr"].mean()).clip(0, 1)
    df["AOV"] = (base_aov * df["idx_aov"] / df["idx_aov"].mean()).clip(0)

    # ── Log-normal noise ──
    if noise > 0:
        rng = np.random.default_rng(42)
        df["Clicks"] = (df["Clicks"] * np.exp(rng.normal(0, noise, len(df)))).round().astype(int).clip(0)
        df["CR"]     = (df["CR"]     * np.exp(rng.normal(0, noise, len(df)))).clip(0, 1)
        df["AOV"]    = (df["AOV"]    * np.exp(rng.normal(0, noise, len(df)))).clip(0)

    df["Quantity"] = (df["Clicks"] * df["CR"]).round().astype(int).clip(0)
    df["Sales"]    = (df["Quantity"] * df["AOV"]).round(2)

    return df[["Date", "Clicks", "Quantity", "Sales"]]


# ── Main render ────────────────────────────────────────────────────────────────

def render_brand_forge(profiles: pd.DataFrame) -> None:
    """Render the Brand Forge page."""
    _user_name = st.session_state.get("_user_name", "Unknown")
    _username  = st.session_state.get("_username", "")

    # ── Section 1: DNA Source ─────────────────────────────────────────────────
    st.markdown("#### DNA Source")
    all_brands = sorted(profiles["brand"].unique())

    src_col1, src_col2 = st.columns([1, 1])
    primary_brand = src_col1.selectbox("Primary Brand", all_brands, key="forge_primary")

    blend_on = src_col2.checkbox("Blend with a second brand", key="forge_blend_on")
    secondary_brand: str | None = None
    blend_weight = 0.0

    if blend_on:
        remaining = [b for b in all_brands if b != primary_brand]
        if remaining:
            secondary_brand = src_col2.selectbox("Secondary Brand", remaining, key="forge_secondary")
            blend_weight = st.slider(
                "Blend weight (← primary / secondary →)",
                0.0, 1.0, 0.3, 0.05, key="forge_bw",
            )
        else:
            src_col2.warning("Only one brand available — blending disabled.")

    # ── Build base DNA ────────────────────────────────────────────────────────
    primary_dna = _get_monthly_dna(profiles, primary_brand)

    if secondary_brand and blend_on:
        secondary_dna = _get_monthly_dna(profiles, secondary_brand)
        base_dna = _blend_dna(primary_dna, secondary_dna, blend_weight)
    else:
        base_dna = primary_dna.copy()

    st.divider()

    # ── Section 2: Monthly Adjustments ───────────────────────────────────────
    st.markdown("#### Monthly Index Adjustments *(multiplier — 1.0 = unchanged)*")

    with st.expander("Clicks index multipliers", expanded=False):
        cols = st.columns(12)
        mult_clicks = [
            col.number_input(name, 0.1, 5.0, 1.0, 0.05, key=f"forge_mc_{i}")
            for i, (col, name) in enumerate(zip(cols, _MONTH_NAMES))
        ]

    with st.expander("CR index multipliers", expanded=False):
        cols = st.columns(12)
        mult_cr = [
            col.number_input(name, 0.1, 5.0, 1.0, 0.05, key=f"forge_mcr_{i}")
            for i, (col, name) in enumerate(zip(cols, _MONTH_NAMES))
        ]

    with st.expander("AOV index multipliers", expanded=False):
        cols = st.columns(12)
        mult_aov = [
            col.number_input(name, 0.1, 5.0, 1.0, 0.05, key=f"forge_maov_{i}")
            for i, (col, name) in enumerate(zip(cols, _MONTH_NAMES))
        ]

    forge_dna = _apply_multipliers(base_dna, mult_clicks, mult_cr, mult_aov)

    # ── DNA Preview chart ─────────────────────────────────────────────────────
    fig_dna = go.Figure()
    traces = [
        ("idx_clicks", "#1a1a6b", "Clicks index"),
        ("idx_cr",     "#10b981", "CR index"),
        ("idx_aov",    "#f59e0b", "AOV index"),
    ]
    for col, color, name in traces:
        fig_dna.add_trace(go.Scatter(
            x=forge_dna["TimeIdx"], y=forge_dna[col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
        ))
    fig_dna.update_layout(
        title="Forged DNA Preview (monthly indices)",
        xaxis=dict(title="Month", tickvals=list(range(1, 13)),
                   ticktext=_MONTH_NAMES),
        yaxis_title="Index",
        template=_TEMPLATE, height=320, margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig_dna, use_container_width=True)

    st.divider()

    # ── Section 3: Shocks & De-Shock (optional) ───────────────────────────────
    with st.expander("Shock / De-Shock overrides *(optional)*", expanded=False):
        st.info(
            "After saving this brand you can inject shocks and de-shock it "
            "in the **Simulation Lab** exactly like any other brand."
        )

    st.divider()

    # ── Section 4: Volume targets & generation ────────────────────────────────
    st.markdown("#### Volume Targets & Data Generation")

    tgt_col1, tgt_col2, tgt_col3 = st.columns(3)
    annual_clicks  = tgt_col1.number_input("Annual Clicks",        100, 50_000_000, 100_000, 1_000, key="forge_clicks")
    annual_orders  = tgt_col2.number_input("Annual Orders",         10,  5_000_000,  10_000,   100, key="forge_orders")
    annual_revenue = tgt_col3.number_input("Annual Revenue (€)",   100, 100_000_000, 500_000, 5_000, key="forge_rev")

    d_col1, d_col2 = st.columns(2)
    data_start = d_col1.date_input("Data Start Date", date(2023, 1, 1), key="forge_dstart")
    data_end   = d_col2.date_input("Data End Date",   date(2024, 12, 31), key="forge_dend")

    noise_level = st.slider(
        "Day-to-day noise level (σ of log-normal)",
        0.0, 0.5, 0.10, 0.01, key="forge_noise",
    )

    new_brand_name = st.text_input(
        "New Brand Name", placeholder="e.g. brand_x", key="forge_name"
    )

    st.divider()

    # ── Generate & preview ────────────────────────────────────────────────────
    if st.button("Generate & Preview", key="forge_preview", use_container_width=True):
        errors = []
        if not new_brand_name.strip():
            errors.append("Enter a brand name.")
        if data_start >= data_end:
            errors.append("Start date must be before end date.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            raw_df = _generate_synthetic(
                forge_dna, data_start, data_end,
                annual_clicks, annual_orders, annual_revenue, noise_level,
            )
            st.session_state["forge_raw_df"]     = raw_df
            st.session_state["forge_brand_name"] = new_brand_name.strip().lower()

            st.success(f"Generated **{len(raw_df):,} days** of synthetic data — ready to save.")
            log_action(
                name=_user_name, username=_username,
                action="Brand Forge: Preview Generated",
                details=(
                    f"Draft name: {new_brand_name.strip().lower()} | "
                    f"Primary DNA: {primary_brand} | "
                    f"Blend: {f'{secondary_brand} @ {blend_weight:.0%}' if secondary_brand and blend_on else 'none'} | "
                    f"Period: {data_start} → {data_end} | "
                    f"Annual targets — Clicks: {annual_clicks:,} | "
                    f"Orders: {annual_orders:,} | Revenue: €{annual_revenue:,} | "
                    f"Noise σ: {noise_level:.2f} | Rows generated: {len(raw_df):,}"
                ),
            )

            # Preview chart
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Scatter(
                x=raw_df["Date"], y=raw_df["Clicks"],
                name="Clicks", line=dict(color="#1a1a6b"),
            ))
            fig_prev.add_trace(go.Scatter(
                x=raw_df["Date"], y=raw_df["Quantity"],
                name="Orders", line=dict(color="#10b981"),
                yaxis="y2",
            ))
            fig_prev.update_layout(
                title="Generated Daily Data Preview",
                template=_TEMPLATE, height=320,
                yaxis=dict(title="Clicks"),
                yaxis2=dict(title="Orders", overlaying="y", side="right"),
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_prev, use_container_width=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    forge_ready = (
        "forge_raw_df" in st.session_state
        and st.session_state.get("forge_raw_df") is not None
    )
    if forge_ready:
        saved_name = st.session_state.get("forge_brand_name", "")
        if st.button(
            f"Save brand  \"{saved_name}\"",
            key="forge_save", type="primary", use_container_width=True,
        ):
            ok, msg = save_brand(
                saved_name,
                st.session_state["forge_raw_df"],
                ["Daily", "Weekly", "Monthly"],
                PROFILES_PATH,
                DATASET_PATH,
                overwrite=False,
            )
            if ok:
                _raw = st.session_state["forge_raw_df"]
                log_action(
                    name=_user_name, username=_username,
                    action="Brand Forge: Brand Saved",
                    details=(
                        f"Brand: {saved_name} | "
                        f"Rows saved: {len(_raw):,} | "
                        f"Date range: {_raw['Date'].min().date()} → {_raw['Date'].max().date()} | "
                        f"Total clicks: {int(_raw['Clicks'].sum()):,} | "
                        f"Total orders: {int(_raw['Quantity'].sum()):,} | "
                        f"Total sales: €{_raw['Sales'].sum():,.0f}"
                    ),
                )
                st.cache_data.clear()   # force profiles reload → brand appears in DNA selector
                st.success(
                    msg + f"  **{saved_name}** now appears in the DNA Brands selector."
                )
                st.session_state.pop("forge_raw_df", None)
                st.session_state.pop("forge_brand_name", None)
            else:
                st.error(msg)
