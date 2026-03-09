"""Monitor — standalone brand dashboard with historical exploration and forecasting."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from engine.monitor_models import MODELS, run_monitor_forecast, forecast_nns_var

# ── Colour palette ───────────────────────────────────────────────────────────
_ORANGE = "#F47920"
_BLUE   = "#1a1a6b"
_GREY   = "#888888"
_MODEL_COLORS = {
    "SARIMAX":            "#1f77b4",
    "Local Linear Trend": "#ff7f0e",
    "Neural Network":     "#2ca02c",
    "NNS.VAR":            "#d62728",
}
_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_YEAR_COLORS = px.colors.qualitative.Set2


# ── Helper: aggregate raw data ──────────────────────────────────────────────

def _aggregate(brand_df, resolution, metric):
    """Aggregate brand_df to the requested resolution for a single metric."""
    d = brand_df.copy()
    d["Date"] = pd.to_datetime(d["Date"])

    if metric == "CR":
        d["_metric"] = d["quantity"] / d["clicks"].replace(0, np.nan)
    elif metric == "AOV":
        d["_metric"] = d["sales"] / d["quantity"].replace(0, np.nan)
    else:
        d["_metric"] = d[metric]

    if resolution == "Daily":
        agg = d.groupby("Date").agg(_val=("_metric", "sum" if metric not in ("CR", "AOV") else "mean")).reset_index()
    elif resolution == "Weekly":
        d["Week"] = d["Date"].dt.to_period("W").dt.start_time
        agg = d.groupby("Week").agg(_val=("_metric", "sum" if metric not in ("CR", "AOV") else "mean")).reset_index()
        agg.rename(columns={"Week": "Date"}, inplace=True)
    else:  # Monthly
        d["Month"] = d["Date"].dt.to_period("M").dt.to_timestamp()
        agg = d.groupby("Month").agg(_val=("_metric", "sum" if metric not in ("CR", "AOV") else "mean")).reset_index()
        agg.rename(columns={"Month": "Date"}, inplace=True)

    agg.rename(columns={"_val": "value"}, inplace=True)
    return agg.sort_values("Date").reset_index(drop=True)


def _prepare_monthly_single(brand_df, metric):
    """Prepare monthly aggregated data for forecasting."""
    d = brand_df.copy()
    d["Date"] = pd.to_datetime(d["Date"])

    if metric == "CR":
        d["_m"] = d["quantity"] / d["clicks"].replace(0, np.nan)
    elif metric == "AOV":
        d["_m"] = d["sales"] / d["quantity"].replace(0, np.nan)
    else:
        d["_m"] = d[metric]

    d["YearMonth"] = d["Date"].dt.to_period("M")
    if metric in ("CR", "AOV"):
        monthly = d.groupby("YearMonth")["_m"].mean().reset_index()
    else:
        monthly = d.groupby("YearMonth")["_m"].sum().reset_index()
    monthly.columns = ["YearMonth", "value"]
    monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()

    if len(monthly) > 1:
        last_ym = monthly["YearMonth"].iloc[-1]
        n_days = d[d["YearMonth"] == last_ym]["Date"].nunique()
        if n_days < 15:
            monthly = monthly.iloc[:-1]

    return monthly[["Date", "value"]].sort_values("Date").reset_index(drop=True)


def _monthly_by_year(brand_df, metric):
    """Return DataFrame with columns Year, Month, value for a metric."""
    d = brand_df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month

    if metric == "CR":
        agg = d.groupby(["Year", "Month"]).apply(
            lambda g: g["quantity"].sum() / max(g["clicks"].sum(), 1) * 100,
            include_groups=False,
        ).reset_index(name="value")
    elif metric == "AOV":
        agg = d.groupby(["Year", "Month"]).apply(
            lambda g: g["sales"].sum() / max(g["quantity"].sum(), 1),
            include_groups=False,
        ).reset_index(name="value")
    else:
        agg = d.groupby(["Year", "Month"])[metric].sum().reset_index(name="value")
    return agg.sort_values(["Year", "Month"]).reset_index(drop=True)


# ── Main render function ────────────────────────────────────────────────────

def render_monitor(profiles, df_raw, lang="en", pipeline_cache=None,
                   event_log=None, sel_brands=None):
    """Render the Monitor dashboard page."""

    all_brands = sorted(df_raw["brand"].unique())
    if not all_brands:
        st.warning("No brand data available.")
        return

    # ── 1. Brand Selector ────────────────────────────────────────────────
    selected_brand = st.selectbox("Select Brand", all_brands, key="monitor_brand")

    brand_df = df_raw[df_raw["brand"] == selected_brand].copy()
    brand_df["Date"] = pd.to_datetime(brand_df["Date"])

    if brand_df.empty:
        st.warning(f"No data available for **{selected_brand}**.")
        return

    # ── 2. Year Dashboard ──────────────────────────────────────────────
    st.markdown("### Year Dashboard")

    brand_df["Year"] = brand_df["Date"].dt.year
    available_years = sorted(brand_df["Year"].unique())
    year_options = ["All Years"] + [str(y) for y in available_years]
    selected_year = st.radio("Select Year", year_options, horizontal=True,
                             key="monitor_year_sel")

    if selected_year == "All Years":
        # Grand totals + yearly summary table
        total_clicks = brand_df["clicks"].sum()
        total_qty = brand_df["quantity"].sum()
        total_sales = brand_df["sales"].sum()
        avg_cr = (total_qty / total_clicks * 100) if total_clicks > 0 else 0
        avg_aov = (total_sales / total_qty) if total_qty > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Clicks", f"{total_clicks:,.0f}")
        c2.metric("Total Quantity", f"{total_qty:,.0f}")
        c3.metric("Total Sales", f"{total_sales:,.2f}")
        c4.metric("Avg CR", f"{avg_cr:.2f}%")
        c5.metric("Avg AOV", f"{avg_aov:,.2f}")

        # Yearly summary table
        yearly = brand_df.groupby("Year").agg(
            Clicks=("clicks", "sum"),
            Quantity=("quantity", "sum"),
            Sales=("sales", "sum"),
        ).reset_index()
        yearly["CR (%)"] = (yearly["Quantity"] / yearly["Clicks"].replace(0, np.nan) * 100).round(2)
        yearly["AOV"] = (yearly["Sales"] / yearly["Quantity"].replace(0, np.nan)).round(2)

        # YoY growth columns
        yearly["Clicks Growth (%)"] = yearly["Clicks"].pct_change().apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
        yearly["Sales Growth (%)"] = yearly["Sales"].pct_change().apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "—")

        yearly_fmt = yearly.copy()
        yearly_fmt["Clicks"] = yearly_fmt["Clicks"].apply(lambda x: f"{x:,.0f}")
        yearly_fmt["Quantity"] = yearly_fmt["Quantity"].apply(lambda x: f"{x:,.0f}")
        yearly_fmt["Sales"] = yearly_fmt["Sales"].apply(lambda x: f"{x:,.2f}")
        st.dataframe(yearly_fmt, use_container_width=True, hide_index=True)

    else:
        # Single year selected
        yr_int = int(selected_year)
        yr_df = brand_df[brand_df["Year"] == yr_int]
        prev_yr_df = brand_df[brand_df["Year"] == yr_int - 1]

        yr_clicks = yr_df["clicks"].sum()
        yr_qty = yr_df["quantity"].sum()
        yr_sales = yr_df["sales"].sum()
        yr_cr = (yr_qty / yr_clicks * 100) if yr_clicks > 0 else 0
        yr_aov = (yr_sales / yr_qty) if yr_qty > 0 else 0

        # YoY deltas
        prev_clicks = prev_yr_df["clicks"].sum() if not prev_yr_df.empty else None
        prev_qty = prev_yr_df["quantity"].sum() if not prev_yr_df.empty else None
        prev_sales = prev_yr_df["sales"].sum() if not prev_yr_df.empty else None
        prev_cr = (prev_qty / prev_clicks * 100) if prev_clicks and prev_clicks > 0 else None
        prev_aov = (prev_sales / prev_qty) if prev_qty and prev_qty > 0 else None

        def _delta_str(cur, prev):
            if prev is None or prev == 0:
                return None
            pct = (cur - prev) / prev * 100
            return f"{pct:+.1f}% vs {yr_int - 1}"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Clicks", f"{yr_clicks:,.0f}", delta=_delta_str(yr_clicks, prev_clicks))
        c2.metric("Quantity", f"{yr_qty:,.0f}", delta=_delta_str(yr_qty, prev_qty))
        c3.metric("Sales", f"{yr_sales:,.2f}", delta=_delta_str(yr_sales, prev_sales))
        c4.metric("CR", f"{yr_cr:.2f}%", delta=_delta_str(yr_cr, prev_cr))
        c5.metric("AOV", f"{yr_aov:,.2f}", delta=_delta_str(yr_aov, prev_aov))

        # Monthly breakdown bar chart for selected year
        yr_bar_metric = st.selectbox("Monthly Breakdown Metric",
                                      ["clicks", "quantity", "sales", "CR", "AOV"],
                                      key="monitor_yr_bar_met")
        yr_monthly = _monthly_by_year(yr_df, yr_bar_metric)

        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(
            x=[_MONTH_LABELS[m - 1] for m in yr_monthly["Month"]],
            y=yr_monthly["value"],
            marker_color=_ORANGE,
            text=[f"{v:,.0f}" if yr_bar_metric not in ("CR", "AOV") else f"{v:.2f}"
                  for v in yr_monthly["value"]],
            textposition="outside",
        ))
        fig_yr.update_layout(
            title=f"{yr_bar_metric.upper()} — Monthly Breakdown ({selected_year})",
            yaxis_title=yr_bar_metric.upper(), height=380,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_yr, use_container_width=True)

        # Monthly table for selected year: all metrics
        yr_df_copy = yr_df.copy()
        yr_df_copy["Month"] = yr_df_copy["Date"].dt.month
        yr_tbl = yr_df_copy.groupby("Month").agg(
            Clicks=("clicks", "sum"),
            Quantity=("quantity", "sum"),
            Sales=("sales", "sum"),
        ).reset_index()
        yr_tbl["CR (%)"] = (yr_tbl["Quantity"] / yr_tbl["Clicks"].replace(0, np.nan) * 100).round(2)
        yr_tbl["AOV"] = (yr_tbl["Sales"] / yr_tbl["Quantity"].replace(0, np.nan)).round(2)
        yr_tbl["Month"] = yr_tbl["Month"].apply(lambda m: _MONTH_LABELS[m - 1])
        yr_tbl["Clicks"] = yr_tbl["Clicks"].apply(lambda x: f"{x:,.0f}")
        yr_tbl["Quantity"] = yr_tbl["Quantity"].apply(lambda x: f"{x:,.0f}")
        yr_tbl["Sales"] = yr_tbl["Sales"].apply(lambda x: f"{x:,.2f}")
        st.dataframe(yr_tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 3. Cross-Year Comparison ──────────────────────────────────────────
    st.markdown("### Cross-Year Comparison")

    cyr_metric = st.selectbox(
        "Comparison Metric", ["clicks", "quantity", "sales", "CR", "AOV"],
        key="monitor_cyr_metric",
    )

    cyr_data = _monthly_by_year(brand_df, cyr_metric)
    cyr_years = sorted(cyr_data["Year"].unique())

    tab_overlay, tab_heatmap, tab_growth = st.tabs(
        ["Monthly Overlay", "Heatmap", "Growth Rates"])

    with tab_overlay:
        fig_ov = go.Figure()
        for i, year in enumerate(cyr_years):
            yd = cyr_data[cyr_data["Year"] == year].sort_values("Month")
            color = _YEAR_COLORS[i % len(_YEAR_COLORS)]
            fig_ov.add_trace(go.Scatter(
                x=yd["Month"].values,
                y=yd["value"].values, mode="lines+markers", name=str(year),
                line=dict(color=color, width=2),
            ))
        fig_ov.update_layout(
            title=f"{cyr_metric.upper()} — Monthly Pattern per Year",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=_MONTH_LABELS,
            ),
            yaxis_title=cyr_metric.upper(), height=420,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_ov, use_container_width=True)

    with tab_heatmap:
        hm_pivot = cyr_data.pivot(index="Year", columns="Month", values="value").fillna(0)
        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_pivot.values,
            x=[_MONTH_LABELS[m - 1] for m in hm_pivot.columns],
            y=[str(y) for y in hm_pivot.index],
            colorscale="YlOrRd",
            hoverongaps=False,
        ))
        fig_hm.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"{cyr_metric.upper()} by Month and Year",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_growth:
        # YoY growth rate heatmap
        growth_pivot = cyr_data.pivot(index="Year", columns="Month", values="value").fillna(0)
        if len(growth_pivot) >= 2:
            growth_pct = growth_pivot.pct_change() * 100
            growth_pct = growth_pct.iloc[1:]  # drop first year (no prior)

            # Label rows as transitions
            growth_labels = [f"{cyr_years[i]}→{cyr_years[i+1]}"
                             for i in range(len(cyr_years) - 1)]

            fig_gr = go.Figure(data=go.Heatmap(
                z=growth_pct.values,
                x=[_MONTH_LABELS[m - 1] for m in growth_pct.columns],
                y=growth_labels,
                colorscale=[[0, "#d32f2f"], [0.5, "#ffffff"], [1, "#2e7d32"]],
                zmid=0,
                hoverongaps=False,
                text=[[f"{v:+.1f}%" if np.isfinite(v) else "—"
                       for v in row] for row in growth_pct.values],
                texttemplate="%{text}",
            ))
            fig_gr.update_layout(
                title=f"{cyr_metric.upper()} — Year-over-Year Growth by Month",
                height=max(200, len(growth_labels) * 60 + 80),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_gr, use_container_width=True)
        else:
            st.info("Need at least 2 years of data for growth analysis.")

    st.markdown("---")

    # ── 4. Time Series Explorer ──────────────────────────────────────────
    st.markdown("### Time Series Explorer")

    ts_col1, ts_col2 = st.columns(2)
    ts_resolution = ts_col1.radio(
        "Resolution", ["Daily", "Weekly", "Monthly"], key="monitor_ts_res", horizontal=True)
    ts_metric = ts_col2.selectbox(
        "Metric", ["clicks", "quantity", "sales", "CR", "AOV"], key="monitor_ts_metric")

    ts_data = _aggregate(brand_df, ts_resolution, ts_metric)
    ts_data["Year"] = ts_data["Date"].dt.year

    overlay = st.checkbox("Overlay years for comparison", key="monitor_ts_overlay")

    if overlay:
        fig_ts = go.Figure()
        for year in sorted(ts_data["Year"].unique()):
            yr_data = ts_data[ts_data["Year"] == year].copy()
            yr_data["DOY"] = yr_data["Date"].dt.dayofyear
            fig_ts.add_trace(go.Scatter(
                x=yr_data["DOY"], y=yr_data["value"],
                mode="lines", name=str(year),
            ))
        fig_ts.update_layout(
            xaxis_title="Day of Year",
            yaxis_title=ts_metric.upper(),
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
        )
    else:
        fig_ts = px.line(
            ts_data, x="Date", y="value",
            title=f"{ts_metric.upper()} — {ts_resolution}",
            color_discrete_sequence=[_ORANGE],
        )
        fig_ts.update_layout(
            yaxis_title=ts_metric.upper(),
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
        )

    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # ── 5. Forecast Section ──────────────────────────────────────────────
    st.markdown("### Forecast")

    fc_col1, fc_col2, fc_col3 = st.columns(3)
    fc_metric = fc_col1.selectbox(
        "Forecast Metric", ["clicks", "quantity", "sales", "CR", "AOV"],
        key="monitor_fc_metric",
    )
    fc_horizon = fc_col2.number_input(
        "Months Ahead", min_value=1, max_value=36, value=12, key="monitor_fc_horizon")
    fc_models = fc_col3.multiselect(
        "Models", list(MODELS.keys()), default=list(MODELS.keys()),
        key="monitor_fc_models",
    )

    if st.button("Run Forecast", key="monitor_run_fc", type="primary"):
        monthly = _prepare_monthly_single(brand_df, fc_metric)

        if len(monthly) < 6:
            st.warning("Not enough monthly data for forecasting (need at least 6 months).")
        else:
            with st.spinner("Running models..."):
                result = run_monitor_forecast(monthly, fc_horizon, fc_models)

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=monthly["Date"], y=monthly["value"],
                mode="lines", name="Historical",
                line=dict(color=_BLUE, width=2),
            ))

            all_fc_vals = []
            for model_name, fc_vals in result["forecasts"].items():
                color = _MODEL_COLORS.get(model_name, _GREY)
                fig_fc.add_trace(go.Scatter(
                    x=result["future_dates"], y=fc_vals,
                    mode="lines", name=model_name,
                    line=dict(color=color, width=2, dash="dash"),
                ))
                all_fc_vals.append(fc_vals)

            if len(all_fc_vals) > 1:
                fc_matrix = np.array(all_fc_vals)
                p10 = np.percentile(fc_matrix, 10, axis=0)
                p90 = np.percentile(fc_matrix, 90, axis=0)
                fig_fc.add_trace(go.Scatter(
                    x=list(result["future_dates"]) + list(result["future_dates"][::-1]),
                    y=list(p90) + list(p10[::-1]),
                    fill="toself",
                    fillcolor="rgba(26,26,107,0.08)",
                    line=dict(width=0),
                    name="Ensemble Band (P10-P90)",
                    showlegend=True,
                ))

            fig_fc.update_layout(
                title=f"{fc_metric.upper()} — Forecast ({fc_horizon} months)",
                yaxis_title=fc_metric.upper(),
                height=450,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            st.markdown("#### Model Performance (MAPE — Leave-Last-3-Out)")
            score_rows = []
            for model_name in fc_models:
                mape = result["scores"].get(model_name)
                score_rows.append({
                    "Model": model_name,
                    "MAPE": f"{mape:.2%}" if mape is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 6. Forecast vs Historical Comparison ─────────────────────────────
    st.markdown("### Forecast vs Historical Years")
    st.caption("Compare forecasted monthly values against each historical year.")

    cmp_metric = st.selectbox(
        "Comparison Metric", ["clicks", "quantity", "sales", "CR", "AOV"],
        key="monitor_cmp_metric",
    )

    hist_by_year = _monthly_by_year(brand_df, cmp_metric)
    years_available = sorted(hist_by_year["Year"].unique())

    # Quick NNS.VAR forecast for 12 months
    monthly_for_fc = _prepare_monthly_single(brand_df, cmp_metric)
    fc_vals_12 = None
    if len(monthly_for_fc) >= 6:
        try:
            fc_vals_12 = forecast_nns_var(monthly_for_fc, 12)
        except Exception:
            fc_vals_12 = None

    # Also check if pipeline forecast is available
    pipeline_fc_monthly = None
    if pipeline_cache is not None:
        df_proj = pipeline_cache.get("df")
        if df_proj is not None:
            _col_map = {"clicks": "Clicks_Sim", "quantity": "Qty_Sim", "sales": "Sales_Sim"}
            sim_col = _col_map.get(cmp_metric)
            if sim_col and sim_col in df_proj.columns:
                pf = df_proj[["Date", sim_col]].copy()
                pf["Month"] = pf["Date"].dt.month
                pipeline_fc_monthly = pf.groupby("Month")[sim_col].sum().reset_index()
                pipeline_fc_monthly.columns = ["Month", "value"]

    # Grouped bar chart: each year + forecast
    fig_cmp = go.Figure()
    for i, year in enumerate(years_available):
        yr_data = hist_by_year[hist_by_year["Year"] == year]
        color = _YEAR_COLORS[i % len(_YEAR_COLORS)]
        fig_cmp.add_trace(go.Bar(
            x=[_MONTH_LABELS[m - 1] for m in yr_data["Month"]],
            y=yr_data["value"],
            name=str(year),
            marker_color=color,
        ))

    if fc_vals_12 is not None:
        fig_cmp.add_trace(go.Bar(
            x=_MONTH_LABELS,
            y=fc_vals_12,
            name="Forecast (NNS.VAR)",
            marker_color=_ORANGE,
            marker_pattern_shape="/",
        ))

    if pipeline_fc_monthly is not None and cmp_metric in ("clicks", "quantity", "sales"):
        fig_cmp.add_trace(go.Bar(
            x=[_MONTH_LABELS[m - 1] for m in pipeline_fc_monthly["Month"]],
            y=pipeline_fc_monthly["value"],
            name="Pipeline Forecast",
            marker_color=_BLUE,
            marker_pattern_shape="x",
        ))

    fig_cmp.update_layout(
        barmode="group",
        title=f"{cmp_metric.upper()} — Historical Years vs Forecast",
        yaxis_title=cmp_metric.upper(),
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Comparison table
    pivot = hist_by_year.pivot(index="Month", columns="Year", values="value").fillna(0)
    pivot.index = [_MONTH_LABELS[m - 1] for m in pivot.index]
    median_col = pivot.median(axis=1)
    pivot["Median"] = median_col.round(2)
    if fc_vals_12 is not None:
        pivot["Forecast"] = fc_vals_12[:len(pivot)]
        pivot["Dev from Median (%)"] = (
            ((pivot["Forecast"] - pivot["Median"]) / pivot["Median"].replace(0, np.nan)) * 100
        ).round(1)
    st.dataframe(pivot, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # 7. PERIOD DEEP DIVE — "WHY THIS FORECAST?"
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Period Deep Dive — Why This Forecast?")
    st.caption("Select a month and metric to understand how the forecast is built "
               "and how it relates to each historical year.")

    dd_c1, dd_c2 = st.columns(2)
    dd_month_idx = dd_c1.selectbox(
        "Month", list(range(1, 13)),
        format_func=lambda m: _MONTH_LABELS[m - 1],
        key="monitor_dd_month",
    )
    dd_metric = dd_c2.selectbox(
        "Metric", ["clicks", "quantity", "sales", "CR", "AOV"],
        key="monitor_dd_metric",
    )

    # ── Panel A: Historical Context ──────────────────────────────────
    st.markdown("#### Historical Context")

    dd_hist = _monthly_by_year(brand_df, dd_metric)
    dd_month_data = dd_hist[dd_hist["Month"] == dd_month_idx].sort_values("Year")

    # Get forecast for this month (NNS.VAR quick)
    dd_fc_val = None
    dd_monthly_fc = _prepare_monthly_single(brand_df, dd_metric)
    if len(dd_monthly_fc) >= 6:
        try:
            _dd_fc_12 = forecast_nns_var(dd_monthly_fc, 12)
            dd_fc_val = _dd_fc_12[dd_month_idx - 1] if dd_month_idx <= len(_dd_fc_12) else None
        except Exception:
            pass

    # Pipeline forecast for this month
    dd_pipe_fc_val = None
    if pipeline_cache is not None:
        _dd_proj = pipeline_cache.get("df")
        if _dd_proj is not None:
            _dd_col_map = {"clicks": "Clicks_Sim", "quantity": "Qty_Sim", "sales": "Sales_Sim"}
            _dd_sim_col = _dd_col_map.get(dd_metric)
            if _dd_sim_col and _dd_sim_col in _dd_proj.columns:
                _dd_pf = _dd_proj[_dd_proj["Month"] == dd_month_idx]
                dd_pipe_fc_val = _dd_pf[_dd_sim_col].sum()

    # Horizontal bar chart: each year + forecast
    fig_dd_hist = go.Figure()
    bar_labels = [str(y) for y in dd_month_data["Year"]]
    bar_values = list(dd_month_data["value"])
    bar_colors = [_YEAR_COLORS[i % len(_YEAR_COLORS)] for i in range(len(bar_labels))]

    if dd_pipe_fc_val is not None:
        bar_labels.append("Forecast (Pipeline)")
        bar_values.append(dd_pipe_fc_val)
        bar_colors.append(_ORANGE)
    elif dd_fc_val is not None:
        bar_labels.append("Forecast (NNS.VAR)")
        bar_values.append(dd_fc_val)
        bar_colors.append(_ORANGE)

    fig_dd_hist.add_trace(go.Bar(
        y=bar_labels, x=bar_values,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:,.0f}" if dd_metric not in ("CR", "AOV") else f"{v:.2f}"
              for v in bar_values],
        textposition="outside",
    ))
    fig_dd_hist.update_layout(
        title=f"{dd_metric.upper()} for {_MONTH_LABELS[dd_month_idx - 1]} — All Years vs Forecast",
        xaxis_title=dd_metric.upper(),
        height=max(250, len(bar_labels) * 45 + 80),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_dd_hist, use_container_width=True)

    # Summary table: all metrics for the selected month across years
    dd_all_metrics = {}
    for m_name in ["clicks", "quantity", "sales", "CR", "AOV"]:
        m_data = _monthly_by_year(brand_df, m_name)
        m_month = m_data[m_data["Month"] == dd_month_idx].set_index("Year")["value"]
        dd_all_metrics[m_name.upper()] = m_month

    dd_tbl = pd.DataFrame(dd_all_metrics)
    dd_tbl.index.name = "Year"

    # Add similarity weights if available
    if pipeline_cache is not None:
        _dd_weights = pipeline_cache.get("norm_weights") or {}
        if _dd_weights:
            dd_tbl["Weight"] = dd_tbl.index.map(
                lambda y: f"{_dd_weights.get(y, 0):.1%}" if y in _dd_weights else "—")

    st.dataframe(dd_tbl, use_container_width=True)

    # ── Panel B: Forecast Composition (requires pipeline_cache) ──────
    if pipeline_cache is not None:
        _dd_proj = pipeline_cache.get("df")
        _dd_pure_dna = pipeline_cache.get("pure_dna")
        _dd_base_clicks = pipeline_cache.get("base_clicks")
        _dd_base_cr = pipeline_cache.get("base_cr")
        _dd_base_aov = pipeline_cache.get("base_aov")

        if _dd_proj is not None and dd_metric in ("clicks", "quantity", "sales"):
            st.markdown("#### Forecast Composition")

            _dd_met_map = {"clicks": "Clicks", "quantity": "Qty", "sales": "Sales"}
            _dd_pmet = _dd_met_map[dd_metric]
            _dd_base_col = f"{_dd_pmet}_Base"
            _dd_sim_col = f"{_dd_pmet}_Sim"

            _dd_month_mask = _dd_proj["Month"] == dd_month_idx
            _dd_month_rows = _dd_proj[_dd_month_mask]

            base_proj = _dd_month_rows[_dd_base_col].sum() if _dd_base_col in _dd_proj.columns else 0
            final_fc = _dd_month_rows[_dd_sim_col].sum() if _dd_sim_col in _dd_proj.columns else 0
            campaign_lift = final_fc - base_proj

            # DNA index for this month
            dna_idx_val = None
            if _dd_pure_dna is not None and not _dd_pure_dna.empty:
                _dd_idx_col = {"clicks": "idx_clicks", "quantity": "idx_cr", "sales": "idx_aov"}
                _dd_dna_col = _dd_idx_col.get(dd_metric, "idx_clicks")
                _dd_dna_row = _dd_pure_dna[_dd_pure_dna["TimeIdx"] == dd_month_idx]
                if not _dd_dna_row.empty and _dd_dna_col in _dd_dna_row.columns:
                    dna_idx_val = float(_dd_dna_row[_dd_dna_col].iloc[0])

            # Base value
            _dd_base_map = {"clicks": _dd_base_clicks, "quantity": None, "sales": None}
            base_val = _dd_base_map.get(dd_metric)

            # Historical median for this month
            hist_med_val = None
            profiles_use = pipeline_cache.get("profiles_mod") if pipeline_cache.get("profiles_mod") is not None else profiles
            if sel_brands:
                _dd_mp = profiles_use[
                    (profiles_use["brand"].isin(sel_brands))
                    & (profiles_use["Level"] == "Monthly")
                    & (profiles_use["Year"] != "Overall")
                ]
                if not _dd_mp.empty:
                    _dd_prof_col = {"clicks": "clicks", "quantity": "quantity", "sales": "sales"}
                    _dd_pc = _dd_prof_col.get(dd_metric, "clicks")
                    _dd_yr_agg = _dd_mp.groupby(["Year", "TimeIdx"])[_dd_pc].sum().reset_index()
                    _dd_med = _dd_yr_agg[_dd_yr_agg["TimeIdx"] == dd_month_idx][_dd_pc].median()
                    hist_med_val = float(_dd_med) if pd.notna(_dd_med) else None

            # Metric cards
            mc1, mc2, mc3, mc4 = st.columns(4)
            if dna_idx_val is not None:
                mc1.metric("DNA Index", f"{dna_idx_val:.3f}")
            else:
                mc1.metric("DNA Index", "N/A")
            mc2.metric("Base Projection", f"{base_proj:,.0f}")
            mc3.metric("Campaign Effect", f"{campaign_lift:+,.0f}")
            mc4.metric("Final Forecast", f"{final_fc:,.0f}")

            # Waterfall chart
            wf_labels = ["Base Projection", "Campaign / Events", "Final Forecast"]
            wf_measures = ["absolute", "relative", "total"]
            wf_values = [base_proj, campaign_lift, final_fc]

            fig_wf = go.Figure(go.Waterfall(
                name="", orientation="v",
                x=wf_labels, y=wf_values,
                measure=wf_measures,
                connector=dict(line=dict(color=_GREY, width=1)),
                increasing=dict(marker_color="#2e7d32"),
                decreasing=dict(marker_color="#d32f2f"),
                totals=dict(marker_color=_ORANGE),
                text=[f"{v:,.0f}" for v in wf_values],
                textposition="outside",
            ))

            if hist_med_val is not None:
                fig_wf.add_hline(y=hist_med_val, line_dash="dot", line_color=_BLUE,
                                 annotation_text=f"Historical Median: {hist_med_val:,.0f}",
                                 annotation_position="top right")

            fig_wf.update_layout(
                title=f"{dd_metric.upper()} Forecast Breakdown — {_MONTH_LABELS[dd_month_idx - 1]}",
                yaxis_title=dd_metric.upper(), height=400,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    # ── Panel C: Year Dynamics ───────────────────────────────────────
    st.markdown("#### Year Dynamics")

    # Trend line: value per year for this month + forecast as last point
    dd_trend_years = list(dd_month_data["Year"])
    dd_trend_vals = list(dd_month_data["value"])

    fc_year_label = None
    fc_val_for_trend = dd_pipe_fc_val if dd_pipe_fc_val is not None else dd_fc_val
    if fc_val_for_trend is not None:
        fc_year_label = max(dd_trend_years) + 1 if dd_trend_years else 2025
        dd_trend_years.append(fc_year_label)
        dd_trend_vals.append(fc_val_for_trend)

    fig_trend = go.Figure()
    # Historical points
    n_hist = len(dd_month_data)
    fig_trend.add_trace(go.Scatter(
        x=dd_trend_years[:n_hist], y=dd_trend_vals[:n_hist],
        mode="lines+markers", name="Historical",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=8),
    ))
    # Forecast point
    if fc_year_label is not None:
        fig_trend.add_trace(go.Scatter(
            x=[dd_trend_years[-2], dd_trend_years[-1]],
            y=[dd_trend_vals[-2], dd_trend_vals[-1]],
            mode="lines+markers", name="Forecast",
            line=dict(color=_ORANGE, width=2, dash="dash"),
            marker=dict(size=10, symbol="star"),
        ))

    fig_trend.update_layout(
        title=f"{dd_metric.upper()} for {_MONTH_LABELS[dd_month_idx - 1]} — Trend Across Years",
        xaxis_title="Year", yaxis_title=dd_metric.upper(),
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # YoY growth table
    if len(dd_month_data) >= 2:
        growth_rows = []
        vals = dd_month_data["value"].values
        yrs = dd_month_data["Year"].values
        median_val = np.median(vals) if len(vals) > 0 else 0
        for i, (yr, val) in enumerate(zip(yrs, vals)):
            prev_chg = f"{(val - vals[i-1]) / max(abs(vals[i-1]), 1) * 100:+.1f}%" if i > 0 else "—"
            med_chg = f"{(val - median_val) / max(abs(median_val), 1) * 100:+.1f}%" if median_val else "—"
            growth_rows.append({
                "Year": yr,
                "Value": f"{val:,.0f}" if dd_metric not in ("CR", "AOV") else f"{val:.2f}",
                "vs Previous Year": prev_chg,
                "vs Median": med_chg,
            })
        if fc_val_for_trend is not None:
            med_chg = f"{(fc_val_for_trend - median_val) / max(abs(median_val), 1) * 100:+.1f}%"
            prev_chg = f"{(fc_val_for_trend - vals[-1]) / max(abs(vals[-1]), 1) * 100:+.1f}%"
            growth_rows.append({
                "Year": "Forecast",
                "Value": f"{fc_val_for_trend:,.0f}" if dd_metric not in ("CR", "AOV") else f"{fc_val_for_trend:.2f}",
                "vs Previous Year": prev_chg,
                "vs Median": med_chg,
            })
        st.dataframe(pd.DataFrame(growth_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # 8. FORECAST DECOMPOSITION MATRIX
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Forecast Decomposition Matrix")
    st.caption("All 12 months at a glance — see how the forecast is composed for each month.")

    if pipeline_cache is not None:
        _fdm_proj = pipeline_cache.get("df")
        if _fdm_proj is not None:
            fdm_metric = st.selectbox(
                "Decomposition Metric", ["Clicks", "Qty", "Sales"],
                key="monitor_fdm_metric",
            )
            _fdm_base_col = f"{fdm_metric}_Base"
            _fdm_sim_col = f"{fdm_metric}_Sim"

            if _fdm_base_col in _fdm_proj.columns and _fdm_sim_col in _fdm_proj.columns:
                # Aggregate by month
                fdm_monthly = _fdm_proj.groupby("Month").agg(
                    Base=(_fdm_base_col, "sum"),
                    Forecast=(_fdm_sim_col, "sum"),
                ).reset_index()
                fdm_monthly["Campaign / Events"] = fdm_monthly["Forecast"] - fdm_monthly["Base"]

                # DNA index per month
                _fdm_pure_dna = pipeline_cache.get("pure_dna")
                if _fdm_pure_dna is not None and not _fdm_pure_dna.empty:
                    _fdm_idx_map = {"Clicks": "idx_clicks", "Qty": "idx_cr", "Sales": "idx_aov"}
                    _fdm_idx_col = _fdm_idx_map.get(fdm_metric, "idx_clicks")
                    if _fdm_idx_col in _fdm_pure_dna.columns:
                        dna_map = dict(zip(_fdm_pure_dna["TimeIdx"].astype(int),
                                           _fdm_pure_dna[_fdm_idx_col]))
                        fdm_monthly["DNA Index"] = fdm_monthly["Month"].map(dna_map).round(3)
                    else:
                        fdm_monthly["DNA Index"] = "—"
                else:
                    fdm_monthly["DNA Index"] = "—"

                # Historical median
                profiles_use = pipeline_cache.get("profiles_mod") if pipeline_cache.get("profiles_mod") is not None else profiles
                _fdm_prof_map = {"Clicks": "clicks", "Qty": "quantity", "Sales": "sales"}
                _fdm_pc = _fdm_prof_map.get(fdm_metric, "clicks")
                hist_med_series = pd.Series(dtype=float)
                if sel_brands:
                    _fdm_mp = profiles_use[
                        (profiles_use["brand"].isin(sel_brands))
                        & (profiles_use["Level"] == "Monthly")
                        & (profiles_use["Year"] != "Overall")
                    ]
                    if not _fdm_mp.empty:
                        _fdm_ya = _fdm_mp.groupby(["Year", "TimeIdx"])[_fdm_pc].sum().reset_index()
                        hist_med_series = _fdm_ya.groupby("TimeIdx")[_fdm_pc].median()

                fdm_monthly["Hist Median"] = fdm_monthly["Month"].map(hist_med_series).fillna(0)
                fdm_monthly["Deviation (%)"] = (
                    (fdm_monthly["Forecast"] - fdm_monthly["Hist Median"])
                    / fdm_monthly["Hist Median"].replace(0, np.nan) * 100
                ).round(1)

                # Stacked bar chart
                fig_fdm = go.Figure()
                fig_fdm.add_trace(go.Bar(
                    x=[_MONTH_LABELS[m - 1] for m in fdm_monthly["Month"]],
                    y=fdm_monthly["Base"],
                    name="Base Projection",
                    marker_color=_BLUE,
                ))
                fig_fdm.add_trace(go.Bar(
                    x=[_MONTH_LABELS[m - 1] for m in fdm_monthly["Month"]],
                    y=fdm_monthly["Campaign / Events"],
                    name="Campaign / Events",
                    marker_color=_ORANGE,
                ))
                # Historical median overlay line
                fig_fdm.add_trace(go.Scatter(
                    x=[_MONTH_LABELS[m - 1] for m in fdm_monthly["Month"]],
                    y=fdm_monthly["Hist Median"],
                    mode="lines+markers", name="Historical Median",
                    line=dict(color=_GREY, width=2, dash="dot"),
                ))
                fig_fdm.update_layout(
                    barmode="stack",
                    title=f"{fdm_metric} — Forecast Composition by Month",
                    yaxis_title=fdm_metric, height=420,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                )
                st.plotly_chart(fig_fdm, use_container_width=True)

                # Formatted table
                fdm_display = fdm_monthly.copy()
                fdm_display["Month"] = fdm_display["Month"].apply(lambda m: _MONTH_LABELS[m - 1])
                for col in ["Base", "Campaign / Events", "Forecast", "Hist Median"]:
                    fdm_display[col] = fdm_display[col].apply(lambda x: f"{x:,.0f}")

                # Color-code deviation
                def _dev_color(val):
                    if pd.isna(val):
                        return "—"
                    if abs(val) < 20:
                        return f"{val:+.1f}%"
                    elif abs(val) < 50:
                        return f"{val:+.1f}%"
                    return f"{val:+.1f}%"

                fdm_display["Deviation (%)"] = fdm_monthly["Deviation (%)"].apply(_dev_color)
                display_cols = ["Month", "DNA Index", "Base", "Campaign / Events",
                                "Forecast", "Hist Median", "Deviation (%)"]
                st.dataframe(fdm_display[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("Projection columns not found for the selected metric.")
        else:
            st.info("Complete Steps 1-3 in the Strategy Workflow to see the decomposition matrix.")
    else:
        st.info("Complete Steps 1-3 in the Strategy Workflow to see the forecast decomposition.")

    st.markdown("---")

    # ── 9. Distribution Analysis ─────────────────────────────────────────
    st.markdown("### Monthly Distribution")

    dist_metric = st.selectbox(
        "Distribution Metric", ["clicks", "quantity", "sales", "CR", "AOV"],
        key="monitor_dist_metric",
    )

    dist_df = brand_df.copy()
    dist_df["Month"] = dist_df["Date"].dt.month
    dist_df["MonthName"] = dist_df["Date"].dt.strftime("%b")

    if dist_metric == "CR":
        month_agg = dist_df.groupby(["Year", "Month", "MonthName"]).apply(
            lambda g: g["quantity"].sum() / max(g["clicks"].sum(), 1) * 100, include_groups=False
        ).reset_index(name="value")
    elif dist_metric == "AOV":
        month_agg = dist_df.groupby(["Year", "Month", "MonthName"]).apply(
            lambda g: g["sales"].sum() / max(g["quantity"].sum(), 1), include_groups=False
        ).reset_index(name="value")
    else:
        month_agg = dist_df.groupby(["Year", "Month", "MonthName"])[dist_metric].sum().reset_index(name="value")

    month_agg = month_agg.sort_values("Month")

    fig_box = px.box(
        month_agg, x="MonthName", y="value",
        title=f"{dist_metric.upper()} Distribution by Month (across years)",
        color_discrete_sequence=[_ORANGE],
        category_orders={"MonthName": _MONTH_LABELS},
    )
    fig_box.update_layout(
        yaxis_title=dist_metric.upper(),
        xaxis_title="Month",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # 10. STEPWISE PIPELINE WALKTHROUGH
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Pipeline Walkthrough")

    if pipeline_cache is None:
        st.info("Complete Steps 1-3 in the Strategy Workflow to unlock the pipeline walkthrough. "
                "This section shows step-by-step how the forecast is built from raw data.")
        return

    if event_log is None:
        event_log = []
    if sel_brands is None:
        sel_brands = []

    # Step definitions
    _STEPS = [
        ("Raw Historical Data",      "Starting point: raw monthly data for the selected brand across all years."),
        ("Month Swaps",              "How pre-trial month swaps redistribute historical data between months."),
        ("Similarity Weights",       "How much weight each historical year gets in the DNA blend."),
        ("Pure DNA Profile",         "The blended 12-month seasonality index (35% overall + 65% weighted history)."),
        ("DNA Layers",               "How events modify the DNA: pure -> pre-trial -> working layers."),
        ("Base Calibration",         "Deriving base scalars from trial metrics and trial-window DNA."),
        ("Projections Built",        "The full forecast: Baseline (Before) and Simulation (After)."),
        ("Historical Shrinkage",     "Penalizing forecast values that deviate too far from historical medians."),
        ("Final Forecast + Bands",   "The final forecast with Monte Carlo noise bands."),
    ]

    if "monitor_pipeline_step" not in st.session_state:
        st.session_state.monitor_pipeline_step = 0

    step_idx = st.session_state.monitor_pipeline_step
    n_steps = len(_STEPS)
    step_idx = max(0, min(step_idx, n_steps - 1))

    # Navigation
    nav_c1, nav_c2, nav_c3 = st.columns([1, 4, 1])
    if nav_c1.button("Previous", key="pw_prev", disabled=(step_idx == 0), use_container_width=True):
        st.session_state.monitor_pipeline_step = step_idx - 1
        st.rerun()
    nav_c2.markdown(
        f"<div style='text-align:center;font-family:Inter,sans-serif;font-weight:600;"
        f"font-size:1.05rem;padding:6px 0'>Step {step_idx + 1}/{n_steps}: "
        f"{_STEPS[step_idx][0]}</div>",
        unsafe_allow_html=True,
    )
    if nav_c3.button("Next", key="pw_next", disabled=(step_idx == n_steps - 1), use_container_width=True):
        st.session_state.monitor_pipeline_step = step_idx + 1
        st.rerun()

    st.caption(_STEPS[step_idx][1])

    df_proj = pipeline_cache.get("df")
    pure_dna = pipeline_cache.get("pure_dna")
    norm_weights = pipeline_cache.get("norm_weights") or {}
    base_clicks = pipeline_cache.get("base_clicks")
    base_cr = pipeline_cache.get("base_cr")
    base_aov = pipeline_cache.get("base_aov")
    adj_c = pipeline_cache.get("adj_c")
    adj_q = pipeline_cache.get("adj_q")
    adj_s = pipeline_cache.get("adj_s")
    df_raw_mod = pipeline_cache.get("df_raw_mod")

    # ─── STEP 0: Raw Historical Data ─────────────────────────────────────
    if step_idx == 0:
        pw_met = st.selectbox("Metric", ["clicks", "quantity", "sales"],
                              key="pw_s0_met")
        by_yr = _monthly_by_year(brand_df, pw_met)
        fig = go.Figure()
        for year in sorted(by_yr["Year"].unique()):
            yd = by_yr[by_yr["Year"] == year]
            fig.add_trace(go.Scatter(
                x=[_MONTH_LABELS[m - 1] for m in yd["Month"]],
                y=yd["value"], mode="lines+markers", name=str(year),
            ))
        fig.update_layout(
            title=f"Raw Monthly {pw_met.upper()} — {selected_brand}",
            yaxis_title=pw_met.upper(), height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── STEP 1: Month Swaps ─────────────────────────────────────────────
    elif step_idx == 1:
        swap_events = [e for e in event_log
                       if e.get("type") == "swap" and e.get("scope") == "pre_trial"]
        if not swap_events:
            st.info("No month swaps were applied in this pipeline run.")
        else:
            st.markdown("**Swap events applied:**")
            for ev in swap_events:
                if "a_start" in ev:
                    st.write(f"  - {ev['a_start']} – {ev['a_end']}  <->  {ev['b_start']} – {ev['b_end']}")
                else:
                    ma, mb = ev.get("a"), ev.get("b")
                    st.write(f"  - Month {ma} ({_MONTH_LABELS[ma-1]})  <->  Month {mb} ({_MONTH_LABELS[mb-1]})")

            # Compare original vs swapped
            pw_met = st.selectbox("Metric", ["clicks", "quantity", "sales"],
                                  key="pw_s1_met")
            orig_brand = df_raw[df_raw["brand"] == selected_brand].copy()
            orig_brand["Date"] = pd.to_datetime(orig_brand["Date"])
            orig_brand["Month"] = orig_brand["Date"].dt.month
            orig_monthly = orig_brand.groupby("Month")[pw_met].sum().reset_index()

            if df_raw_mod is not None:
                mod_brand = df_raw_mod[df_raw_mod["brand"] == selected_brand].copy()
                mod_brand["Date"] = pd.to_datetime(mod_brand["Date"])
                mod_brand["Month"] = mod_brand["Date"].dt.month
                mod_monthly = mod_brand.groupby("Month")[pw_met].sum().reset_index()
            else:
                mod_monthly = orig_monthly.copy()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[_MONTH_LABELS[m - 1] for m in orig_monthly["Month"]],
                y=orig_monthly[pw_met], name="Before Swap",
                marker_color=_GREY,
            ))
            fig.add_trace(go.Bar(
                x=[_MONTH_LABELS[m - 1] for m in mod_monthly["Month"]],
                y=mod_monthly[pw_met], name="After Swap",
                marker_color=_ORANGE,
            ))
            fig.update_layout(
                barmode="group",
                title=f"{pw_met.upper()} — Before vs After Month Swap",
                yaxis_title=pw_met.upper(), height=420,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── STEP 2: Similarity Weights ──────────────────────────────────────
    elif step_idx == 2:
        if not norm_weights:
            st.info("No historical year weights computed (only one year of data or no trial data).")
        else:
            years_sorted = sorted(norm_weights.keys(), key=lambda k: norm_weights[k], reverse=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[str(y) for y in years_sorted],
                x=[norm_weights[y] for y in years_sorted],
                orientation="h",
                marker_color=_ORANGE,
                text=[f"{norm_weights[y]:.1%}" for y in years_sorted],
                textposition="auto",
            ))
            fig.update_layout(
                title="Historical Year Weights (65% component)",
                xaxis_title="Weight", height=max(200, len(years_sorted) * 50),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Years more similar to your trial period get higher weight in the DNA blend. "
                       "The remaining 35% comes from the Overall (all-years) pattern.")

            if adj_c and adj_q and adj_s:
                c1, c2, c3 = st.columns(3)
                c1.metric("Trial Clicks", f"{adj_c:,.0f}")
                c2.metric("Trial Quantity", f"{adj_q:,.0f}")
                c3.metric("Trial Sales", f"{adj_s:,.2f}")

    # ─── STEP 3: Pure DNA Profile ────────────────────────────────────────
    elif step_idx == 3:
        if pure_dna is None or pure_dna.empty:
            st.info("No pure DNA available.")
        else:
            fig = go.Figure()
            for col, label, color in [
                ("idx_clicks", "Clicks Index", _ORANGE),
                ("idx_cr",     "CR Index",     _BLUE),
                ("idx_aov",    "AOV Index",    "#2ca02c"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[_MONTH_LABELS[int(t) - 1] for t in pure_dna["TimeIdx"]],
                    y=pure_dna[col], mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                ))
            fig.add_hline(y=1.0, line_dash="dot", line_color=_GREY,
                          annotation_text="Baseline (1.0)")
            fig.update_layout(
                title="Pure DNA — Monthly Seasonality Indices",
                yaxis_title="Index", height=420,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Values > 1.0 mean that month is above average; < 1.0 means below average. "
                       "This is the blended result of 35% Overall + 65% weighted historical years.")

    # ─── STEP 4: DNA Layers ──────────────────────────────────────────────
    elif step_idx == 4:
        if df_proj is None:
            st.info("No projection data available.")
        else:
            pw_dna_met = st.selectbox("DNA Metric",
                                      ["clicks", "cr", "aov"], key="pw_s4_met")
            fig = go.Figure()
            for suffix, label, color, dash in [
                ("pure",     "Pure DNA",    _GREY,   "dot"),
                ("pretrial", "Pre-Trial",   _BLUE,   "dash"),
                ("work",     "Working",     _ORANGE, "solid"),
            ]:
                col = f"idx_{pw_dna_met}_{suffix}"
                if col in df_proj.columns:
                    fig.add_trace(go.Scatter(
                        x=df_proj["Date"], y=df_proj[col],
                        mode="lines", name=label,
                        line=dict(color=color, width=2, dash=dash),
                    ))
            fig.add_hline(y=1.0, line_dash="dot", line_color="#ddd")
            fig.update_layout(
                title=f"DNA Layers — idx_{pw_dna_met} (365 days)",
                yaxis_title="Index", height=420,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Pure = raw DNA blend. Pre-Trial = after pre-trial events (swaps, sculpts). "
                       "Working = after all events (post-trial sculpts, campaigns).")

    # ─── STEP 5: Base Calibration ────────────────────────────────────────
    elif step_idx == 5:
        if base_clicks is None:
            st.info("No calibration data available.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("base_clicks", f"{base_clicks:.4f}")
            c2.metric("base_cr", f"{base_cr:.6f}")
            c3.metric("base_aov", f"{base_aov:.2f}")

            if adj_c and df_proj is not None:
                t_start = st.session_state.get("ui_t_start")
                t_end = st.session_state.get("ui_t_end")
                if t_start and t_end:
                    t_mask = (df_proj["Date"].dt.date >= t_start) & (df_proj["Date"].dt.date <= t_end)
                    trial_sum = df_proj.loc[t_mask, "idx_clicks_pretrial"].sum()

                    st.markdown("**Calibration Formula:**")
                    st.code(
                        f"base_clicks = adj_clicks / sum(idx_clicks_pretrial[trial])\n"
                        f"            = {adj_c:,.0f} / {trial_sum:.4f}\n"
                        f"            = {base_clicks:.4f}",
                        language=None,
                    )

                    # Highlight trial window on DNA
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_proj["Date"], y=df_proj["idx_clicks_pretrial"],
                        mode="lines", name="idx_clicks_pretrial",
                        line=dict(color=_BLUE, width=1.5),
                    ))
                    fig.add_vrect(
                        x0=str(t_start), x1=str(t_end),
                        fillcolor=_ORANGE, opacity=0.15,
                        annotation_text="Trial Window",
                        annotation_position="top left",
                    )
                    fig.update_layout(
                        title="Trial Window on Pre-Trial DNA",
                        yaxis_title="idx_clicks_pretrial", height=350,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ─── STEP 6: Projections Built ───────────────────────────────────────
    elif step_idx == 6:
        if df_proj is None:
            st.info("No projection data available.")
        else:
            pw_proj_met = st.selectbox("Projection Metric",
                                       ["Clicks", "Qty", "Sales"], key="pw_s6_met")
            base_col = f"{pw_proj_met}_Base"
            sim_col = f"{pw_proj_met}_Sim"

            fig = go.Figure()
            if base_col in df_proj.columns:
                fig.add_trace(go.Scatter(
                    x=df_proj["Date"], y=df_proj[base_col],
                    mode="lines", name="Baseline (Before)",
                    line=dict(color=_BLUE, width=2),
                ))
            if sim_col in df_proj.columns:
                fig.add_trace(go.Scatter(
                    x=df_proj["Date"], y=df_proj[sim_col],
                    mode="lines", name="Simulation (After)",
                    line=dict(color=_ORANGE, width=2),
                ))

            # Annotate campaign shocks
            if "Shock" in df_proj.columns:
                shock_days = df_proj[df_proj["Shock"].abs() > 0.01]
                if not shock_days.empty:
                    shock_start = shock_days["Date"].iloc[0]
                    shock_end = shock_days["Date"].iloc[-1]
                    fig.add_vrect(
                        x0=str(shock_start.date()), x1=str(shock_end.date()),
                        fillcolor="rgba(244,121,32,0.08)", line_width=0,
                        annotation_text="Campaign Period",
                        annotation_position="top left",
                    )

            fig.update_layout(
                title=f"{pw_proj_met} — Baseline vs Simulation (365 days)",
                yaxis_title=pw_proj_met, height=420,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Baseline uses pre-trial DNA (no campaigns). "
                       "Simulation uses working DNA + shock multipliers + injections.")

    # ─── STEP 7: Historical Shrinkage ────────────────────────────────────
    elif step_idx == 7:
        if df_proj is None:
            st.info("No projection data available.")
        else:
            profiles_use = pipeline_cache.get("profiles_mod")
            if profiles_use is None:
                profiles_use = profiles

            # Compute historical medians
            m_prof = profiles_use[
                (profiles_use["brand"].isin(sel_brands))
                & (profiles_use["Level"] == "Monthly")
                & (profiles_use["Year"] != "Overall")
            ]

            pw_shrink_met = st.selectbox("Metric", ["Clicks", "Qty", "Sales"],
                                          key="pw_s7_met")
            prof_met_map = {"Clicks": "clicks", "Qty": "quantity", "Sales": "sales"}
            col_map = {"Clicks": "Clicks_Sim", "Qty": "Qty_Sim", "Sales": "Sales_Sim"}
            prof_met = prof_met_map[pw_shrink_met]
            sim_col = col_map[pw_shrink_met]

            if not m_prof.empty and sim_col in df_proj.columns:
                yr_agg = (
                    m_prof.groupby(["Year", "TimeIdx"])
                    .agg(**{prof_met: (prof_met, "sum")})
                    .reset_index()
                )
                hist_med = (
                    yr_agg.groupby("TimeIdx")[prof_met].median().reset_index()
                )
                hist_med.columns = ["Month", "Median"]

                fc_monthly = df_proj.groupby("Month")[sim_col].sum().reset_index()
                fc_monthly.columns = ["Month", "Forecast"]

                merged = hist_med.merge(fc_monthly, on="Month", how="left").fillna(0)
                merged["Deviation"] = (merged["Forecast"] / merged["Median"].replace(0, np.nan)).round(3)
                merged["Shrunk"] = merged["Deviation"].apply(
                    lambda d: "Yes" if (d > 2 or d < 0.5) and np.isfinite(d) else "No"
                )

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[_MONTH_LABELS[m - 1] for m in merged["Month"]],
                    y=merged["Median"], name="Historical Median",
                    marker_color=_GREY,
                ))
                fig.add_trace(go.Bar(
                    x=[_MONTH_LABELS[m - 1] for m in merged["Month"]],
                    y=merged["Forecast"], name="Forecast (after shrinkage)",
                    marker_color=_ORANGE,
                ))
                fig.update_layout(
                    barmode="group",
                    title=f"{pw_shrink_met} — Forecast vs Historical Median per Month",
                    yaxis_title=pw_shrink_met, height=420,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                n_shrunk = (merged["Shrunk"] == "Yes").sum()
                st.caption(f"{n_shrunk} month(s) had deviation > 2x and were blended toward the median.")

                show_tbl = merged[["Month", "Median", "Forecast", "Deviation", "Shrunk"]].copy()
                show_tbl["Month"] = show_tbl["Month"].apply(lambda m: _MONTH_LABELS[m - 1])
                show_tbl["Median"] = show_tbl["Median"].apply(lambda x: f"{x:,.0f}")
                show_tbl["Forecast"] = show_tbl["Forecast"].apply(lambda x: f"{x:,.0f}")
                st.dataframe(show_tbl, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough profile data to show shrinkage analysis.")

    # ─── STEP 8: Final Forecast + Noise Bands ────────────────────────────
    elif step_idx == 8:
        if df_proj is None:
            st.info("No projection data available.")
        else:
            pw_final_met = st.selectbox("Metric", ["Clicks", "Qty", "Sales"],
                                         key="pw_s8_met")
            sim_col = f"{pw_final_met}_Sim"
            min_col = f"{pw_final_met}_Sim_Min"
            max_col = f"{pw_final_met}_Sim_Max"

            fig = go.Figure()

            # Noise band
            if min_col in df_proj.columns and max_col in df_proj.columns:
                fig.add_trace(go.Scatter(
                    x=list(df_proj["Date"]) + list(df_proj["Date"][::-1]),
                    y=list(df_proj[max_col]) + list(df_proj[min_col][::-1]),
                    fill="toself",
                    fillcolor="rgba(244,121,32,0.12)",
                    line=dict(width=0),
                    name="Noise Band (P10-P90)",
                ))

            # Forecast line
            if sim_col in df_proj.columns:
                fig.add_trace(go.Scatter(
                    x=df_proj["Date"], y=df_proj[sim_col],
                    mode="lines", name="Forecast",
                    line=dict(color=_ORANGE, width=2),
                ))

            # Historical median reference
            profiles_use = pipeline_cache.get("profiles_mod") if pipeline_cache.get("profiles_mod") is not None else profiles
            m_prof = profiles_use[
                (profiles_use["brand"].isin(sel_brands))
                & (profiles_use["Level"] == "Monthly")
                & (profiles_use["Year"] != "Overall")
            ]
            prof_met_map = {"Clicks": "clicks", "Qty": "quantity", "Sales": "sales"}
            prof_met = prof_met_map.get(pw_final_met, "sales")

            if not m_prof.empty:
                yr_agg = m_prof.groupby(["Year", "TimeIdx"]).agg(
                    **{prof_met: (prof_met, "sum")}
                ).reset_index()
                med_by_month = yr_agg.groupby("TimeIdx")[prof_met].median()

                # Map monthly median to each day
                med_daily = df_proj["Month"].map(med_by_month)
                # Scale: divide by days in month for daily equivalent
                days_per_month = df_proj.groupby("Month")["Date"].transform("count")
                med_daily_scaled = med_daily / days_per_month

                fig.add_trace(go.Scatter(
                    x=df_proj["Date"], y=med_daily_scaled,
                    mode="lines", name="Historical Median (daily)",
                    line=dict(color=_BLUE, width=1.5, dash="dot"),
                ))

            fig.update_layout(
                title=f"{pw_final_met} — Final Forecast with Noise Bands",
                yaxis_title=pw_final_met, height=450,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            if sim_col in df_proj.columns:
                total = df_proj[sim_col].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Total {pw_final_met} (Year)", f"{total:,.0f}")
                if min_col in df_proj.columns:
                    c2.metric("Lower Bound (P10)", f"{df_proj[min_col].sum():,.0f}")
                if max_col in df_proj.columns:
                    c3.metric("Upper Bound (P90)", f"{df_proj[max_col].sum():,.0f}")
