"""Risk & Uncertainty Calculation page."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from engine.risk import run_risk_pipeline

_TMPL   = "plotly_white"
_C_BASE = "#94a3b8"
_C_SIM  = "#1a1a6b"

_RISK_COLORS = {
    "Low":    {"bg": "#dcfce7", "border": "#16a34a", "text": "#166534"},
    "Medium": {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e"},
    "High":   {"bg": "#fee2e2", "border": "#dc2626", "text": "#991b1b"},
}


def _render_step_toolbar(step_key, on_undo, on_reset):
    """Render Undo / Reset inline buttons for a workflow step."""
    c_undo, c_reset, _ = st.columns([1, 1, 6])
    if c_undo.button("Undo", key=f"tb_undo_{step_key}", use_container_width=True):
        on_undo()
    if c_reset.button("Reset", key=f"tb_reset_{step_key}", use_container_width=True):
        on_reset()
    st.markdown("---")


def render_risk(df_raw, sel_brands, df):
    """Render the Risk & Uncertainty Calculation page.

    Risk bands wrap the DNA forecast — the DNA projection is the center line,
    and multiple statistical models provide uncertainty quantification around it.
    """

    def _undo():
        st.toast("Nothing to undo — Risk is a read-only analysis.")

    def _reset():
        st.toast("Nothing to reset — Risk is a read-only analysis.")

    _render_step_toolbar("nav_risk", _undo, _reset)

    st.markdown(
        "Evaluate forecast uncertainty using multiple statistical models. "
        "Model divergence is applied around the DNA forecast to produce "
        "confidence bands for risk assessment."
    )

    metric = st.selectbox(
        "Forecast Metric",
        ["sales", "quantity", "clicks"],
        key="risk_metric",
    )

    result = run_risk_pipeline(df_raw, sel_brands, df, metric=metric)

    if result is None:
        st.warning(
            "Not enough historical data for risk estimation. "
            "Need at least 6 months of data for the selected brands."
        )
        return

    risk = result["risk_summary"]
    rcolors = _RISK_COLORS.get(risk["risk_label"], _RISK_COLORS["Medium"])

    # ── Risk Summary Card ────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="
            background:{rcolors['bg']};
            border-left:5px solid {rcolors['border']};
            border-radius:10px;
            padding:18px 22px;
            margin:16px 0;
        ">
          <div style="font-family:Inter,sans-serif;font-size:0.72rem;font-weight:700;
               text-transform:uppercase;letter-spacing:0.1em;color:{rcolors['text']}">
            Forecast Risk Level
          </div>
          <div style="font-family:Inter,sans-serif;font-size:1.8rem;font-weight:700;
               color:{rcolors['text']};margin:4px 0">
            {risk['risk_label']}
          </div>
          <div style="font-family:Inter,sans-serif;font-size:0.85rem;color:{rcolors['text']}">
            Average spread: {risk['spread_pct']:.0%} &nbsp;|&nbsp;
            Volatility: {risk['volatility']:,.0f} &nbsp;|&nbsp;
            Best model: {risk['best_model']} (MAPE: {risk['best_mape']:.1%})
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Model Performance ────────────────────────────────────────────────────
    st.markdown("##### Model Comparison")
    n_models = len(result["model_scores"])
    model_cols = st.columns(min(n_models, 4))
    for i, (name, mape) in enumerate(result["model_scores"].items()):
        weight = result["weights"].get(name, 0)
        col = model_cols[i % len(model_cols)]
        col.metric(
            name,
            f"MAPE: {mape:.1%}",
            f"Weight: {weight:.0%}",
        )

    # ── DNA Forecast with Risk Bands ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### DNA Forecast with Risk Bands")

    history = result["history"]
    fc_df   = result["forecasts"]
    n_months = len(fc_df)

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=history["Date"], y=history["value"],
        mode="lines+markers", line=dict(color=_C_BASE, width=2),
        marker=dict(size=4),
        name=f"Historical {metric.title()}",
    ))

    # 10-90 band
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["upper_90"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["lower_10"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(26,26,107,0.08)",
        name="10th–90th Percentile",
    ))

    # 25-75 band
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["upper_75"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["lower_25"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(26,26,107,0.15)",
        name="25th–75th Percentile",
    ))

    # DNA Forecast line (center)
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["dna_forecast"],
        mode="lines+markers", line=dict(color=_C_SIM, width=3),
        marker=dict(size=5),
        name="DNA Forecast",
    ))

    # Model Median
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["median"],
        mode="lines", line=dict(color=_C_BASE, dash="dash", width=1.5),
        name="Model Median",
    ))

    fig.update_layout(
        template=_TMPL,
        height=420,
        title=dict(
            text=f"Risk-Adjusted {metric.title()} Forecast ({n_months} months)",
            font=dict(color="#12124a"),
        ),
        xaxis_title="Date",
        yaxis_title=metric.title(),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Individual Model Divergence ──────────────────────────────────────────
    with st.expander("Individual Model Forecasts (adjusted to DNA)", expanded=False):
        colors = ["#f59e0b", "#10b981", "#8b5cf6", "#dc2626"]
        fig_models = go.Figure()

        fig_models.add_trace(go.Scatter(
            x=history["Date"], y=history["value"],
            mode="lines", line=dict(color=_C_BASE, width=2),
            name="Historical",
        ))

        for i, name in enumerate(result["model_scores"].keys()):
            if name in fc_df.columns:
                fig_models.add_trace(go.Scatter(
                    x=fc_df["Date"], y=fc_df[name],
                    mode="lines+markers",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    name=name,
                ))

        fig_models.add_trace(go.Scatter(
            x=fc_df["Date"], y=fc_df["dna_forecast"],
            mode="lines", line=dict(color=_C_SIM, width=3, dash="dash"),
            name="DNA Forecast",
        ))

        fig_models.update_layout(
            template=_TMPL, height=350,
            title="Model Divergence Around DNA Forecast",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_models, use_container_width=True)

    # ── Forecast Data Table ──────────────────────────────────────────────────
    with st.expander("Forecast Data Table", expanded=False):
        display_df = fc_df.copy()
        display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m")
        num_cols = [c for c in display_df.columns if c != "Date"]
        st.dataframe(
            display_df.style.format({c: "{:,.0f}" for c in num_cols}),
            use_container_width=True,
            hide_index=True,
        )

    # ── Confirm step complete ──
    st.markdown("---")
    if st.button("Confirm Risk Analysis →", type="primary", use_container_width=True):
        st.session_state.step_completed["nav_risk"] = True
        st.session_state.nav_page = "nav_audit"
        st.rerun()
