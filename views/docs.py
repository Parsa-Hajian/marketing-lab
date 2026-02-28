"""Documentation page: formulas, architecture, assumptions, and usage guide."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from engine.i18n import t

_TMPL = "plotly_white"


def _campaign_shape_fig():
    duration = 30
    tt = np.arange(duration)
    p  = tt / duration
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tt, y=np.exp(-3.0 * p),
        name="Push/DEM (Front-Loaded)", line=dict(width=2.5, color="#F47920")))
    fig.add_trace(go.Scatter(
        x=tt, y=1 - p,
        name="High Discount (Linear Fade)", line=dict(width=2.5, color="#10B981")))
    fig.add_trace(go.Scatter(
        x=tt, y=np.exp(-((tt - duration * 0.4) ** 2) / (2 * (duration * 0.3) ** 2)),
        name="Product Launch (Delayed Peak)", line=dict(width=2.5, color="#8B5CF6")))
    fig.add_trace(go.Scatter(
        x=tt, y=np.ones(duration),
        name="Field Campaign (Step)", line=dict(width=2.5, color="#EF4444")))
    fig.update_layout(
        template=_TMPL, height=300,
        title="Campaign Shape Functions (normalised lift over a 30-day event)",
        xaxis_title="Days elapsed", yaxis_title="Lift multiplier (fraction of peak)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def render_docs(lang: str = "en"):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ Architecture",
        "📐 Models & Formulas",
        "⚙️ Assumptions",
        "📊 Metric Selection Guide",
        "🚀 How to Use",
    ])

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 1: ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("## Platform Architecture")
        st.markdown(
            "**Tech Strategy Lab** is a decision-support platform for marketing and commercial teams. "
            "It combines historical demand modeling with forward-looking scenario simulation to answer:\n\n"
            "> *What should my business do organically — and what will my planned marketing activities contribute?*"
        )

        st.markdown("---")
        st.markdown("### Three-Layer Model")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "<div style='background:#EEF2FF;border-radius:10px;padding:16px 18px'>"
                "<div style='font-size:1.4rem'>🧬</div>"
                "<strong>Layer 1 — DNA</strong><br>"
                "<span style='font-size:0.85rem;color:#555'>Historical demand pattern modeling. "
                "Captures the seasonal shape of Clicks, CR, and AOV as normalized indices "
                "at monthly, weekly, and daily granularity.</span></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<div style='background:#F0FDF4;border-radius:10px;padding:16px 18px'>"
                "<div style='font-size:1.4rem'>🎯</div>"
                "<strong>Layer 2 — Calibration</strong><br>"
                "<span style='font-size:0.85rem;color:#555'>Anchors the model to current reality "
                "using a known trial period. Computes base constants (base_clicks, base_CR, base_AOV) "
                "that reconcile history with today's observations.</span></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                "<div style='background:#FFF7ED;border-radius:10px;padding:16px 18px'>"
                "<div style='font-size:1.4rem'>📈</div>"
                "<strong>Layer 3 — Simulation</strong><br>"
                "<span style='font-size:0.85rem;color:#555'>Projects full-year performance and "
                "applies campaign events (shocks, DNA adjustments, re-injections). "
                "Shows Before/After views with ±15% confidence bands.</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Data Flow")
        st.code(
            """super_dataset.csv   (daily clicks / quantity / sales per brand)
        │
        ▼
individual_brand_profiles_granular.csv  (pre-computed DNA indices — Monthly / Weekly / Daily)
        │
        ▼
[DNA Engine]  compute_similarity_weights()  →  35/65 blend  →  pure_dna
        │
        ▼
[Calibration]  calibrate_base()  →  base_clicks, base_CR, base_AOV
        │
        ▼
[Projection]  build_projections()  →  Baseline, Simulation, ±15% bands
        │
        ▼
Dashboard  ·  Lab  ·  Audit & Attribution""",
            language="text",
        )

        st.markdown("---")
        st.markdown("### Key Concepts")
        with st.expander("What is the DNA?"):
            st.markdown(
                "The **DNA (Demand Normalization Architecture)** is a set of normalized index series "
                "representing the *shape* of historical business performance. "
                "An index of **1.0** = the historical median for that time period. "
                "Values **> 1** indicate above-average periods; **< 1** indicate below-average.\n\n"
                "Three indices are tracked per period: **Clicks** (traffic demand), "
                "**CR** (conversion tendency), **AOV** (basket size tendency).\n\n"
                "The DNA separates *amplitude* (calibrated to current observations) "
                "from *shape* (learned from history)."
            )
        with st.expander("What is calibration?"):
            st.markdown(
                "**Calibration** anchors the model to current reality. "
                "You provide observed metrics for a 'trial period' (a recent window of actual data). "
                "The engine computes base constants that, when multiplied by the historical DNA shape, "
                "reproduce those observed values — then extrapolates the full year.\n\n"
                "Think of calibration as telling the model: *'Here is what my business actually did. "
                "Now project forward using the historical shape.'*"
            )
        with st.expander("What is the 35/65 DNA blend?"):
            st.markdown(
                "The Pure DNA blends two signals:\n\n"
                "- **35% Overall** — the long-run median across all historical years\n"
                "- **65% Historical** — weighted average of per-year patterns\n\n"
                "The year weights are computed by **inverse similarity error**: "
                "years whose pattern was closest to your current trial period get higher weight. "
                "This makes the model adapt to changing business conditions year by year."
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 2: MODELS & FORMULAS
    # ─────────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("## Models & Formulas")

        st.markdown("### 1. DNA Similarity Weights")
        st.info("**Goal:** determine which historical year's pattern is most similar to the "
                "current trial period, so we can blend it more heavily.")
        st.code(
            """For each historical year i:
    err_i = ( |ΔClicks| + |ΔQuantity| + |ΔSales| ) / 3
           where Δ = (observed_trial − historical_trial) / observed_trial

    w_i   = 1 / (err_i + 0.01)        ← inverse-error weight (ε=0.01 avoids ÷0)

    norm_w_i = w_i / Σ w_j             ← normalised to sum to 1""",
            language="text",
        )

        st.markdown("### 2. Pure DNA Construction")
        st.info("**Goal:** build the baseline monthly demand shape as a blended historical profile.")
        st.code(
            """For each calendar month t:

    Pure_DNA(t) = 0.35 × Overall_median(t)
                + 0.65 × Σᵢ [ norm_w_i × Year_i_median(t) ]

where Overall_median(t) is the median index across all years for month t,
and Year_i_median(t) is the median index for year i at month t.""",
            language="text",
        )

        st.markdown("### 3. Calibration")
        st.info("**Goal:** compute base constants from observed trial data.")
        st.code(
            """Given trial period [t_start, t_end] with observed:
    adj_clicks, adj_quantity, adj_sales

base_clicks = adj_clicks / Σₜ( idx_clicks_pretrial(t) )
base_CR     = trial_CR  / mean( idx_cr_pretrial(t) )
base_AOV    = trial_AOV / mean( idx_aov_pretrial(t) )

where  trial_CR  = adj_quantity / adj_clicks
       trial_AOV = adj_sales    / adj_quantity""",
            language="text",
        )

        st.markdown("### 4. Projections")
        st.info("**Goal:** produce full-year baseline and simulation forecasts.")
        st.code(
            """Baseline (no events — uses pre-trial DNA):
    Clicks(t)   = base_clicks × idx_clicks_pretrial(t)
    Quantity(t) = Clicks(t)   × base_CR  × idx_cr_pretrial(t)
    Sales(t)    = Quantity(t) × base_AOV × idx_aov_pretrial(t)

Simulation (with events — uses work DNA + shocks):
    Clicks_sim(t)   = base_clicks × idx_clicks_work(t) × (1 + Shock(t))
    Quantity_sim(t) = Clicks_sim(t) × base_CR  × idx_cr_work(t)
    Sales_sim(t)    = Quantity_sim(t) × base_AOV × idx_aov_work(t)

Confidence margins: ±15% on all projected values.""",
            language="text",
        )

        st.markdown("### 5. Campaign Shape Functions")
        st.markdown(
            "Each campaign distributes its traffic lift over the campaign window. "
            "`p = days_elapsed / duration`  (0 → 1)."
        )
        st.plotly_chart(_campaign_shape_fig(), use_container_width=True)

        shapes_df = pd.DataFrame({
            "Shape":    ["Push/DEM", "High Discount", "Product Launch", "Field Campaign"],
            "Type":     ["Front-Loaded", "Linear Fade", "Delayed Peak", "Step"],
            "Formula":  [
                "lift × e^(−3p)",
                "lift × (1 − p)",
                "lift × exp(−((t − 0.4d)² / (2(0.3d)²)))",
                "lift × 1",
            ],
            "Best for": [
                "Push notifications, DEM email blasts",
                "Discount promotions, flash sales",
                "New product reveals, launches",
                "Field activities, brand campaigns",
            ],
        })
        st.dataframe(shapes_df, use_container_width=True, hide_index=True)

        st.markdown("### 6. De-Shock Isolation")
        st.info("**Goal:** separate the artificial (event-driven) component from the organic baseline.")
        st.code(
            """Given shock window [ds_start, ds_end]:

    floor = 10th percentile of clicks within [ds_start, ds_end]
          (conservative estimate of organic floor during the event)

    ΔClicks(t) = max(0, observed_clicks(t) − floor)

Organic CR  = floor_quantity / floor_clicks
Event CR    = ΔQuantity.sum() / ΔClicks.sum()
CR Delta    = Event CR − Organic CR   (quality of event traffic vs. organic)""",
            language="text",
        )

        st.markdown("### 7. Attribution Engine")
        st.info("**Goal:** measure the marginal contribution of each event to the target metric.")
        st.code(
            """For each event i in the active event list (in order):

    Contribution_i = metric(events[0..i]) − metric(events[0..i−1])
    Coverage_i (%) = Contribution_i / (Target − Organic_Base) × 100

This is a sequential 'leave-one-in' attribution:
events are credited in the order they were added.""",
            language="text",
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 3: ASSUMPTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("## Model Assumptions & Requirements")
        st.warning(
            "All quantitative models rely on assumptions. Understanding these helps you "
            "interpret results correctly and know when the model may be less reliable.")

        st.markdown("### Data Requirements")
        reqs = [
            ("Minimum history",
             "At least **12 months** of daily data per brand for reliable DNA. 2+ years recommended."),
            ("Metric consistency",
             "Your data must satisfy: **CR = Quantity ÷ Clicks** and **AOV = Sales ÷ Quantity** "
             "consistently throughout the dataset."),
            ("Daily granularity",
             "The transactions file must have one row per brand per calendar day. "
             "The DNA system aggregates up to Monthly/Weekly/Daily resolutions."),
            ("Numeric stability",
             "Clicks should be > 0 for most days. Very sparse data (< 1 click/day on average) "
             "may produce unstable DNA indices."),
        ]
        for title, desc in reqs:
            st.markdown(f"**{title}:** {desc}")

        st.markdown("---")
        st.markdown("### Model Assumptions")
        assumptions = [
            ("1. Demand shape stationarity",
             "The seasonal pattern (shape) of Clicks, CR, and AOV is assumed stable across years. "
             "Only the amplitude can change. If the brand underwent a structural break "
             "(e.g., major product pivot, market entry), the DNA from pre-break years "
             "may distort the model. Use the DNA similarity weights display to check: "
             "if one year dominates near 100%, earlier years are being discounted automatically."),
            ("2. Additive campaign effects",
             "Multiple campaign events are assumed to have **additive** effects on Clicks. "
             "In reality, overlapping campaigns may have diminishing returns (saturation) "
             "or amplifying effects (cross-channel synergy). These interactions are not modeled."),
            ("3. 10th-percentile organic floor",
             "In the De-Shock tool, the 10th percentile of Clicks within the shock window "
             "is used as the organic baseline. This is **conservative**: it assumes the lowest "
             "10% of traffic during the event period is organic. For short, sharp events "
             "(< 2 weeks) it works well. For long events, consider a tighter window."),
            ("4. ±15% uniform confidence margin",
             "All projections carry a ±15% uncertainty band. This is a heuristic and does not "
             "reflect actual statistical confidence intervals (which depend on data volume, "
             "noise level, and structural stability). Higher-noise brands warrant wider margins."),
            ("5. Linear CR and AOV index scaling",
             "Conversion Rate and AOV scale multiplicatively with their DNA indices. "
             "This implies that the CR and AOV *shape* is stable across time. "
             "In practice, CR often varies with acquisition channel mix, which is not modeled here."),
            ("6. Trial period representativeness",
             "The trial period you select must represent **normal business conditions**. "
             "Do not select a trial period that coincides with a major event (flash sale, "
             "seasonal peak) unless you use the Pre-Adjustment feature to strip the lift. "
             "An unrepresentative trial period will miscalibrate the entire projection."),
            ("7. No external confounder modeling",
             "The model does not account for: macroeconomic shocks, competitor actions, "
             "platform algorithm changes, data pipeline interruptions, or major holidays "
             "not already captured in the historical pattern."),
            ("8. Sales identity: Sales = Quantity × AOV",
             "Sales = Quantity × AOV is assumed throughout. If your sales data includes "
             "non-transactional sources (returns, subscription revenue, etc.), "
             "the Sales projections will be systematically biased."),
            ("9. Day-of-week effects at monthly resolution",
             "When using Monthly resolution, weekend suppression and weekday effects are "
             "averaged out. Switch to Daily resolution to capture day-of-week patterns explicitly."),
            ("10. Attribution is sequential, not causal",
             "The Attribution Engine assigns credit in the order events were added to the log. "
             "This is a sequential marginal contribution model, not a causal model. "
             "Reordering events will change individual contributions. "
             "For a more robust attribution, use Shapley-value averaging (not currently implemented)."),
        ]
        for title, desc in assumptions:
            with st.expander(title):
                st.markdown(desc)

        st.markdown("---")
        st.markdown("### Known Limitations")
        st.markdown(
            "- **No uncertainty propagation**: errors in calibration inputs are not propagated "
            "to the confidence bands.\n"
            "- **No channel-level granularity**: the model operates at brand level, "
            "not channel level (paid search vs. organic vs. email).\n"
            "- **No competitor modeling**: market share shifts are not captured.\n"
            "- **DNA computed at monthly granularity for projections**: even at Weekly/Daily "
            "resolution display, the baseline DNA is always blended monthly for the core projection.\n"
            "- **Re-injection signatures are brand-aggregated**: the de-shock tool "
            "aggregates across selected brands, not per-brand."
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 4: METRIC SELECTION GUIDE
    # ─────────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("## Metric Selection Guide")
        st.markdown(
            "For the model to work correctly, your three core metrics must satisfy "
            "a set of coherence conditions. This guide helps you choose correctly."
        )

        st.markdown("### The Three Core Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                "<div style='background:#EEF2FF;border-radius:8px;padding:14px'>"
                "<strong>Clicks</strong><br>"
                "<span style='font-size:0.85rem;color:#555'>"
                "Your primary <b>traffic / demand volume</b> metric. "
                "Numerator in the CR calculation. Should respond to campaign activity.<br><br>"
                "<em>Examples: website clicks, product page views, app opens, "
                "email link clicks, ad clicks.</em>"
                "</span></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                "<div style='background:#F0FDF4;border-radius:8px;padding:14px'>"
                "<strong>Quantity</strong><br>"
                "<span style='font-size:0.85rem;color:#555'>"
                "Your primary <b>conversion / order volume</b>. "
                "Denominator in AOV and numerator in CR.<br><br>"
                "<em>Examples: units sold, orders placed, leads converted, "
                "appointments booked.</em>"
                "</span></div>",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                "<div style='background:#FFF7ED;border-radius:8px;padding:14px'>"
                "<strong>Sales</strong><br>"
                "<span style='font-size:0.85rem;color:#555'>"
                "Monetary value of your conversions. "
                "Must equal Quantity × AOV within acceptable tolerance.<br><br>"
                "<em>Examples: gross sales, net revenue, GMV, "
                "contract value.</em>"
                "</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Coherence Checklist")
        st.success("Before entering data, verify these identities hold within ±5% tolerance:")
        st.code(
            """CR  = Quantity / Clicks         (e.g. 100 orders / 5,000 clicks = 2.0% CR)
AOV = Sales    / Quantity       (e.g. €10,000 sales / 100 orders = €100 AOV)
Sales = Clicks × CR × AOV      (the fundamental identity)""",
            language="text",
        )

        st.markdown("---")
        st.markdown("### Choosing the Right Trial Period")
        with st.expander("What is a trial period?"):
            st.markdown(
                "The **trial period** is a recent window of actual data you give the model "
                "to anchor its projections. You enter three numbers: total Clicks, "
                "Quantity, and Sales for that period.\n\n"
                "The model uses these to compute `base_clicks`, `base_CR`, and `base_AOV`, "
                "which scale the historical DNA to produce current-level projections."
            )
        with st.expander("Good vs. bad trial periods"):
            good_col, bad_col = st.columns(2)
            with good_col:
                st.markdown("**Good trial periods:**")
                st.markdown(
                    "- Recent (last 1–3 months)\n"
                    "- Representative of normal operations\n"
                    "- No major events or anomalies\n"
                    "- At least 7 days (ideally 14–30)\n"
                    "- Fully closed (data is complete)")
            with bad_col:
                st.markdown("**Avoid as trial periods:**")
                st.markdown(
                    "- Periods with major campaigns running\n"
                    "- Black Friday / holiday peak\n"
                    "- Outage or data gaps\n"
                    "- Only 1–2 days\n"
                    "- Future dates (no actual data)")
        with st.expander("Using the Pre-Adjustment"):
            st.markdown(
                "If your best available trial period had a known event (e.g., a 30% traffic boost "
                "from a DEM), use the **Pre-Adjustment** slider:\n\n"
                "- Enter `+30%` in 'Clicks adj (%)' to strip the lift.\n"
                "- The model divides your observed clicks by 1.30, calibrating to the organic base.\n\n"
                "Formula: `adj_clicks = raw_clicks / (1 + adj_pct / 100)`\n\n"
                "A **positive** adjustment means the trial was inflated → strip lift.\n"
                "A **negative** adjustment means the trial was suppressed → add lift back."
            )

        st.markdown("---")
        st.markdown("### When Metrics Behave Unexpectedly")
        issues = {
            "CR is very high (> 20%)":
                "Check that Clicks and Quantity use the same definition. "
                "If Clicks = add-to-cart events and Quantity = purchases, "
                "CR will be inflated vs. site-wide CR. Use consistent definitions.",
            "AOV varies wildly by month":
                "High AOV volatility may indicate product mix shifts or "
                "currency effects. The DNA AOV index will capture the shape, "
                "but consider whether the variation is structural or noise.",
            "Sales doesn't match Clicks × CR × AOV":
                "Sales may include subscription revenue, returns/refunds, "
                "or FX effects. Ensure Sales is the gross transactional value "
                "from direct conversions only.",
            "DNA weights show 100% on one year":
                "One year dominates because it was most similar to your trial period. "
                "This is normal if that year had a very similar seasonal pattern. "
                "If it seems wrong, double-check your trial period dates and values.",
            "Baseline projection seems too high/low":
                "Re-check your Trial Reality: total Clicks, Quantity, and Sales "
                "must match what actually happened in those dates. "
                "Also verify the Pre-Adjustment is set correctly.",
        }
        for issue, solution in issues.items():
            with st.expander(f"⚠️ {issue}"):
                st.markdown(solution)

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 5: HOW TO USE
    # ─────────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("## Complete Usage Guide")
        st.info(
            "Follow these steps in order for your first session. "
            "After setup (steps 1–3), you can go straight to simulation in future sessions.")

        steps = [
            ("Step 1 — Log In",
             "Enter your Full Name, Username, and Password on the login screen. "
             "The full name is logged with every action you take in the platform.\n\n"
             "Your session state (events, targets, trial period) persists until you sign out "
             "or the server restarts."),

            ("Step 2 — Add / Update Brand Data (first time only)",
             "Navigate to **➕ Add Brand** to upload a new brand's historical CSV, or "
             "**✏️ Update Brand** to replace or extend an existing brand's data.\n\n"
             "**CSV format required:**\n"
             "```\nDate, brand, clicks, quantity, sales\n```\n"
             "- Date: `YYYY-MM-DD` format\n"
             "- brand: lowercase string\n"
             "- clicks, quantity, sales: numeric\n\n"
             "The system will recompute the brand's DNA profiles automatically after upload."),

            ("Step 3 — Configure the Sidebar",
             "The sidebar controls the model for the entire session:\n\n"
             "**Market Resolution** — Monthly, Weekly, or Daily. "
             "Monthly is fastest and best for strategic planning; "
             "Daily gives the most granular view.\n\n"
             "**DNA Brands** — select which brands to include. "
             "Multi-brand mode blends their DNA profiles. "
             "Single-brand mode unlocks the Goal Tracker's historical chart.\n\n"
             "**Trial Reality** — enter your observed actual numbers:\n"
             "- Start/End Date: a known period with complete actual data\n"
             "- Clicks: total clicks during that period\n"
             "- Quantity: total orders/conversions\n"
             "- Sales: total sales value\n\n"
             "**Pre-Adjustment** — if the trial period had a known event, "
             "adjust by the estimated lift % to strip it (positive = strip boost, "
             "negative = add suppressed demand back).\n\n"
             "**DNA Weights** — the sidebar shows how much weight each historical year gets. "
             "35% goes to the all-time overall; 65% is distributed across years by "
             "similarity to your trial period."),

            ("Step 4 — Dashboard: Projection Overview",
             "Navigate to **📊 Dashboard** → **📈 Projection Overview**.\n\n"
             "- Select a metric (Sales, Clicks, Qty, CR, AOV)\n"
             "- The chart shows the **Baseline** (organic, no campaigns) with ±15% bands\n"
             "- As you add events in the Lab, a **Forecast** line appears "
             "showing the simulated outcome\n"
             "- Shaded windows mark campaign periods\n\n"
             "If the baseline looks wrong, return to the sidebar and check your Trial Reality values."),

            ("Step 5 — Dashboard: Goal Tracker",
             "Navigate to **📊 Dashboard** → **🎯 Goal Tracker**.\n\n"
             "1. *(Single brand only)* Select a historical year and growth % "
             "to compute a target automatically\n"
             "2. Set your **Target Period** (future planning window)\n"
             "3. Select your **Target Metric** (Sales, Qty, Clicks, CR, or AOV)\n"
             "4. Choose the **Volume Driver**: more traffic, higher CR, or higher AOV?\n"
             "5. Review the **KPI Matrix**: needed vs. baseline\n"
             "6. Add campaigns in the Lab and return to see if the gap closes\n\n"
             "Gap charts show period-by-period surplus (+) or shortfall (−) vs. target."),

            ("Step 6 — Dashboard: DNA Profile",
             "Navigate to **📊 Dashboard** → **🧬 Market DNA Profile**.\n\n"
             "Shows seasonal index patterns for Clicks (orange), CR (teal), and AOV (violet).\n\n"
             "**Three layers:**\n"
             "- **Pure** (dashed, light): raw blended historical shape\n"
             "- **Pre-Trial** (dashed, medium): after any Pre-Trial DNA edits\n"
             "- **Work** (solid, bold): final shape after all modifications\n\n"
             "Use this to understand when your business naturally peaks and troughs "
             "before planning campaign timing."),

            ("Step 7 — Lab: Inject a Campaign",
             "Navigate to **⚡ Lab** → **🚀 Events**.\n\n"
             "1. Select campaign Start and End dates\n"
             "2. Choose a **Campaign Shape**:\n"
             "   - **Push/DEM**: front-heavy, most impact on day 1, decays fast\n"
             "   - **High Discount**: linear decay from day 1\n"
             "   - **Product Launch**: builds to a peak at ~40% of duration, then decays\n"
             "   - **Field Campaign**: constant lift throughout\n"
             "3. Set **Traffic Lift (%)**: expected clicks increase vs. baseline\n"
             "   *(default loaded from Settings — override as needed)*\n"
             "4. Click **Inject Campaign**\n\n"
             "The campaign appears immediately in all Dashboard charts. "
             "Repeat for multiple campaigns.\n\n"
             "**Swap DNA**: the right-hand column lets you swap the demand profile "
             "of two time periods. Choose Pre-Trial to affect calibration, "
             "or Post-Trial for projection only."),

            ("Step 8 — Lab: DNA Drag",
             "Navigate to **⚡ Lab** → **🖱️ Visual DNA Drag**.\n\n"
             "1. Click a point on the chart to select a time period\n"
             "2. Set the Multiplier (×1.0 = no change, ×2.0 = double that period's index)\n"
             "3. Choose Pre-Trial or Post-Trial scope\n"
             "4. Click Apply\n\n"
             "**Use case:** you know a new product launch will shift your November peak "
             "to October. Drag October's multiplier up and November's down."),

            ("Step 9 — Lab: De-Shock Tool",
             "Navigate to **⚡ Lab** → **🧹 De-Shock Tool**.\n\n"
             "1. Pick a date window where you know a campaign ran historically\n"
             "2. Review the forensic chart: organic floor (10th pct) = red dashed line\n"
             "3. The green area above the floor = extracted shock (artificial demand)\n"
             "4. Review: Δ Clicks, Δ Orders, Δ Sales, Event CR vs. Organic CR\n"
             "5. Name and **Save to Library**\n\n"
             "**Signature Library**: once saved, re-inject the signature into any future date. "
             "Two modes:\n"
             "- **Absolute Volume**: replays exact historical daily increments\n"
             "- **Relative Baseline Multiplier**: scales to current forecast baseline "
             "(recommended when scale has changed year-over-year)"),

            ("Step 10 — Lab: Audit & Attribution",
             "Navigate to **⚡ Lab** → **📋 Audit & Gap Attribution**.\n\n"
             "Shows every active event and its **marginal contribution** to the "
             "target metric set in the Goal Tracker:\n\n"
             "- Each event shows: metric delta (absolute) and gap coverage (%)\n"
             "- **Shift (↔)**: move a campaign to a different start date without re-entering it\n"
             "- **Delete (❌)**: remove an event\n"
             "- **Clear All**: reset the simulation\n\n"
             "Events are credited sequentially in the order added. "
             "To test sensitivity, add events in different orders."),

            ("Step 11 — Settings",
             "Navigate to **⚙️ Settings**.\n\n"
             "Set the **default Traffic Lift (%)** for each campaign shape per brand. "
             "These pre-populate the slider in the Events tab when you select a shape.\n\n"
             "- **Language**: switch between English and Italian\n"
             "- **Campaign defaults table**: rows = brands + global, columns = shapes\n"
             "- **Apply Global to All Brands**: copies the global row to every brand\n"
             "- **Save Settings**: persists to `data/settings.json`"),

            ("Step 12 — Export",
             "In the sidebar, click **Download Strategy Report**.\n\n"
             "Generates an Excel file with:\n"
             "- Full-year daily projections (Baseline + Simulation for all metrics)\n"
             "- Summary KPI table\n"
             "- Active event log\n\n"
             "Share with stakeholders who don't have direct access to the app."),
        ]

        for title, body in steps:
            with st.expander(title):
                st.markdown(body)

        st.markdown("---")
        st.markdown("### Tips & Best Practices")
        tips = [
            ("Start with one brand",
             "Multi-brand DNA blending is powerful but harder to interpret. "
             "Master the model on a single brand first, then explore combined views."),
            ("Use Monthly for strategy, Daily for execution",
             "Monthly is faster, less noisy, and better for quarterly planning. "
             "Switch to Daily when scheduling specific campaign dates."),
            ("Calibrate on a quiet period",
             "A 2–4 week period with no major campaigns and complete data gives "
             "the most reliable calibration. January is often ideal for seasonal businesses."),
            ("Test campaign timing by shifting events",
             "Use the Shift (↔) button to move a campaign window and instantly see "
             "if an earlier or later date gives better goal coverage."),
            ("Build a de-shock library before planning season",
             "Before annual planning, extract shock signatures for your key recurring events "
             "(Black Friday, seasonal sale, product launch). "
             "Then re-inject them to model the next year."),
            ("Review DNA weights each session",
             "If one year dominates (> 80%), check whether it truly was most similar "
             "to your current period. If not, your trial dates or values may need adjustment."),
        ]
        for title, desc in tips:
            with st.expander(f"💡 {title}"):
                st.markdown(desc)
