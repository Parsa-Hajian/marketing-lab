"""User Log page — shows all recorded activity with rich snapshot rendering."""
import json
import streamlit as st
from engine.activity_log import load_log


def _render_login_snapshot(details_str: str):
    """Expand and render a JSON login snapshot in a readable format."""
    try:
        snap = json.loads(details_str)
    except Exception:
        st.text(details_str)
        return

    brands = snap.get("brands", {})
    history = snap.get("modification_history", {})

    if brands:
        st.markdown(f"**{snap.get('brand_count', len(brands))} brand(s) in dataset:**")
        rows_html = ""
        for brand, stats in brands.items():
            rows_html += (
                f"<tr>"
                f"<td style='padding:4px 12px 4px 0;font-weight:600'>{brand.title()}</td>"
                f"<td style='padding:4px 12px'>{stats.get('rows', 0):,} rows</td>"
                f"<td style='padding:4px 12px'>{stats.get('total_clicks', 0):,} clicks</td>"
                f"<td style='padding:4px 12px'>{stats.get('total_qty', 0):,} qty</td>"
                f"<td style='padding:4px 12px'>€{stats.get('total_sales', 0):,.0f} sales</td>"
                f"<td style='padding:4px 12px;color:#888;font-size:0.8rem'>"
                f"{stats.get('date_min','?')} → {stats.get('date_max','?')}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='font-size:0.83rem;border-collapse:collapse'>"
            f"<thead><tr>"
            f"<th style='padding:4px 12px 4px 0;color:#888;font-weight:600'>Brand</th>"
            f"<th style='padding:4px 12px;color:#888;font-weight:600'>Rows</th>"
            f"<th style='padding:4px 12px;color:#888;font-weight:600'>Clicks</th>"
            f"<th style='padding:4px 12px;color:#888;font-weight:600'>Qty</th>"
            f"<th style='padding:4px 12px;color:#888;font-weight:600'>Sales</th>"
            f"<th style='padding:4px 12px;color:#888;font-weight:600'>Date Range</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

    total_past = snap.get("total_past_events", 0)
    if total_past and history:
        st.markdown(f"**{total_past} past modification(s) on record:**")
        for brand_key, events in history.items():
            if not isinstance(events, list):
                continue
            label = brand_key.title() if brand_key != "__global__" else "Global"
            st.markdown(f"*{label}* — {len(events)} event(s)")
            for ev in events:
                ts  = ev.get("timestamp", "")
                by  = ev.get("by", "")
                act = ev.get("action", "")
                det = ev.get("details", "")
                st.markdown(
                    f"<div style='margin-left:16px;font-size:0.8rem;color:#555;"
                    f"padding:2px 0'>"
                    f"<span style='color:#888'>{ts}</span> — "
                    f"<b>{act}</b> by {by}<br>"
                    f"<span style='color:#AAAAAA'>{det}</span></div>",
                    unsafe_allow_html=True,
                )
    elif not total_past:
        st.caption("No prior modifications recorded at login time.")


def render_user_log():
    df = load_log()

    if df.empty:
        st.info("No activity recorded yet.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events",    f"{len(df):,}")
    col2.metric("Unique Users",    f"{df['Name'].nunique():,}")
    _mod_actions = {"Add Brand", "Update Brand", "Replace Brand",
                    "Campaign Injected", "DNA Swap", "DNA Drag",
                    "De-Shock Extracted", "De-Shock Re-Injected"}
    col3.metric("Modifications",
                f"{df[df['Action'].isin(_mod_actions)].shape[0]:,}")

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────────
    fc1, fc2 = st.columns([2, 2])
    with fc1:
        users = ["All"] + sorted(df["Name"].unique().tolist())
        sel_user = st.selectbox("Filter by user", users)
    with fc2:
        actions = ["All"] + sorted(df["Action"].unique().tolist())
        sel_action = st.selectbox("Filter by action", actions)

    filtered = df.copy()
    if sel_user   != "All": filtered = filtered[filtered["Name"]   == sel_user]
    if sel_action != "All": filtered = filtered[filtered["Action"] == sel_action]

    st.markdown(
        f"<div style='margin-bottom:8px;font-size:0.8rem;color:#999'>"
        f"{len(filtered):,} events</div>",
        unsafe_allow_html=True,
    )

    # ── Render each event ─────────────────────────────────────────────────────
    for _, row in filtered.iterrows():
        action = row["Action"]
        ts     = row["Timestamp"]
        name   = row["Name"]
        uname  = row["Username"]
        det    = str(row.get("Details", ""))

        border = "#E0E0E0"
        if action == "Login":
            border = "#10B981"
        elif action in ("Add Brand",):
            border = "#3B82F6"
        elif action in ("Update Brand", "Replace Brand"):
            border = "#F59E0B"
        elif action in ("Campaign Injected",):
            border = "#F47920"
        elif action in ("DNA Swap", "DNA Drag"):
            border = "#8B5CF6"
        elif action in ("De-Shock Extracted", "De-Shock Re-Injected"):
            border = "#EF4444"
        elif action.startswith("Settings"):
            border = "#6366F1"

        with st.expander(f"{ts}  ·  **{action}**  ·  {name} ({uname})"):
            if action == "Login":
                _render_login_snapshot(det)
            else:
                st.markdown(
                    f"<div style='font-size:0.85rem;color:#333;line-height:1.6'>"
                    + "<br>".join(det.split(" | ")) +
                    "</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    # ── Download ───────────────────────────────────────────────────────────────
    st.download_button(
        "Download Log CSV",
        data=filtered.to_csv(index=False).encode(),
        file_name="activity_log.csv",
        mime="text/csv",
    )
