"""User Log page — shows all recorded activity with rich snapshot rendering."""
import json
import streamlit as st
from engine.activity_log import load_log, delete_log_entries


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


def _border_for(action: str) -> str:
    if action == "Login":                              return "#10B981"
    if action == "Sign Out":                           return "#6B7280"
    if action in ("Add Brand", "Brand Forge: Brand Saved"): return "#3B82F6"
    if action in ("Update Brand", "Replace Brand"):    return "#F59E0B"
    if action == "Campaign Injected":                  return "#F47920"
    if action in ("DNA Swap", "DNA Drag"):             return "#8B5CF6"
    if action in ("De-Shock Extracted", "De-Shock Re-Injected"): return "#EF4444"
    if action in ("Event Deleted", "Event Log Cleared"): return "#DC2626"
    if action == "Event Shifted":                      return "#0EA5E9"
    if action.startswith("Settings") or action.startswith("Goal Tracker"): return "#6366F1"
    if action == "Brand Forge: Preview Generated":     return "#A78BFA"
    if action == "Page Navigation":                    return "#CBD5E1"
    if action == "Resolution Changed":                 return "#94A3B8"
    if action == "Brand Selection Changed":            return "#64748B"
    return "#E0E0E0"


def render_user_log():
    df = load_log()

    if df.empty:
        st.info("No activity recorded yet.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events",    f"{len(df):,}")
    col2.metric("Unique Users",    f"{df['Name'].nunique():,}")
    _mod_actions = {
        "Add Brand", "Update Brand", "Replace Brand",
        "Campaign Injected", "DNA Swap", "DNA Drag",
        "De-Shock Extracted", "De-Shock Re-Injected",
        "Event Deleted", "Event Shifted", "Event Log Cleared",
        "Brand Forge: Brand Saved", "Settings: Save", "Settings: Apply Global Defaults",
    }
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

    n_shown = len(filtered)
    st.markdown(
        f"<div style='margin-bottom:8px;font-size:0.8rem;color:#999'>"
        f"{n_shown:,} events</div>",
        unsafe_allow_html=True,
    )

    # ── Delete controls ────────────────────────────────────────────────────────
    with st.expander("🗑️ Delete entries", expanded=False):
        sel_all = st.checkbox("Select all visible entries", key="log_sel_all")

        # Keep track of which display-row indices are checked
        checked: list[int] = []
        for pos, (disp_idx, row) in enumerate(filtered.iterrows()):
            action = row["Action"]
            ts     = row["Timestamp"]
            name   = row["Name"]
            label  = f"{ts}  ·  {action}  ·  {name}"
            ticked = st.checkbox(label, value=sel_all, key=f"log_chk_{pos}_{disp_idx}")
            if ticked:
                checked.append(int(disp_idx))

        if checked:
            st.warning(f"{len(checked)} entr{'y' if len(checked)==1 else 'ies'} selected.")
            if st.button("🗑️ Delete selected", type="primary", key="log_delete_btn"):
                delete_log_entries(checked)
                st.success(f"Deleted {len(checked)} entr{'y' if len(checked)==1 else 'ies'}.")
                st.rerun()
        else:
            st.caption("Tick entries above then press Delete.")

    st.markdown("---")

    # ── Render each event ─────────────────────────────────────────────────────
    for _, row in filtered.iterrows():
        action = row["Action"]
        ts     = row["Timestamp"]
        name   = row["Name"]
        uname  = row["Username"]
        det    = str(row.get("Details", ""))

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
