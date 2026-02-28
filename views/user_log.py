"""User Log page — shows all recorded activity."""
import streamlit as st
from engine.activity_log import load_log

_ACTION_COLORS = {
    "Login":          "#4CAF50",
    "Add Brand":      "#2196F3",
    "Update Brand":   "#FF9800",
    "Replace Brand":  "#FF5722",
}


def render_user_log():
    df = load_log()

    if df.empty:
        st.info("No activity recorded yet.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events",    f"{len(df):,}")
    col2.metric("Unique Users",    f"{df['Name'].nunique():,}")
    col3.metric("Brand Modifications",
                f"{df[df['Action'].isin(['Add Brand','Update Brand','Replace Brand'])].shape[0]:,}")

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

    st.markdown(f"<div style='margin-bottom:8px;font-size:0.8rem;color:#999'>"
                f"{len(filtered):,} events</div>", unsafe_allow_html=True)

    # ── Log table ──────────────────────────────────────────────────────────────
    st.dataframe(
        filtered[["Timestamp", "Name", "Username", "Action", "Details"]],
        use_container_width=True,
        hide_index=True,
    )

    # ── Download ───────────────────────────────────────────────────────────────
    st.download_button(
        "Download Log CSV",
        data=filtered.to_csv(index=False).encode(),
        file_name="activity_log.csv",
        mime="text/csv",
    )
