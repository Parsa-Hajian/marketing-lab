"""Sidebar AI chatbot assistant powered by Anthropic."""
import streamlit as st


_SYSTEM_PROMPT = """You are an expert marketing analytics assistant embedded in the
Tech Strategy Lab — a Streamlit application used by e-commerce professionals to:
- Analyse seasonal purchase DNA across multiple brands
- Calibrate sales/clicks/quantity projections from a real trial window
- Simulate the impact of marketing campaigns, promotions, and demand shocks
- Attribute revenue uplift to specific events

Core concepts users may ask about:
- **DNA Index**: a seasonal multiplier (1.0 = average month). >1 = stronger season.
- **Trial Reality**: a short observed window used to calibrate the annual base rate.
- **Base Projection**: what happens with no campaigns.
- **Simulation Projection**: base + all logged events.
- **Shock**: a short-term demand spike (promo, launch).
- **De-Shock**: isolating and removing an artificial spike from historical data.
- **DNA Drag**: manually adjusting a period's seasonal weight.
- **Swap**: exchanging the seasonal pattern of two periods.
- **Goal Tracker**: reverse-engineering what campaign is needed to hit a revenue target.

Answer clearly and concisely. If the user asks about numbers, remind them to look at the
dashboard metrics. Be friendly and professional. Keep answers under 150 words unless
the user clearly needs a detailed explanation."""


def _get_client():
    """Return an Anthropic client if the API key is available."""
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return None
    if not key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def render_chatbot():
    """Render the AI assistant expander in the sidebar."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    client = _get_client()

    with st.sidebar.expander("🤖 AI Assistant", expanded=False):
        if client is None:
            st.caption(
                "AI Assistant requires an `ANTHROPIC_API_KEY` in Streamlit Secrets.  \n"
                "Add it via **App → Settings → Secrets** on Streamlit Cloud."
            )
            return

        # Display last 6 messages
        history = st.session_state.chat_history[-6:]
        for msg in history:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f"<div style='font-size:0.82rem;margin-bottom:6px'>"
                f"<b>{role_icon}</b> {msg['content']}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask anything about the app…",
                height=68,
                label_visibility="collapsed",
                placeholder="e.g. What does the DNA index mean?",
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button("Send ➤", use_container_width=True)
            with col2:
                clear = st.form_submit_button("Clear", use_container_width=True)

        if clear:
            st.session_state.chat_history = []
            st.rerun()

        if submitted and user_input.strip():
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input.strip()}
            )

            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    system=_SYSTEM_PROMPT,
                    messages=messages,
                )
                answer = resp.content[0].text
            except Exception as e:
                answer = f"⚠️ Error: {e}"

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
