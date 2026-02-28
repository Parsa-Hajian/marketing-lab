"""Settings page — language, campaign default coefficients."""
import pandas as pd
import streamlit as st

from config import EVENT_MAPPING, PROFILES_PATH
from engine.settings_store import load_settings, save_settings
from engine.activity_log import log_action
from engine.i18n import t


_SHAPES = list(EVENT_MAPPING.keys())


def render_settings(lang: str = "en"):
    settings = load_settings()
    user_name = st.session_state.get("_user_name", "Unknown")
    username  = st.session_state.get("_username", "")

    # ── Language ──────────────────────────────────────────────────────────────
    st.markdown(f"### {t('settings_language', lang)}")
    lang_options = [t("settings_lang_en", lang), t("settings_lang_it", lang)]
    current_idx  = 1 if settings.get("language", "en") == "it" else 0
    lang_choice  = st.radio(
        t("settings_language", lang),
        lang_options,
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
    )
    new_lang = "it" if lang_choice == t("settings_lang_it", lang) else "en"

    st.markdown("---")

    # ── Campaign defaults ─────────────────────────────────────────────────────
    st.markdown(f"### {t('settings_camp_title', lang)}")
    st.caption(t("settings_camp_desc", lang))

    # Load brand list
    try:
        profiles = pd.read_csv(PROFILES_PATH)
        all_brands = sorted(profiles["brand"].str.strip().str.lower().unique())
    except Exception:
        all_brands = []

    cd = settings.get("campaign_defaults", {"__all__": {s: 25 for s in _SHAPES}})

    # Build editable dataframe: rows = global + each brand, cols = shapes
    rows = []
    for b in ["__all__"] + all_brands:
        label = t("settings_global_row", lang) if b == "__all__" else b.title()
        row   = {"Brand": label}
        for shape in _SHAPES:
            row[shape] = int(cd.get(b, cd.get("__all__", {})).get(shape, 25))
        rows.append(row)

    df_defaults = pd.DataFrame(rows)

    # Column config: each shape column is a number 0–300
    col_cfg = {"Brand": st.column_config.TextColumn("Brand", disabled=True)}
    for shape in _SHAPES:
        col_cfg[shape] = st.column_config.NumberColumn(
            shape,
            min_value=-100,
            max_value=300,
            step=5,
            format="%d %%",
            help=f"Default traffic lift (%) for {shape}",
        )

    edited = st.data_editor(
        df_defaults,
        column_config=col_cfg,
        use_container_width=True,
        hide_index=True,
        key="campaign_defaults_editor",
    )

    st.markdown("---")

    # Apply Global to All button
    col_apply, col_save, _ = st.columns([1.5, 1.5, 3])
    with col_apply:
        if st.button(t("settings_apply_all", lang), use_container_width=True):
            # Read global row from edited df and apply to all brand rows
            global_row = edited[edited["Brand"] == t("settings_global_row", lang)]
            if not global_row.empty:
                global_vals = {s: int(global_row.iloc[0][s]) for s in _SHAPES}
                for i in range(len(edited)):
                    for s in _SHAPES:
                        edited.at[i, s] = global_vals[s]
                # Persist
                new_cd = {"__all__": global_vals}
                for b in all_brands:
                    new_cd[b] = dict(global_vals)
                settings["campaign_defaults"] = new_cd
                settings["language"] = new_lang
                save_settings(settings)
                log_action(
                    name=user_name, username=username,
                    action="Settings: Apply Global Defaults",
                    details=f"Global defaults applied to all {len(all_brands)} brands: "
                            + ", ".join(f"{s}={global_vals[s]}%" for s in _SHAPES),
                )
                st.success(t("settings_applied", lang))
                st.rerun()

    with col_save:
        if st.button(t("settings_save", lang), type="primary", use_container_width=True):
            # Parse edited df back into settings structure
            new_cd = {}
            for _, row in edited.iterrows():
                brand_label = row["Brand"]
                if brand_label == t("settings_global_row", lang):
                    key = "__all__"
                else:
                    key = brand_label.lower()
                new_cd[key] = {s: int(row[s]) for s in _SHAPES}

            old_lang = settings.get("language", "en")
            settings["campaign_defaults"] = new_cd
            settings["language"]          = new_lang
            save_settings(settings)

            # Build change summary for log
            changes = []
            if old_lang != new_lang:
                changes.append(f"Language: {old_lang} → {new_lang}")
            for key, vals in new_cd.items():
                old_vals = cd.get(key, {})
                diffs = [f"{s}: {old_vals.get(s,'?')} → {vals[s]}%"
                         for s in _SHAPES if old_vals.get(s) != vals[s]]
                if diffs:
                    label = "global" if key == "__all__" else key
                    changes.append(f"{label}: " + ", ".join(diffs))

            log_action(
                name=user_name, username=username,
                action="Settings: Save",
                details="; ".join(changes) if changes else "No changes",
            )
            st.success(t("settings_saved", lang))
            st.rerun()

    # ── Current defaults preview ───────────────────────────────────────────────
    with st.expander("Current effective defaults per brand"):
        preview_rows = []
        for b in all_brands:
            row = {"Brand": b.title()}
            for s in _SHAPES:
                val = int(cd.get(b, cd.get("__all__", {})).get(s, 25))
                row[s] = f"{val}%"
            preview_rows.append(row)
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No brands loaded yet.")
