"""Minimal EN / IT translation layer.  Usage:  t("key") or t("key", lang="it")."""

_T = {
    "en": {
        # Navigation
        "nav_dashboard":       "Dashboard",
        "nav_sim_lab":         "Simulation Lab",
        "nav_add_brand":       "Add Brand",
        "nav_update_brand":    "Update Brand",
        "nav_settings":        "Settings",
        "nav_user_log":        "User Log",
        # Auth
        "full_name":           "Full Name",
        "username":            "Username",
        "password":            "Password",
        "sign_in":             "Sign In",
        "sign_out":            "Sign Out",
        "sign_in_subtitle":    "Sign in to continue",
        "name_placeholder":    "e.g. Parsa Hajiannejad",
        "invalid_credentials": "Invalid username or password.",
        "name_required":       "Please enter your full name.",
        # Page subtitles
        "sub_dashboard":       "Overview of projections, DNA, and goal tracking",
        "sub_sim_lab":         "Inject events, shocks, and campaign simulations",
        "sub_add_brand":       "Upload historical data to create a new brand profile",
        "sub_update_brand":    "Replace or extend data for an existing brand",
        "sub_settings":        "Language, campaign defaults, and app configuration",
        "sub_user_log":        "Activity log — modifications and people who made them",
        # Settings page
        "settings_language":   "Language",
        "settings_lang_en":    "English",
        "settings_lang_it":    "Italian",
        "settings_camp_title": "Default Campaign Impact Coefficients",
        "settings_camp_desc":  (
            "Set the default traffic lift (%) for each campaign shape per brand. "
            "Brand-level values override the global defaults."
        ),
        "settings_global_row": "All Brands (Global Default)",
        "settings_apply_all":  "Apply Global Defaults to All Brands",
        "settings_save":       "Save Settings",
        "settings_saved":      "Settings saved.",
        "settings_applied":    "Global defaults applied to all brands.",
        # Lab
        "lab_add_campaign":    "Add Time-Bound Campaign",
        "lab_traffic_lift":    "Traffic Lift (%)",
        "lab_campaign_shape":  "Campaign Shape",
        "lab_inject":          "Inject Campaign",
        "lab_swap":            "Swap Time Periods",
        "lab_exec_swap":       "Execute DNA Swap",
        "lab_default_hint":    "Default from settings",
        # Brand pages
        "brand_name":          "Brand Name",
        "upload_data":         "Upload Data",
        "update_mode":         "Update mode",
        "replace_all":         "Replace all data",
        "add_records":         "Add new records",
        "confirm_replace":     "Confirm Replace",
        "confirm_add":         "Confirm Add Records",
        # DNA chart labels
        "dna_clicks":          "Clicks",
        "dna_cr":              "CR",
        "dna_aov":             "AOV",
        "dna_pure":            "Pure",
        "dna_pretrial":        "Pre-Trial",
        "dna_work":            "Work / After",
    },
    "it": {
        # Navigation
        "nav_dashboard":       "Dashboard",
        "nav_sim_lab":         "Laboratorio Simulazione",
        "nav_add_brand":       "Aggiungi Brand",
        "nav_update_brand":    "Aggiorna Brand",
        "nav_settings":        "Impostazioni",
        "nav_user_log":        "Registro Attività",
        # Auth
        "full_name":           "Nome Completo",
        "username":            "Nome Utente",
        "password":            "Password",
        "sign_in":             "Accedi",
        "sign_out":            "Esci",
        "sign_in_subtitle":    "Accedi per continuare",
        "name_placeholder":    "es. Mario Rossi",
        "invalid_credentials": "Nome utente o password non validi.",
        "name_required":       "Inserisci il tuo nome completo.",
        # Page subtitles
        "sub_dashboard":       "Panoramica di proiezioni, DNA e monitoraggio obiettivi",
        "sub_sim_lab":         "Inietta eventi, shock e simulazioni di campagna",
        "sub_add_brand":       "Carica dati storici per creare un nuovo profilo brand",
        "sub_update_brand":    "Sostituisci o estendi i dati per un brand esistente",
        "sub_settings":        "Lingua, impostazioni predefinite campagne e configurazione",
        "sub_user_log":        "Registro attività — modifiche e chi le ha effettuate",
        # Settings page
        "settings_language":   "Lingua",
        "settings_lang_en":    "Inglese",
        "settings_lang_it":    "Italiano",
        "settings_camp_title": "Coefficienti di Impatto Campagna Predefiniti",
        "settings_camp_desc":  (
            "Imposta il lift di traffico predefinito (%) per ogni forma di campagna per brand. "
            "I valori a livello di brand hanno la precedenza sui predefiniti globali."
        ),
        "settings_global_row": "Tutti i Brand (Predefinito Globale)",
        "settings_apply_all":  "Applica Predefiniti Globali a Tutti i Brand",
        "settings_save":       "Salva Impostazioni",
        "settings_saved":      "Impostazioni salvate.",
        "settings_applied":    "Predefiniti globali applicati a tutti i brand.",
        # Lab
        "lab_add_campaign":    "Aggiungi Campagna Temporizzata",
        "lab_traffic_lift":    "Incremento Traffico (%)",
        "lab_campaign_shape":  "Forma Campagna",
        "lab_inject":          "Inietta Campagna",
        "lab_swap":            "Scambia Periodi Temporali",
        "lab_exec_swap":       "Esegui Scambio DNA",
        "lab_default_hint":    "Valore predefinito dalle impostazioni",
        # Brand pages
        "brand_name":          "Nome Brand",
        "upload_data":         "Carica Dati",
        "update_mode":         "Modalità aggiornamento",
        "replace_all":         "Sostituisci tutti i dati",
        "add_records":         "Aggiungi nuovi record",
        "confirm_replace":     "Conferma Sostituzione",
        "confirm_add":         "Conferma Aggiungi Record",
        # DNA chart labels
        "dna_clicks":          "Clic",
        "dna_cr":              "TC",
        "dna_aov":             "VOM",
        "dna_pure":            "Puro",
        "dna_pretrial":        "Pre-Trial",
        "dna_work":            "Operativo / Post",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Return translated string for key in given language, falling back to English."""
    return _T.get(lang, _T["en"]).get(key, _T["en"].get(key, key))
