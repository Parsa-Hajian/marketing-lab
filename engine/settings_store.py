"""App settings: language, per-brand campaign defaults."""
import json
import os
from config import SETTINGS_PATH, EVENT_MAPPING

_SHAPES = list(EVENT_MAPPING.keys())

_DEFAULT_CAMPAIGN_DEFAULTS = {shape: 25 for shape in _SHAPES}

_DEFAULTS = {
    "language": "en",
    "campaign_defaults": {
        "__all__": dict(_DEFAULT_CAMPAIGN_DEFAULTS),
    },
}


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return {
            "language": _DEFAULTS["language"],
            "campaign_defaults": {
                "__all__": dict(_DEFAULT_CAMPAIGN_DEFAULTS),
            },
        }
    with open(SETTINGS_PATH) as f:
        data = json.load(f)
    # Ensure __all__ always present
    if "campaign_defaults" not in data:
        data["campaign_defaults"] = {"__all__": dict(_DEFAULT_CAMPAIGN_DEFAULTS)}
    if "__all__" not in data["campaign_defaults"]:
        data["campaign_defaults"]["__all__"] = dict(_DEFAULT_CAMPAIGN_DEFAULTS)
    return data


def save_settings(settings: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def get_campaign_default(settings: dict, brand: str, shape: str) -> int:
    """
    Return default % lift for (brand, shape).
    Brand-level override takes priority over __all__.
    """
    cd = settings.get("campaign_defaults", {})
    brand_key = brand.strip().lower()
    if brand_key in cd and shape in cd[brand_key]:
        return int(cd[brand_key][shape])
    all_defaults = cd.get("__all__", {})
    return int(all_defaults.get(shape, 25))
