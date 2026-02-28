"""App settings: language, per-brand campaign defaults."""
import json
import os

# Derive paths relative to this file's location (engine/ → repo root → data/)
_ENGINE_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_ENGINE_DIR)
_SETTINGS_PATH = os.path.join(_ROOT_DIR, "data", "settings.json")

_SHAPES = ["Push/DEM", "High discount", "Product Launch", "Field Campaign"]

_DEFAULT_CAMPAIGN_DEFAULTS = {shape: 25 for shape in _SHAPES}


def load_settings() -> dict:
    if not os.path.exists(_SETTINGS_PATH):
        return {
            "language": "en",
            "campaign_defaults": {
                "__all__": dict(_DEFAULT_CAMPAIGN_DEFAULTS),
            },
        }
    with open(_SETTINGS_PATH) as f:
        data = json.load(f)
    if "campaign_defaults" not in data:
        data["campaign_defaults"] = {"__all__": dict(_DEFAULT_CAMPAIGN_DEFAULTS)}
    if "__all__" not in data["campaign_defaults"]:
        data["campaign_defaults"]["__all__"] = dict(_DEFAULT_CAMPAIGN_DEFAULTS)
    return data


def save_settings(settings: dict) -> None:
    os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
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
