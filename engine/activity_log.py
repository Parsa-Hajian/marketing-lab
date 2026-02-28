"""Persistent activity log: records who did what and when."""
import os
import json
import pandas as pd
from datetime import datetime

from config import LOG_PATH, PROFILES_PATH, DATASET_PATH

_COLUMNS = ["Timestamp", "Name", "Username", "Action", "Details"]


# ── Login snapshot ────────────────────────────────────────────────────────────

def _data_state_snapshot() -> str:
    """
    Build a comprehensive JSON snapshot of the data state at login:
      - Per-brand: rows, total clicks, quantity, sales
      - Past modification history per brand from the activity log
    Stored in the Details column of the Login row.
    """
    snapshot = {}

    # ── 1. Per-brand aggregates from dataset ──────────────────────────────────
    brands_data = {}
    try:
        ds = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
        ds["brand"] = ds["brand"].str.strip().str.lower()
        for brand, grp in ds.groupby("brand"):
            brands_data[brand] = {
                "rows":         int(len(grp)),
                "total_clicks": int(grp["clicks"].sum()),
                "total_qty":    int(grp["quantity"].sum()),
                "total_sales":  round(float(grp["sales"].sum()), 2),
                "date_min":     grp["Date"].min().strftime("%Y-%m-%d"),
                "date_max":     grp["Date"].max().strftime("%Y-%m-%d"),
            }
    except Exception as e:
        brands_data["_error"] = str(e)

    snapshot["brands"] = brands_data
    snapshot["brand_count"] = len(brands_data)

    # ── 2. Past modification history per brand ────────────────────────────────
    brand_history: dict = {}
    _MOD_ACTIONS = {
        "Add Brand", "Update Brand", "Replace Brand",
        "Campaign Injected", "DNA Swap", "DNA Drag",
        "De-Shock Extracted", "De-Shock Re-Injected",
        "Event Deleted", "Event Shifted", "Event Log Cleared",
        "Brand Forge: Brand Saved",
        "Settings: Save", "Settings: Apply Global Defaults",
    }
    if os.path.exists(LOG_PATH):
        try:
            log_df = pd.read_csv(LOG_PATH)
            mod_df = log_df[log_df["Action"].isin(_MOD_ACTIONS)].copy()
            for _, row in mod_df.iterrows():
                details_str = str(row.get("Details", ""))
                # Extract brand name from Details field (format "Brand: xxx | ...")
                brand_key = None
                for part in details_str.split("|"):
                    part = part.strip()
                    if part.lower().startswith("brand:"):
                        brand_key = part.split(":", 1)[1].strip().lower()
                        break
                if brand_key is None:
                    brand_key = "__global__"
                entry = {
                    "timestamp": str(row.get("Timestamp", "")),
                    "by":        str(row.get("Name", "")),
                    "action":    str(row.get("Action", "")),
                    "details":   details_str,
                }
                brand_history.setdefault(brand_key, []).append(entry)
        except Exception as e:
            brand_history["_error"] = [{"details": str(e)}]

    snapshot["modification_history"] = brand_history
    snapshot["total_past_events"] = sum(
        len(v) for v in brand_history.values() if isinstance(v, list))

    return json.dumps(snapshot, ensure_ascii=False)


def log_action(name: str, username: str, action: str, details: str = "") -> None:
    """Append one row to the activity log CSV (creates file if absent)."""
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Name":      name.strip(),
        "Username":  username.strip(),
        "Action":    action,
        "Details":   details,
    }
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=_COLUMNS)
    df.to_csv(LOG_PATH, index=False)


def log_login(name: str, username: str) -> None:
    """Log a login event with a full data-state snapshot."""
    snapshot_json = _data_state_snapshot()
    log_action(
        name=name,
        username=username,
        action="Login",
        details=snapshot_json,
    )


def load_log() -> pd.DataFrame:
    """Return the full activity log, newest first."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(LOG_PATH)
    return df.iloc[::-1].reset_index(drop=True)
