"""Persistent activity log: records who did what and when."""
import os
import pandas as pd
from datetime import datetime

from config import LOG_PATH, PROFILES_PATH, DATASET_PATH

_COLUMNS = ["Timestamp", "Name", "Username", "Action", "Details"]


def _data_state_snapshot() -> str:
    """
    Capture a compact snapshot of the current data state:
    brand count, total rows, and dataset date range.
    Appended to Login log entries so the record reflects
    what the data looked like when the session started.
    """
    parts = []
    try:
        profiles = pd.read_csv(PROFILES_PATH)
        n_brands = profiles["brand"].nunique()
        parts.append(f"Brands: {n_brands}")
    except Exception:
        parts.append("Brands: n/a")

    try:
        ds = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
        parts.append(f"Dataset rows: {len(ds):,}")
        if not ds.empty:
            dmin = ds["Date"].min().strftime("%Y-%m-%d")
            dmax = ds["Date"].max().strftime("%Y-%m-%d")
            parts.append(f"Date range: {dmin} → {dmax}")
    except Exception:
        parts.append("Dataset: n/a")

    return " | ".join(parts)


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
    """Log a login event with a snapshot of the current data state."""
    snapshot = _data_state_snapshot()
    log_action(
        name=name,
        username=username,
        action="Login",
        details=f"Data state on login — {snapshot}",
    )


def load_log() -> pd.DataFrame:
    """Return the full activity log, newest first. Empty DataFrame if no log yet."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(LOG_PATH)
    return df.iloc[::-1].reset_index(drop=True)
