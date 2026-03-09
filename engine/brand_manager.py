"""Brand data management: add / update brands in the CSV data files."""
import numpy as np
import pandas as pd


# ── Validation ───────────────────────────────────────────────────────────────

def validate_upload(df: pd.DataFrame) -> tuple[bool, str]:
    """Return (is_valid, error_message)."""
    required = {"Date", "Clicks", "Quantity", "Sales"}
    missing  = required - set(df.columns)
    if missing:
        return False, f"Missing columns: {', '.join(sorted(missing))}"
    try:
        pd.to_datetime(df["Date"])
    except Exception:
        return False, "Could not parse 'Date' column — use YYYY-MM-DD format."
    for col in ("Clicks", "Quantity", "Sales"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric."
    if (df[["Clicks", "Quantity", "Sales"]] < 0).any().any():
        return False, "Clicks, Quantity, and Sales must be non-negative."
    return True, "OK"


# ── DNA index computation ────────────────────────────────────────────────────

def _agg_by_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Aggregate a raw DataFrame (Date, Clicks, Quantity, Sales) by level."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    if level == "Monthly":
        d["TimeIdx"] = d["Date"].dt.month
    elif level == "Weekly":
        d["TimeIdx"] = d["Date"].dt.isocalendar().week.astype(int)
    else:  # Daily
        d["TimeIdx"] = d["Date"].dt.dayofyear
    return (
        d.groupby("TimeIdx")
         .agg(clicks=("Clicks", "sum"), quantity=("Quantity", "sum"), sales=("Sales", "sum"))
         .reset_index()
    )


def _indices_from_agg(agg: pd.DataFrame) -> pd.DataFrame:
    """Compute normalised DNA indices from an aggregated DataFrame."""
    g = agg.copy()
    g["cr"]  = g["quantity"] / g["clicks"].replace(0, np.nan)
    g["aov"] = g["sales"]   / g["quantity"].replace(0, np.nan)

    avg_c   = g["clicks"].mean()
    avg_cr  = g["cr"].mean()
    avg_aov = g["aov"].mean()

    g["idx_clicks"] = (g["clicks"] / avg_c  ).fillna(1.0) if avg_c   > 0 else 1.0
    g["idx_cr"]     = (g["cr"]     / avg_cr ).fillna(1.0) if avg_cr  > 0 else 1.0
    g["idx_aov"]    = (g["aov"]    / avg_aov).fillna(1.0) if avg_aov > 0 else 1.0

    return g[["TimeIdx", "idx_clicks", "idx_cr", "idx_aov"]]


def build_profiles(brand_name: str, raw_df: pd.DataFrame, levels: list) -> pd.DataFrame:
    """
    Build a profiles DataFrame matching `individual_brand_profiles_granular.csv`
    from raw daily data with columns [Date, Clicks, Quantity, Sales].
    """
    raw = raw_df.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Year"] = raw["Date"].dt.year
    brand_key   = brand_name.strip().lower()

    rows = []
    for level in levels:
        # Per-year slices
        for year, yr_df in raw.groupby("Year"):
            agg = _agg_by_level(yr_df, level)
            idx = _indices_from_agg(agg)
            idx["brand"] = brand_key
            idx["Year"]  = str(year)
            idx["Level"] = level
            rows.append(idx)

        # Overall (all years combined)
        agg_all = _agg_by_level(raw, level)
        idx_all = _indices_from_agg(agg_all)
        idx_all["brand"] = brand_key
        idx_all["Year"]  = "Overall"
        idx_all["Level"] = level
        rows.append(idx_all)

    return pd.concat(rows, ignore_index=True)[
        ["brand", "Year", "Level", "TimeIdx", "idx_clicks", "idx_cr", "idx_aov"]
    ]


# ── File operations ──────────────────────────────────────────────────────────

def load_raw_for_brand(brand_key: str, dataset_path: str) -> pd.DataFrame:
    """Return the existing raw rows for a brand in upload format (Clicks/Quantity/Sales)."""
    ds = pd.read_csv(dataset_path)
    ds["brand"] = ds["brand"].str.strip().str.lower()
    ds["Date"]  = pd.to_datetime(ds["Date"])
    existing = ds[ds["brand"] == brand_key].copy()
    return existing.rename(
        columns={"clicks": "Clicks", "quantity": "Quantity", "sales": "Sales"}
    )[["Date", "Clicks", "Quantity", "Sales"]]


# ── File save ────────────────────────────────────────────────────────────────

def save_brand(
    brand_name: str,
    raw_df: pd.DataFrame,
    levels: list,
    profiles_path: str,
    dataset_path: str,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """
    Add or update a brand in both data files.
    Returns (success, message).
    """
    brand_key = brand_name.strip().lower()

    # Load existing
    profiles = pd.read_csv(profiles_path)
    profiles["brand"] = profiles["brand"].str.strip().str.lower()
    dataset  = pd.read_csv(dataset_path)
    dataset["brand"] = dataset["brand"].str.strip().str.lower()
    dataset["Date"]  = pd.to_datetime(dataset["Date"])

    brand_exists = brand_key in profiles["brand"].unique()

    if brand_exists and not overwrite:
        return False, (
            f"Brand '{brand_key}' already exists. "
            "Use Update Brand to modify it."
        )

    # Drop existing rows for this brand if overwriting
    if brand_exists:
        profiles = profiles[profiles["brand"] != brand_key]
        dataset  = dataset[dataset["brand"] != brand_key]

    # Build new profiles
    new_profiles = build_profiles(brand_key, raw_df, levels)
    profiles = pd.concat([profiles, new_profiles], ignore_index=True)

    # Build new raw dataset rows
    raw_save = raw_df.copy()
    raw_save["Date"]  = pd.to_datetime(raw_save["Date"])
    raw_save["brand"] = brand_key
    raw_save = raw_save.rename(
        columns={"Clicks": "clicks", "Quantity": "quantity", "Sales": "sales"}
    )[["Date", "brand", "clicks", "quantity", "sales"]]
    dataset = pd.concat([dataset, raw_save], ignore_index=True)

    # Save — enforce consistent YYYY-MM-DD date format
    profiles.to_csv(profiles_path, index=False)
    dataset.to_csv(dataset_path, index=False, date_format="%Y-%m-%d")

    action = "updated" if brand_exists else "added"
    return True, f"Brand '{brand_key}' {action} — {len(raw_df):,} rows processed."


def save_brand_append(
    brand_name: str,
    new_raw_df: pd.DataFrame,
    levels: list,
    profiles_path: str,
    dataset_path: str,
) -> tuple[bool, str]:
    """
    Merge new records into an existing brand.
    New data takes priority for any overlapping dates.
    Returns (success, message).
    """
    brand_key = brand_name.strip().lower()
    existing  = load_raw_for_brand(brand_key, dataset_path)

    if existing.empty:
        return save_brand(brand_key, new_raw_df, levels, profiles_path, dataset_path,
                          overwrite=False)

    new = new_raw_df.copy()
    new["Date"] = pd.to_datetime(new["Date"])

    # Concatenate — put new last so drop_duplicates(keep="last") keeps new values
    combined = (
        pd.concat([existing, new], ignore_index=True)
          .assign(Date=lambda d: pd.to_datetime(d["Date"]))
          .sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
          .reset_index(drop=True)
    )

    overlap = int(existing["Date"].isin(new["Date"]).sum())
    added   = len(new) - overlap
    ok, msg = save_brand(brand_key, combined, levels, profiles_path, dataset_path,
                         overwrite=True)
    if ok:
        return True, (
            f"Brand '{brand_key}' updated — "
            f"{added:,} new dates added, {overlap:,} dates updated, "
            f"{len(combined):,} total rows."
        )
    return ok, msg
