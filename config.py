import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH  = os.path.join(ASSETS_DIR, "logo.png")

PROFILES_PATH   = os.path.join(DATA_DIR, "individual_brand_profiles_granular.csv")
YEARLY_KPI_PATH = os.path.join(DATA_DIR, "yearly_kpis.csv")
DATASET_PATH    = os.path.join(DATA_DIR, "super_dataset.csv")
LOG_PATH        = os.path.join(DATA_DIR, "activity_log.csv")

EVENT_MAPPING = {
    "Push/DEM":       "Front-Loaded",
    "High discount":  "Linear Fade",
    "Product Launch": "Delayed Peak",
    "Field Campaign": "Step",
}
