import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PROFILES_PATH   = os.path.join(DATA_DIR, "individual_brand_profiles_granular.csv")
YEARLY_KPI_PATH = os.path.join(DATA_DIR, "yearly_kpis.csv")
DATASET_PATH    = os.path.join(DATA_DIR, "super_dataset.csv")

EVENT_MAPPING = {
    "Push/DEM":       "Front-Loaded",
    "High discount":  "Linear Fade",
    "Product Launch": "Delayed Peak",
    "Field Campaign": "Step",
}
