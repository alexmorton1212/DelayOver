
import os
import json
import re
import pandas as pd

# --------------------------------------------------------------------------------------------------------
# DIRECTORIES
# --------------------------------------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')
STATS_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'stats')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

PARQUET_FILE = os.path.join(PROCESSED_DATA_DIR, 'ml_dataset.parquet')


# --------------------------------------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------------------------------------

df = pd.read_parquet(PARQUET_FILE, engine='pyarrow')


# --------------------------------------------------------------------------------------------------------
# FIND MOST RECENT PARQUET FILE
# --------------------------------------------------------------------------------------------------------

parquet_files = [
    f for f in os.listdir(RAW_DATA_DIR)
    if re.match(r'flight_data_\d{4}_\d+\.parquet', f)
]

if not parquet_files:
    raise FileNotFoundError(f"No flight_data_YYYY_M.parquet files found in {RAW_DATA_DIR}")

def extract_date(f):
    m = re.match(r'flight_data_(\d{4})_(\d+)\.parquet', f)
    return int(m.group(1)), int(m.group(2))

parquet_files.sort(key=extract_date)
latest_file = parquet_files[-1]

print(f"Using parquet file: {os.path.join(RAW_DATA_DIR, latest_file)}")


# --------------------------------------------------------------------------------------------------------
# COMPUTE STATS
# --------------------------------------------------------------------------------------------------------

CATEGORICAL_COLS = ["month", "dayofweek", "origin", "dest", "reporting_airline", "if_delay"]
NUMERIC_COLS = [] # --> No numeric columns are currently used in ML model

stats = {}

for col in df.columns:
    if col in CATEGORICAL_COLS:
        stats[col] = df[col].astype(str).value_counts(normalize=True).to_dict()
    elif col in NUMERIC_COLS:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "median": df[col].median(),
        }
    else:
        print(f"Column '{col}' not in dataframe")


# --------------------------------------------------------------------------------------------------------
# CONVERT TO PYTHON TYPES FOR JSON
# --------------------------------------------------------------------------------------------------------

def convert_to_python(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (pd.Series, list)):
        return [convert_to_python(v) for v in obj]
    elif isinstance(obj, (pd._libs.missing.NAType, type(None))):
        return None
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return obj

stats = convert_to_python(stats)


# --------------------------------------------------------------------------------------------------------
# COMPUTE STATS
# --------------------------------------------------------------------------------------------------------

m = re.match(r'flight_data_(\d{4}_\d+)\.parquet', latest_file)
file_date = m.group(1)
stats_file_path = os.path.join(STATS_DIR, f"stats_{file_date}.json")

with open(stats_file_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Stats saved to {stats_file_path}")


# --------------------------------------------------------------------------------------------------------
# CREATE/UPDATE ACTIVE STATS FILE
# --------------------------------------------------------------------------------------------------------

ACTIVE_FILE = os.path.join(STATS_DIR, "active_stats.json")

# --> If this is the first stats file, set it as active
if not os.path.exists(ACTIVE_FILE):
    with open(ACTIVE_FILE, "w") as f:
        json.dump({"active_stats_file": os.path.basename(stats_file_path)}, f)
    print(f"Active stats initialized to {os.path.basename(stats_file_path)}")
else:
    print(f"Active stats already exists: {ACTIVE_FILE}")


# --------------------------------------------------------------------------------------------------------
# DELETE OLD RAW FILES OF THE SAME MONTH
# --------------------------------------------------------------------------------------------------------

m = re.match(r'flight_data_(\d{4})_(\d+)\.parquet', latest_file)
latest_year = int(m.group(1))
latest_month = int(m.group(2))

for f in parquet_files:
    if f == latest_file:
        continue 
    try:
        m_old = re.match(r'flight_data_(\d{4})_(\d+)\.parquet', f)
        old_year = int(m_old.group(1))
        old_month = int(m_old.group(2))
        # --> Deletes if same month from an older year
        # --> Only want to keep the most recent month of each month type
        if old_month == latest_month and old_year < latest_year:
            os.remove(os.path.join(RAW_DATA_DIR, f))
            print(f"Deleted old raw file: {f}")
    except Exception as e:
        print(f"Skipping {f}, could not parse date: {e}")