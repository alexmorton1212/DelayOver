
### Whenever a new month of data is available, check statistics to see if
### ML model needs to be redone / thresholds need revising

import os
import json
import numpy as np
from scipy.spatial.distance import jensenshannon


# --------------------------------------------------------------------------------------------------------
# DIRECTORIES
# --------------------------------------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'stats')
DRIFT_LOG = os.path.join(STATS_DIR, "drift_log.json")
ACTIVE_FILE = os.path.join(STATS_DIR, "active_stats.json")


# --------------------------------------------------------------------------------------------------------
# FIND LATEST STATS FILE
# --------------------------------------------------------------------------------------------------------

stats_files = [f for f in os.listdir(STATS_DIR) if f.startswith("stats_") and f.endswith(".json")]

if not stats_files:
    print("No stats files found. Run compute_stats.py first.")
    exit()

def extract_date(file_name):
    parts = file_name.replace("stats_", "").replace(".json", "").split("_")
    return int(parts[0]), int(parts[1])  # year, month

stats_files.sort(key=extract_date)
latest_file = stats_files[-1]


# --------------------------------------------------------------------------------------------------------
# LOAD ACTIVE STATS
# --------------------------------------------------------------------------------------------------------

if os.path.exists(ACTIVE_FILE):
    with open(ACTIVE_FILE) as f:
        active_data = json.load(f)
    active_file = active_data["active_stats_file"]
else:
    active_file = stats_files[0]
    with open(ACTIVE_FILE, "w") as f:
        json.dump({"active_stats_file": active_file}, f)
    print(f"No active stats found. Setting {active_file} as active.")

if latest_file == active_file:
    print("Latest stats file is already active. No drift detection needed.")
    exit()

print(f"Comparing active stats ({active_file}) to latest stats ({latest_file})")


# --------------------------------------------------------------------------------------------------------
# LOAD STATS FILES
# --------------------------------------------------------------------------------------------------------

with open(os.path.join(STATS_DIR, latest_file)) as f:
    latest_stats = json.load(f)

with open(os.path.join(STATS_DIR, active_file)) as f:
    active_stats = json.load(f)


# --------------------------------------------------------------------------------------------------------
# DRIFT DETECTION
# --------------------------------------------------------------------------------------------------------

drift_detected = False
drift_summary = {}

for col in latest_stats.keys():

    drift_summary[col] = {}

    ### Numeric

    if isinstance(latest_stats[col], dict) and "mean" in latest_stats[col]:
        prev_mean = active_stats[col]["mean"]
        new_mean = latest_stats[col]["mean"]
        threshold = abs(prev_mean) * 0.1 if prev_mean != 0 else 0.1  # --> 10% drift threshold
        drift = abs(new_mean - prev_mean)
        drift_summary[col]["numeric_drift"] = drift
        drift_summary[col]["threshold"] = threshold
        drift_summary[col]["drift_flag"] = drift > threshold
        if drift > threshold:
            drift_detected = True
    else:

    ### Categroical

        prev_dist = active_stats[col]
        new_dist = latest_stats[col]

        # --> Key changes (like a new airport in the Top 40) automatically signal drift
        prev_keys = set(prev_dist.keys())
        new_keys = set(new_dist.keys())
        structure_changed = prev_keys != new_keys

        all_keys = prev_keys.union(new_keys)
        prev_probs = np.array([prev_dist.get(k, 0) for k in all_keys])
        new_probs = np.array([new_dist.get(k, 0) for k in all_keys])
        js_div = jensenshannon(prev_probs, new_probs)

        drift_summary[col]["categorical_drift"] = js_div
        drift_summary[col]["drift_flag"] = js_div > 0.1 or structure_changed
        drift_summary[col]["structure_changed"] = structure_changed

        if drift_summary[col]["drift_flag"]:
            drift_detected = True


# --------------------------------------------------------------------------------------------------------
# CONVERT TO JSON SERIALIZABLE OBJECT
# --------------------------------------------------------------------------------------------------------

def convert_to_python(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [convert_to_python(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    else:
        return obj

# --------------------------------------------------------------------------------------------------------
# SAVE DRIFT LOG
# --------------------------------------------------------------------------------------------------------

log = {
    "latest_stats_file": latest_file,
    "active_stats_file": active_file,
    "drift_detected": drift_detected,
    "drift_summary": drift_summary
}

log = convert_to_python(log)

with open(DRIFT_LOG, "w") as f:
    json.dump(log, f, indent=2)

print(f"Drift log saved to {DRIFT_LOG}")
print(f"Drift detected: {drift_detected}")

# --------------------------------------------------------------------------------------------------------
# UPDATE ACTIVE STATS IF DRIFT
# --------------------------------------------------------------------------------------------------------

if drift_detected:
    with open(ACTIVE_FILE, "w") as f:
        json.dump({"active_stats_file": latest_file}, f)
    print(f"Active stats updated to {latest_file}")
else:
    print(f"Active stats remains {active_file}")
