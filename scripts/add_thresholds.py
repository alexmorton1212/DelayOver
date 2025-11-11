
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')

### Choose thresholds based on 'model_thresholds.csv'

thresholds = {
    "Delay Very Unlikely": 0.25,
    "Delay Unlikely": 0.45,
    "Delay Somewhat Likely": 0.55,
    "Delay Likely": 0.7,
    "Delay Very Likely": 1.0
}

with open(metadata_path, 'r') as f:
    metadata = json.load(f)
    metadata["thresholds"] = thresholds

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
