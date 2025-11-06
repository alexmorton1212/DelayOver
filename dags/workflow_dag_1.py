
# Frequency: run once a week
# --------------------------------------------------------------------------------------------------------

# If drift is NOT detected --> All done, push results to GitHub
# --------------------------------------------------------------------------------------------------------

# If drift is detected --> Re-run model training and manually adjust thresholds
# run workflow_dag_2.py --> evaluate model_thresholds.csv --> update add_thresholds.py
# run workflow_dag_3.py --> push results to GitHub
# --------------------------------------------------------------------------------------------------------

from datetime import datetime
import subprocess
import os

SCRIPT_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, f"workflow_{datetime.now():%Y%m%d_%H%M%S}_1.log")

tasks = ['ingest_data.py', 'clean_data.py', 'compute_stats.py', 'detect_drift.py']

with open(log_file_path, "w") as log_file:

    for task in tasks:

        print(f"\n=== Running {task} ===")
        log_file.write(f"\n=== Running {task} ===\n")
        script_path = os.path.join(SCRIPT_DIR, "..", "scripts", task)
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        log_file.write(result.stdout)
        log_file.write(result.stderr)
        
        if result.returncode != 0:
            print(f"!!! {task} failed! See log: {log_file_path}")
            log_file.write(f"\n!!! {task} failed!\n")
            break 
        else:
            print(f"--- {task} completed successfully.")
            log_file.write(f"\n--- {task} completed successfully.\n")

print(f"\n*** IF DRIFT NOT DETECTED: ALL DONE!!")
print(f"*** IF DRIFT DETECTED: RUN WORKFLOW_DAG_2 TO RE-TRAIN MODEL\n")