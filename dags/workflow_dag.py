
from datetime import datetime
import subprocess
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, f"workflow_{datetime.now():%Y%m%d_%H%M%S}.log")
tasks = ['ingest_data.py', 'clean_data.py', 'compute_stats.py', 'detect_drift.py']

with open(log_file_path, "w") as log_file:
    for task in tasks:
        print(f"\n=== Running {task} ===")
        log_file.write(f"\n=== Running {task} ===\n")
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", task)
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

print(f"\n*** Process Completed\n")