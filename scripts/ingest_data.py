import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

BASE_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
DOWNLOAD_DIR = "data/raw"
MAX_MONTHS_LOOKBACK = 12  # Try up to 12 months back
REQUIRED_MONTHS = 3       # Rolling 3 months data

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Build the correct URL format (no leading zero for month)
# ------------------------------------------------------------------------------

def build_url(year: int, month: int) -> str:
    return f"{BASE_URL}_{year}_{month}.zip"

# ------------------------------------------------------------------------------
# Try downloading the ZIP file using streaming + long timeout + progress bar
# ------------------------------------------------------------------------------

def try_download_zip(url: str) -> BytesIO | None:
    """Downloads a ZIP file from BTS via streaming with timeout and progress bar."""
    try:
        print(f"🌐 Attempting to download: {url}")

        with requests.get(url, stream=True, timeout=600) as response:
            if response.status_code != 200:
                print(f"❌ Status {response.status_code}")
                return None

            total_size = int(response.headers.get('content-length', 0))
            zip_data = BytesIO()

            progress = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc="⬇️  Downloading",
                ncols=80,
            )

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    zip_data.write(chunk)
                    progress.update(len(chunk))

            progress.close()
            zip_data.seek(0)
            print("✅ Download successful.")
            return zip_data

    except requests.exceptions.Timeout:
        print("❌ Timeout occurred while downloading.")
    except Exception as e:
        print(f"❌ Error during download: {e}")

    return None

# ------------------------------------------------------------------------------
# Extract ZIP contents, convert CSV to Parquet, save readme.html only once
# ------------------------------------------------------------------------------

def extract_and_process_zip(zip_bytes: BytesIO, year: int, month: int, save_readme: bool) -> bool:
    """
    Extracts ZIP, converts CSV to Parquet, saves readme.html once.
    Returns True if readme.html saved, else False.
    """
    readme_saved = False
    with ZipFile(zip_bytes) as zip_file:
        for file_info in zip_file.infolist():
            filename = file_info.filename

            if filename.lower() == "readme.html" and save_readme:
                readme_path = os.path.join(DOWNLOAD_DIR, "readme.html")
                with open(readme_path, "wb") as f:
                    f.write(zip_file.read(filename))
                print(f"📄 Saved readme.html to {readme_path}")
                readme_saved = True

            elif filename.endswith(".csv"):
                print(f"📥 Processing CSV file: {filename}")
                with zip_file.open(filename) as csv_file:
                    df = pd.read_csv(csv_file, low_memory=False)

                    parquet_filename = f"flight_data_{year}_{month}.parquet"
                    parquet_path = os.path.join(DOWNLOAD_DIR, parquet_filename)
                    df.to_parquet(parquet_path, index=False)

                    print(f"✅ Saved parquet: {parquet_path}")

    return readme_saved

# ------------------------------------------------------------------------------
# Main logic: find latest 3 months with data, download and process them
# ------------------------------------------------------------------------------

def main():
    current_date = datetime.today()
    months_found = 0
    readme_saved = False

    for i in range(MAX_MONTHS_LOOKBACK):
        check_date = current_date - relativedelta(months=i)
        year = check_date.year
        month = check_date.month

        url = build_url(year, month)
        zip_bytes = try_download_zip(url)

        if zip_bytes:
            # Save readme only for the latest (first) month downloaded
            save_readme = (months_found == 0 and not readme_saved)

            saved_readme_now = extract_and_process_zip(zip_bytes, year, month, save_readme)
            if saved_readme_now:
                readme_saved = True

            months_found += 1
            print(f"🎉 Downloaded & processed data for: {year}-{month}")

            if months_found >= REQUIRED_MONTHS:
                break

    if months_found == 0:
        print("⚠️ No valid BTS data found in the last 12 months.")
    elif months_found < REQUIRED_MONTHS:
        print(f"⚠️ Only found {months_found} months of data.")

if __name__ == "__main__":
    main()
