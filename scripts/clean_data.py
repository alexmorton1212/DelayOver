

### TODO: Check / rewrite holiday features

import pandas as pd
import numpy as np
import holidays
import calendar
import os

# --------------------------------------------------------------------------------------------------------
# DIRECTORIES
# --------------------------------------------------------------------------------------------------------

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MAPS_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'maps')


# --------------------------------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------------------------------

# Load parquet files (should have 12 months worth of data)
# --------------------------------------------------------------------------------------------------------

def load_parquet_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    print("\nAvailable parquet files (" + str(len(files)) + "): ", files)
    df = pd.concat([pd.read_parquet(os.path.join(directory, f)) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df


# General data cleaning (remove duplicates, fill NAs, etc)
# --------------------------------------------------------------------------------------------------------

def clean_and_filter_columns(df):
    kept_cols = ['year', 'month', 'dayofmonth', 'dayofweek', 'origin', 'dest', 'reporting_airline', 
        'originstate', 'deststate', 'crsdeptime', 'crsarrtime','carrierdelay', 'weatherdelay', 'flight_number_reporting_airline', 
        'nasdelay', 'securitydelay', 'lateaircraftdelay', 'arrdelayminutes', 'cancelled', 'diverted']
    delay_cols = ['carrierdelay', 'weatherdelay', 'nasdelay', 'securitydelay', 
        'lateaircraftdelay', 'arrdelayminutes']
    df = df.dropna(axis=1, how='all')
    df = df[[col for col in kept_cols if col in df.columns]].drop_duplicates()
    df[delay_cols] = df[delay_cols].fillna(0)
    df[delay_cols] = df[delay_cols].astype(int)
    return df


# Extract hour (0–23) directly from HHMM-formatted time (e.g., 1420 → 14)
# --------------------------------------------------------------------------------------------------------

def extract_hours_from_hhmm(df):
    df['crsdeptime'] = pd.to_numeric(df['crsdeptime'], errors='coerce')
    df['dep_hour'] = (df['crsdeptime'] // 100).astype('Int64')
    df['crsarrtime'] = pd.to_numeric(df['crsarrtime'], errors='coerce')
    df['arr_hour'] = (df['crsarrtime'] // 100).astype('Int64')
    return df


# Filter to only 50 US states & DC (excludes Canadian and other US territories)
# --------------------------------------------------------------------------------------------------------

def filter_valid_states(df):
    valid_states = ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN',
        'KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV',
        'NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
    return df[df['originstate'].isin(valid_states) & df['deststate'].isin(valid_states)].copy()


# Filter to top 75 airports based on combined arrival and departures
# --------------------------------------------------------------------------------------------------------

def filter_by_top_airports(df):
    n = 20 # --> top 75 airports by flight volume
    origin = df['origin'].value_counts()
    dest = df['dest'].value_counts()
    combined = origin.add(dest, fill_value=0)
    top_airports = combined.nlargest(n).index
    return df[df['origin'].isin(top_airports) & df['dest'].isin(top_airports)].copy()


### Create features based on proximity to holidays
# --------------------------------------------------------------------------------------------------------

def add_date_and_holiday_features(df):

    # Convert to datetime
    df['flight_date'] = pd.to_datetime(df[['year', 'month', 'dayofmonth']].rename(columns={'dayofmonth': 'day'}))

    us_holidays = holidays.US(years=df['year'].unique())
    major_holidays = {"New Year's Day": "A", "Memorial Day": "B", "Independence Day": "C",
        "Labor Day": "D", "Thanksgiving Day": "E", "Christmas Day": "F"}

    # Filter to relevant holiday dates and codes
    holiday_info = [
        (pd.Timestamp(date), major_holidays[name])
        for date, name in us_holidays.items()
        if name in major_holidays
    ]

    if not holiday_info:
        df['holiday_proximity_bucket'] = 5
        df['holiday_code'] = 'NA'
        return df

    # Build holiday date array
    holiday_dates = np.array([d[0] for d in holiday_info], dtype='datetime64[D]')
    holiday_codes = np.array([d[1] for d in holiday_info])

    # Calculate days difference (vectorized)
    flight_dates = df['flight_date'].values.astype('datetime64[D]')
    date_diffs = flight_dates[:, None] - holiday_dates[None, :]
    delta_days = np.abs(date_diffs.astype('timedelta64[D]').astype(int))

    # Find nearest holiday within 7 days
    min_diff = np.min(delta_days, axis=1)
    min_idx = np.argmin(delta_days, axis=1)

    # Assign bucket based on delta
    bucket = np.full(len(df), 5)  # Default: 5 = not near holiday
    bucket[min_diff == 0] = 1
    bucket[(min_diff == 1)] = 2
    bucket[(min_diff >= 2) & (min_diff <= 3)] = 3
    bucket[(min_diff >= 4) & (min_diff <= 7)] = 4

    # Assign holiday code (or NA if not within range)
    code = np.array(['NA'] * len(df), dtype=object)
    within_range = min_diff <= 7
    code[within_range] = holiday_codes[min_idx[within_range]]

    # Assign to dataframe
    df['holiday_proximity_bucket'] = bucket
    df['holiday_code'] = code

    return df


# --------------------------------------------------------------------------------------------------------
# CALL MAIN
# --------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    df_raw = load_parquet_files(RAW_DATA_DIR)
    df_clean = clean_and_filter_columns(df_raw)
    df_filtered = filter_valid_states(df_clean)
    df_filtered = filter_by_top_airports(df_filtered)
    df_filtered = extract_hours_from_hhmm(df_filtered)
    df_final = add_date_and_holiday_features(df_filtered)

    print("\n*** FINAL DATASET CREATED ***")


# --------------------------------------------------------------------------------------------------------
# SUMMARY STATISTICS DATASET
# --------------------------------------------------------------------------------------------------------

DELAY_SUMMARY_THRESHOLD = 15 # Delays for dashboard defined as more than 15 minutes

map_airline_df = pd.read_csv(os.path.join(MAPS_DATA_DIR, 'L_UNIQUE_CARRIERS.csv'))
map_airport_df = pd.read_csv(os.path.join(MAPS_DATA_DIR, 'L_AIRPORT.csv'))

df_summary = df_final[['origin', 'dest', 'reporting_airline', 'month', 'dayofweek', 'dep_hour', 'cancelled', 'diverted','arrdelayminutes']].copy()

df_summary['if_delay'] = np.where(df_summary['arrdelayminutes'] <= DELAY_SUMMARY_THRESHOLD, 0, 1)
df_summary = df_summary.rename(columns={'cancelled': 'if_cancelled', 'diverted': 'if_diverted'})
df_summary[['if_cancelled', 'if_diverted']] = df_summary[['if_cancelled', 'if_diverted']].astype(int)

df_summary = df_summary.merge(map_airline_df, how='left', left_on='reporting_airline', right_on='Code')
df_summary['airline_ui'] = df_summary['Description'].str.replace(r'\s*(Inc\.|Co\.)$', '', regex=True)
df_summary = df_summary.drop(columns=['reporting_airline', 'Code', 'Description'])

df_summary = df_summary.merge(map_airport_df, how='left', left_on='origin', right_on='Code')
df_summary['origin_ui'] = df_summary['origin'] + ' (' + df_summary['Description'].str.split(':').str[-1].str.strip() + ')'
df_summary = df_summary.drop(columns=['origin', 'Code', 'Description'])

df_summary = df_summary.merge(map_airport_df, how='left', left_on='dest', right_on='Code')
df_summary['destination_ui'] = df_summary['dest'] + ' (' + df_summary['Description'].str.split(':').str[-1].str.strip() + ')'
df_summary = df_summary.drop(columns=['dest', 'Code', 'Description'])

df_summary['month_ui'] = df_summary['month'].apply(lambda x: calendar.month_name[x])
df_summary = df_summary.drop(columns=['month'])

df_summary['day_ui'] = df_summary['dayofweek'].apply(lambda x: calendar.day_name[x-1])
df_summary = df_summary.drop(columns=['dayofweek'])

df_summary['hour_ui'] = pd.to_datetime(df_summary['dep_hour'], format='%H').dt.strftime('%I:%M %p')
df_summary = df_summary.drop(columns=['dep_hour'])

df_summary.to_parquet(PROCESSED_DATA_DIR + '/summary_dataset.parquet', engine='pyarrow', use_deprecated_int96_timestamps=False)

print("*** SUMMARY DATASET CREATED ***")


# --------------------------------------------------------------------------------------------------------
# MACHINE LEARNING DATASET
# --------------------------------------------------------------------------------------------------------

DELAY_ML_THRESHOLD = 30 # Delays in modeling defined as more than 30 minutes

df_ml = df_final[['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 
    'holiday_proximity_bucket', 'holiday_code', 'arrdelayminutes']].copy()

df_ml['if_delay'] = np.where(df_ml['arrdelayminutes'] <= DELAY_ML_THRESHOLD, 0, 1)
df_ml = df_ml.drop(columns=['arrdelayminutes'])

df_ml.to_parquet(PROCESSED_DATA_DIR + '/ml_dataset.parquet')

print("*** ML DATASET CREATED ***\n")
