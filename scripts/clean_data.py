
import pandas as pd
import numpy as np
import holidays
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(script_dir, '..', 'data', 'raw')
processed_data_dir = os.path.join(script_dir, '..', 'data', 'processed')

#############################################################################################################
### FUNCTIONS
#############################################################################################################

### ---------------------------------------------------------------------------------------------------------
### Load parquet files (should have 12 months worth of data)
### ---------------------------------------------------------------------------------------------------------

def load_parquet_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    print("Available parquet files (" + str(len(files)) + "): ", files)
    df = pd.concat([pd.read_parquet(os.path.join(directory, f)) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df

### ---------------------------------------------------------------------------------------------------------
### General data cleaning (remove duplicates, fill NAs, etc)
### ---------------------------------------------------------------------------------------------------------

def clean_and_filter_columns(df, columns, delay_cols):
    df = df.dropna(axis=1, how='all')
    df = df[[col for col in columns if col in df.columns]].drop_duplicates()
    df[delay_cols] = df[delay_cols].fillna(0)
    df[delay_cols] = df[delay_cols].astype(int)
    return df

### ---------------------------------------------------------------------------------------------------------
### NOT USING THIS ONE ANYMORE
### Convert departure / arrival times from HHMM (ex. 1420 for 14:20 PM) to minutes after midnight (ex. 860)
### ---------------------------------------------------------------------------------------------------------

def extract_minutes_after_midnight(df, colname, new_colname):
    df[colname] = pd.to_numeric(df[colname], errors='coerce')  # convert to numeric, NaNs if invalid
    hours = df[colname] // 100
    minutes = df[colname] % 100
    df[new_colname] = hours * 60 + minutes
    return df

### ---------------------------------------------------------------------------------------------------------
### Extract hour (0–23) directly from HHMM-formatted time (e.g., 1420 → 14)
### ---------------------------------------------------------------------------------------------------------

def extract_hour_from_hhmm(df, colname, new_colname):
    df[colname] = pd.to_numeric(df[colname], errors='coerce')  # ensure numeric
    df[new_colname] = (df[colname] // 100).astype('Int64')     # supports NA
    return df

### ---------------------------------------------------------------------------------------------------------
### Filter to only 50 US states & DC (excludes Canadian and other US territories)
### ---------------------------------------------------------------------------------------------------------

def filter_valid_states(df, valid_states):
    return df[df['originstate'].isin(valid_states) & df['deststate'].isin(valid_states)].copy()

### ---------------------------------------------------------------------------------------------------------
### Filter to top 200 airports based on combined arrival and departures
### ---------------------------------------------------------------------------------------------------------

def get_top_airports(df, n=200):
    origin = df['origin'].value_counts()
    dest = df['dest'].value_counts()
    combined = origin.add(dest, fill_value=0)
    return combined.nlargest(n).index

def filter_by_top_airports(df, top_airports):
    return df[
        df['origin'].isin(top_airports) & df['dest'].isin(top_airports)
    ].copy()

### ---------------------------------------------------------------------------------------------------------
### Create features based on proximity to holidays
### ---------------------------------------------------------------------------------------------------------

def add_holiday_features(df, max_window=14, sentinel=99):
    # Step 1: Add a datetime column
    date_cols = df[['year', 'month', 'dayofmonth']].copy()
    date_cols.rename(columns={'dayofmonth': 'day'}, inplace=True)
    df['flight_date'] = pd.to_datetime(date_cols)

    # Step 2: Define relevant US holidays
    years = df['year'].unique()
    us_holidays = holidays.US(years=years)

    major_holidays = {
        "New Year's Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving",
        "Christmas Day"
    }

    filtered_holidays = {date: name for date, name in us_holidays.items() if name in major_holidays}
    holiday_dates = sorted(filtered_holidays.keys())

    # Step 3: Calculate proximity to nearest holiday
    def get_days_from_nearest_holiday(date):
        closest_delta = None
        for holiday in holiday_dates:
            delta = (date.date() - holiday).days
            if abs(delta) <= max_window:
                if (closest_delta is None) or (abs(delta) < abs(closest_delta)):
                    closest_delta = delta
                    closest_holiday = holiday
        return closest_delta

    # Step 3: Calculate proximity to nearest holiday
    def get_nearest_holiday(date):
        closest_delta = None
        closest_holiday = None
        for holiday in holiday_dates:
            delta = (date.date() - holiday).days
            if abs(delta) <= max_window:
                if (closest_delta is None) or (abs(delta) < abs(closest_delta)):
                    closest_delta = delta
                    closest_holiday = holiday
        return closest_holiday


    df['days_from_holiday_temp'] = df['flight_date'].apply(get_days_from_nearest_holiday)
    df['if_near_holiday'] = df['days_from_holiday_temp'].notna().astype(int)
    df['days_from_holiday'] = df['days_from_holiday_temp'].fillna(sentinel).astype(int)

    df['holiday'] = df['flight_date'].apply(get_nearest_holiday)
    df['holiday'] = df['holiday'].fillna("NA")

    return df


#############################################################################################################
### CALL MAIN
#############################################################################################################

# delays defined as more than 15 minutes

if __name__ == "__main__":
    cols = ['year', 'month', 'dayofmonth', 'dayofweek', 'origin', 'dest', 'reporting_airline', 
        'originstate', 'deststate', 'crsdeptime', 'crsarrtime','carrierdelay', 'weatherdelay', 
        'nasdelay', 'securitydelay', 'lateaircraftdelay', 'arrdelayminutes', 'cancelled', 'diverted']
    delay_cols = ['carrierdelay', 'weatherdelay', 'nasdelay', 'securitydelay', 
        'lateaircraftdelay', 'arrdelayminutes']
    state_list = ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN',
        'KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV',
        'NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']

    df_raw = load_parquet_files(raw_data_dir)
    df_clean = clean_and_filter_columns(df_raw, cols, delay_cols)
    df_filtered = filter_valid_states(df_clean, state_list)
    df_filtered = df_filtered.drop(columns=['originstate', 'deststate'])
    #df_filtered = extract_minutes_after_midnight(df_filtered, 'crsdeptime', 'deptime_mins')
    #df_filtered = extract_minutes_after_midnight(df_filtered, 'crsarrtime', 'arrtime_mins')
    df_filtered = extract_hour_from_hhmm(df_filtered, 'crsdeptime', 'dep_hour')
    df_filtered = extract_hour_from_hhmm(df_filtered, 'crsarrtime', 'arr_hour')
    df_filtered = df_filtered.drop(columns=['crsdeptime', 'crsarrtime'])
    df_filtered = add_holiday_features(df_filtered)
    df_filtered = df_filtered.drop(columns=['year', 'flight_date', 'days_from_holiday_temp'])
    df_filtered['if_delay'] = np.where(df_filtered['arrdelayminutes'] <= 15, '0', '1').astype(int)
    df_filtered['if_cancelled'] = np.where(df_filtered['cancelled'] == 0, '0', '1').astype(int)
    df_filtered['if_diverted'] = np.where(df_filtered['diverted'] == 0, '0', '1').astype(int)
    df_filtered = df_filtered.drop(columns=['cancelled', 'diverted'])
    df_filtered['nonweatherdelay'] = df_filtered['arrdelayminutes'] - df_filtered['weatherdelay']
    top_airports = get_top_airports(df_filtered)
    df_final = filter_by_top_airports(df_filtered, top_airports)
    
    print("✅ FINAL DATASET CREATED")


#############################################################################################################
### SUMMARY STATISTICS DATASET
#############################################################################################################

summary_cols = ['origin', 'dest', 'reporting_airline', 'month', 'dayofweek', 'if_near_holiday']

df_summary = df_final.copy().groupby(summary_cols).agg(
    total_flights = ('if_delay', 'count'),
    delayed_flights = ('if_delay', 'sum'),
    cancelled_flights = ('if_cancelled', 'sum'),
    diverted_flights = ('if_diverted', 'sum'),
    total_delay_minutes = ('arrdelayminutes', 'sum'),
    #delay_minutes_90th = ('arrdelayminutes', lambda x: int(x.quantile(0.90))),
    #delay_minutes_95th = ('arrdelayminutes', lambda x: int(x.quantile(0.95))),
    #delay_minutes_99th = ('arrdelayminutes', lambda x: int(x.quantile(0.99)))
).reset_index()

df_summary.to_parquet(processed_data_dir + '/summary_dataset.parquet')

print("✅ SUMMARY DATASET CREATED")

#############################################################################################################
### MACHINE LEARNING DATASET
#############################################################################################################

ml_cols = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 'if_near_holiday', 'days_from_holiday', 'arrdelayminutes']

df_ml = df_final[ml_cols].copy()

df_ml.to_parquet(processed_data_dir + '/ml_dataset.parquet')

print("✅ ML DATASET CREATED")
