
### TODO: Change from Pandas querying to DuckDB

import os
import calendar
import datetime
import joblib
import json
import streamlit as st
import pandas as pd
import plotly.express as px


# -----------------------------------------------------------------------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------------------------------------------------------------------

header_container = st.container()

with header_container:

    # Remove Streamlit default padding
    st.markdown("""<style>.block-container {padding-top: 2rem;}</style>""", unsafe_allow_html=True)

    # Main title
    st.markdown("""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:3.2em;">DelayOver</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("___")

    # Subtitle
    st.markdown("""
    <div style="text-align:center;">
        <span style="font-size:1.4em;">
            <i>Based on <b>August 2024 - July 2025</b> Bureau of Transportation Statistics (BTS)
            flight data between the <b>Top 50 U.S. Airports</b></i>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("___")


# -----------------------------------------------------------------------------------------------------------------------------------------
# PRE-PROCESSING
# -----------------------------------------------------------------------------------------------------------------------------------------

st.set_page_config(layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
PKL_FILE = os.path.join(MODEL_DIR, 'final_model.pkl')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
PARQUET_FILE = os.path.join(PROCESSED_DATA_DIR, 'summary_dataset.parquet')

### LOAD DATA
@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path, engine='pyarrow')
df = load_data(PARQUET_FILE)

### LOAD MODEL PIPELINE
@st.cache_data
def load_model(file_path):
    return joblib.load(file_path)
pipeline = load_model(PKL_FILE)

### LOAD MODEL THRESHOLDS
@st.cache_data
def load_metadata(file_path):
    with open(file_path, "r") as f:
        metadata = json.load(f)
    return metadata
metadata = load_metadata(METADATA_FILE)
thresholds = metadata.get("thresholds", {})


# -----------------------------------------------------------------------------------------------------------------------------------------
# COLORS
# -----------------------------------------------------------------------------------------------------------------------------------------

my_green = "#3AA655"
my_lime = "#8EA671"
my_yellow = "#CBA135"
my_orange = "#BF6828"
my_red = "#B04C4C"
my_grey = "#8A8A8A"

my_green_box = "#256B37"
my_lime_box = "#5E6E4C"
my_yellow_box = "#8A6D22"
my_orange_box = "#703D16"
my_red_box = "#662A2A"
my_grey_box = "#616161"

my_lightblue = "#5D8199"


# -----------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------------------------------

def generate_hour_labels():
    hours = []
    for h in range(24):
        start = datetime.time(h).strftime("%I:00 %p")
        end = datetime.time(h).strftime("%I:59 %p")
        label = f"{start} - {end}"
        hours.append(label)
    return hours

def get_airline_code(airline: str) -> str:
    codes = {"Alaska Airlines": "AS", "Allegiant Air": "G4", "American Airlines": "AA", "Delta Air Lines": "DL",
        "Endeavor Air": "9E", "Envoy Air": "MQ", "Frontier Airlines": "F9", "Hawaiian Airlines": "HA",
        "JetBlue Airways": "B6", "PSA Airlines": "OH", "Republic Airline": "YX", "SkyWest Airlines": "OO",
        "Southwest Airlines": "WN", "Spirit Air Lines": "NK", "United Air Lines": "UA"
    }
    return codes.get(airline, "")

def get_prob_label(prob, thresholds):
    if prob is None:
        return "No probability available"
    sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
    for i, (name, value) in enumerate(sorted_thresholds):
        if prob < value:
            return name
    return sorted_thresholds[-1][0]

def safe_quantile(series, q):
    val = series.quantile(q)
    return int(val) if pd.notnull(val) else 0


# -----------------------------------------------------------------------------------------------------------------------------------------
# DROPDOWNS
# -----------------------------------------------------------------------------------------------------------------------------------------

origin_options = ["All Airports"] + sorted(df['origin_ui'].unique().tolist())
destination_options = ["All Airports"] + sorted(df['destination_ui'].unique().tolist())
airline_options = ["All Airlines"] + sorted(df['airline_ui'].unique().tolist())
month_options = ["All Months"] + sorted(df['month_ui'].unique().tolist(), key=lambda m: list(calendar.month_name).index(m))
day_options = ["All Days"] + sorted(df['day_ui'].unique().tolist(), key=lambda d: list(calendar.day_name).index(d))

hour_labels = generate_hour_labels()
hour_map = {label: datetime.time(h).strftime("%I:00 %p") for h, label in enumerate(hour_labels)}
hour_options = ["All Hours"] + hour_labels

st.markdown("""
<style>.centered-label {
    text-align: center !important;
    margin-bottom: 5px;
    font-weight: 500;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    left, col1, col2, col3, right = st.columns([3, 5, 5, 5, 3], gap="small")
    with col1:
        st.markdown('<div class="centered-label">Origin</div>', unsafe_allow_html=True)
        selected_origin = st.selectbox("Airport From", origin_options, key="dd1", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="centered-label">Destination</div>', unsafe_allow_html=True)
        selected_destination = st.selectbox("Airport To", destination_options, key="dd2", label_visibility="collapsed")
    with col3:
        st.markdown('<div class="centered-label">Airline</div>', unsafe_allow_html=True)
        selected_airline = st.selectbox("Airline", airline_options, key="dd3", label_visibility="collapsed")

with st.container():
    left, col4, col5, col6, right = st.columns([3, 5, 5, 5, 3], gap="small")
    with col4:
        st.markdown('<div class="centered-label">Month</div>', unsafe_allow_html=True)
        selected_month = st.selectbox("Month", month_options, key="dd4", label_visibility="collapsed")
    with col5:
        st.markdown('<div class="centered-label">Day of Week</div>', unsafe_allow_html=True)
        selected_day = st.selectbox("Day", day_options, key="dd5", label_visibility="collapsed")
    with col6:
        st.markdown('<div class="centered-label">Departure Time</div>', unsafe_allow_html=True)
        selected_hour_label = st.selectbox("Hour", hour_options, key="dd6", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("___")


# -----------------------------------------------------------------------------------------------------------------------------------------
# APPLY DROPDOWN FILTERS
# -----------------------------------------------------------------------------------------------------------------------------------------

filtered_df = df
selected_hour = hour_map.get(selected_hour_label, "All Hours")

if selected_origin != "All Airports":
    filtered_df = filtered_df[filtered_df["origin_ui"] == selected_origin]
if selected_destination != "All Airports":
    filtered_df = filtered_df[filtered_df["destination_ui"] == selected_destination]
if selected_airline != "All Airlines":
    filtered_df = filtered_df[filtered_df["airline_ui"] == selected_airline]
if selected_month != "All Months":
    filtered_df = filtered_df[filtered_df["month_ui"] == selected_month]
if selected_day != "All Days":
    filtered_df = filtered_df[filtered_df["day_ui"] == selected_day]
if selected_hour != "All Hours":
    filtered_df = filtered_df[filtered_df["hour_ui"] == selected_hour]

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------------------------------------------------------------------
# CALCULATIONS
# -----------------------------------------------------------------------------------------------------------------------------------------

### Number of Flights

total_nf = len(filtered_df)
delayed_nf = filtered_df["if_delay"].sum()
diverted_nf = filtered_df["if_diverted"].sum()
cancelled_nf = filtered_df["if_cancelled"].sum()

### On-Time Percentage

total_otp = 0 if filtered_df.empty else len(filtered_df)
other_otp = 0 if filtered_df.empty else filtered_df["if_delay"].sum() + filtered_df["if_diverted"].sum() + filtered_df["if_cancelled"].sum()
on_time_otp = total_otp - other_otp
on_time_percent_otp = 0 if filtered_df.empty else round(100 * (on_time_otp / total_otp)) if total_otp > 0 else 0

total_all_otp = len(df)
other_all_otp = df["if_delay"].sum() + df["if_diverted"].sum() + df["if_cancelled"].sum()
on_time_percent_all_otp = round(100 * (total_all_otp - other_all_otp) / total_all_otp)

if filtered_df.empty: colors_otp = [my_grey, "#C7C7C7"]
elif on_time_percent_otp >= on_time_percent_all_otp: colors_otp = [my_green, "#C7C7C7"]
elif on_time_percent_otp >= 0.85 * on_time_percent_all_otp: colors_otp = [my_yellow, "#C7C7C7"]
else: colors_otp = [my_red, "#C7C7C7"] 

if filtered_df.empty: box_colors_otp = my_grey_box
elif on_time_percent_otp >= on_time_percent_all_otp: box_colors_otp = my_green_box
elif on_time_percent_otp >= 0.85 * on_time_percent_all_otp: box_colors_otp = my_yellow_box
else: box_colors_otp = my_red_box

on_time_percent_otp = "NA" if filtered_df.empty else str(on_time_percent_otp) + "%"

### Percentiles

quant_90_all = safe_quantile(df['arrdelayminutes'], 0.90)
quant_95_all = safe_quantile(df['arrdelayminutes'], 0.95)
quant_99_all = safe_quantile(df['arrdelayminutes'], 0.99)
quant_90 = safe_quantile(filtered_df['arrdelayminutes'], 0.90)
quant_95 = safe_quantile(filtered_df['arrdelayminutes'], 0.95)
quant_99 = safe_quantile(filtered_df['arrdelayminutes'], 0.99)
quant_90_color = my_grey if filtered_df.empty else my_green if quant_90 <= quant_90_all else my_yellow if quant_90 <= 1.4 * (quant_90_all) else my_red
quant_95_color = my_grey if filtered_df.empty else my_green if quant_95 <= quant_95_all else my_yellow if quant_95 <= 1.3 * (quant_95_all) else my_red
quant_99_color = my_grey if filtered_df.empty else my_green if quant_99 <= quant_99_all else my_yellow if quant_99 <= 1.18 * (quant_99_all) else my_red
quant_90_box = my_grey_box if filtered_df.empty else my_green_box if quant_90 <= quant_90_all else my_yellow_box if quant_90 <= 1.4 * (quant_90_all) else my_red_box
quant_95_box = my_grey_box if filtered_df.empty else my_green_box if quant_95 <= quant_95_all else my_yellow_box if quant_95 <= 1.3 * (quant_95_all) else my_red_box if not filtered_df.empty else my_grey_box
quant_99_box = my_grey_box if filtered_df.empty else my_green_box if quant_99 <= quant_99_all else my_yellow_box if quant_99 <= 1.18 * (quant_99_all) else my_red_box if not filtered_df.empty else my_grey_box
quant_90 = "NA" if filtered_df.empty else str(quant_90) + " min"
quant_95 = "NA" if filtered_df.empty else str(quant_95) + " min"
quant_99 = "NA" if filtered_df.empty else str(quant_99) + " min"

### Prediction

month_pred = datetime.datetime.strptime(selected_month.strip(), "%B").month if selected_month != "All Months" else ""
dayofweek_pred = datetime.datetime.strptime(selected_day.strip(), "%A").isoweekday() if selected_day != "All Days" else ""
origin_pred = selected_origin[:3] if selected_origin != "All Airports" else ""
dest_pred = selected_destination[:3] if selected_destination != "All Airports" else ""
reporting_airline_pred = get_airline_code(selected_airline)
dep_hour_pred =  datetime.datetime.strptime(selected_hour.strip().upper(), "%I:%M %p").hour if selected_hour != "All Hours" else ""
fields_pred = [month_pred, dayofweek_pred, origin_pred, dest_pred, reporting_airline_pred, dep_hour_pred]

data_pred = pd.DataFrame({"month": [month_pred], "dayofweek": [dayofweek_pred], "origin": [origin_pred.strip().upper()],
    "dest": [dest_pred.strip().upper()], "reporting_airline": [reporting_airline_pred.strip().upper()], "dep_hour": [dep_hour_pred]})
prediction = pipeline.predict(data_pred)[0]
try: pred_prob = pipeline.predict_proba(data_pred)[0][1]
except Exception: pred_prob = None

filters_sum = sum(x is not None and x != "" for x in fields_pred)
delay_label = "Select 2 more filters to view" if filters_sum == 0 else "Select 1 more filter to view" if filters_sum == 1 else get_prob_label(pred_prob, thresholds)
if delay_label == "Delay Very Unlikely": pred_color, pred_color_box = my_green, my_green_box
elif delay_label == "Delay Unlikely": pred_color, pred_color_box = my_lime, my_lime_box
elif delay_label == "Delay Somewhat Likely": pred_color, pred_color_box = my_yellow, my_yellow_box
elif delay_label == "Delay Likely": pred_color, pred_color_box = my_orange, my_orange_box
elif delay_label == "Delay Very Likely": pred_color, pred_color_box = my_red, my_red_box
else: pred_color, pred_color_box = my_grey, my_grey_box


# -----------------------------------------------------------------------------------------------------------------------------------------
# STYLES
# -----------------------------------------------------------------------------------------------------------------------------------------

nf_card_style = f"""
<div style="
    background-color: {{color}};
    border-radius:50px;
    padding:10px;
    box-shadow:0 10px 0 0 #3E5566;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    width:100%;
    word-wrap:break-word;
    overflow-wrap:break-word;
    margin-bottom: 50px;
">
    <p style="
        color: white;
        font-size:max(0.8vw, 1.2em);
        font-weight:350;
        margin:0;
        line-height:1.3em;
        text-align:center;
        margin-bottom: 3px;
        word-break:break-word;
    ">{{title}}</p>
    <p style="
        color: white;
        font-size:max(1vw, 1.4em);
        font-weight:bold;
        margin:0;
        line-height:1.3em;
        text-align:center;
        word-break:break-word;
    ">{{metric}}</p>
</div>
"""

pred_style = f"""
<div style="
    background-color: {{color}};
    border-radius:50px;
    padding:10px;
    box-shadow:0 10px 0 0 {{box_color}};
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    min-height:50px;
    width:100%;
    word-wrap:break-word;
    overflow-wrap:break-word;
    margin-bottom: 50px;
">
    <p style="
        color: white;
        font-size:max(1vw, 1.4em);
        font-weight:500;
        margin:0;
        line-height:1.5em;
        text-align:center;
        word-break:break-word;
    ">{{pred}}</p>
</div>
"""

otp_style = f"""
<div style="
    background-color: {{color}};
    border-radius:50px;
    padding:10px;
    box-shadow:0 10px 0 0 {{box_color}};
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    min-height:50px;
    width:100%;
    word-wrap:break-word;
    overflow-wrap:break-word;
    margin-bottom: 50px;
">
    <p style="
        color: white;
        font-size:max(1vw, 1.4em);
        font-weight:500;
        margin:0;
        line-height:1.5em;
        text-align:center;
        word-break:break-word;
    ">{{percent}}</p>
</div>
"""

perc_card_style = f"""
    <p style="
        color: white;
        font-size:max(0.8vw, 1em);
        margin:0;
        line-height:1.2em;
        text-align:center;
        word-break:break-word;
        margin-top: max(1.6vw, 1em);
        margin-bottom: 10px;
    ">{{metric}} of flights land within</p>
<div style="
    background-color: {{color}};
    border-radius:100px;
    padding:10px;
    box-shadow:0 10px 0 0 {{box_color}};
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    width:100%;
    word-wrap:break-word;
    overflow-wrap:break-word;
    margin-bottom: 20px;
">
    <p style="
        color: white;
        font-size:max(1vw, 1.4em);
        font-weight:500;
        margin:0;
        line-height:1.5em;
        text-align:center;
        word-break:break-word;
    ">{{title}}</p>
</div>
    <!--
    <p style="
        color: white;
        font-size:max(0.8vw, 1em);
        margin:0;
        line-height:1.2em;
        text-align:center;
        margin-bottom: 30px;
        word-break:break-word;
    ">of scheduled arrival time</p>
    -->
"""


# -----------------------------------------------------------------------------------------------------------------------------------------
# DASHBOARD SECTION 1: ON-TIME PERCENTAGE & PREDICTION
# -----------------------------------------------------------------------------------------------------------------------------------------

spacer_left_2, col1_2, spacer_middle_2a, col2_2, spacer_right_2 = st.columns([10, 35, 1, 35, 10], gap="small")

# -- PREDICTION

with col1_2:

    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">Likelihood of a 30+ min delay</span>
    </div><br>
    """, unsafe_allow_html=True)

    st.markdown(pred_style.format(pred=f"{delay_label}", color=pred_color, box_color=pred_color_box), unsafe_allow_html=True)

# -- ON-TIME PERCENTAGE

with col2_2:

    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">On-Time %</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">Landed within 15 min of schedule</span>
    </div><br>
    """, unsafe_allow_html=True)

    st.markdown(otp_style.format(percent=f"{on_time_percent_otp}", color=colors_otp[0], box_color=box_colors_otp), unsafe_allow_html=True)


# -----------------------------------------------------------------------------------------------------------------------------------------
# DASHBOARD SECTION 2: NUMBER OF FLIGHTS
# -----------------------------------------------------------------------------------------------------------------------------------------

spacer_left_1, col1_1, spacer_right_1 = st.columns([1, 20, 1], gap="small")

# -- NUMBER OF FLIGHTS

with col1_1:

    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">Number of Flights</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">Flights that land 15+ min late are considered delayed</span>
    </div><br>
    """, unsafe_allow_html=True)

    subcol1_1, subcol2_1, subcol3_1 = st.columns([1, 1, 1], gap="medium")

    with subcol1_1: st.markdown(nf_card_style.format(title="Total", metric=f"{total_nf:,}", color="#5D8199"), unsafe_allow_html=True)
    with subcol2_1: st.markdown(nf_card_style.format(title="Delayed", metric=f"{delayed_nf:,}", color="#5D8199"), unsafe_allow_html=True)
    with subcol3_1: st.markdown(nf_card_style.format(title="Cancelled", metric=f"{cancelled_nf:,}", color="#5D8199"), unsafe_allow_html=True)


# -----------------------------------------------------------------------------------------------------------------------------------------
# DASHBOARD SECTION 3: ARRIVAL TIME PERCENTILES
# -----------------------------------------------------------------------------------------------------------------------------------------

# -- ARRIVAL TIME PERCENTILES

st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">Arrival Time Percentiles</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">When flights land relative to scheduled arrival times</span>
    </div>
    """, unsafe_allow_html=True)

spacer_left, col1, col2, col3, spacer_right = st.columns([1, 6, 6, 6, 1], gap="medium")

with col1: st.markdown(perc_card_style.format(title=f"{quant_90}", metric="90%", color=quant_90_color, box_color=quant_90_box), unsafe_allow_html=True)
with col2: st.markdown(perc_card_style.format(title=f"{quant_95}", metric="95%", color=quant_95_color, box_color=quant_95_box), unsafe_allow_html=True)
with col3: st.markdown(perc_card_style.format(title=f"{quant_99}", metric="99%", color=quant_99_color, box_color=quant_99_box), unsafe_allow_html=True)