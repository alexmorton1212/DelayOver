
### TODO: Change from Pandas querying to DuckDB
### TODO: Add ML prediction

import os
import calendar
import datetime
import joblib
import json
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------------------------------------------------------------
# PRE-PROCESSING
# --------------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------------
# COLORS
# --------------------------------------------------------------------------------------------------------

my_green = "#3AA655"
my_lime = "#8EA671"
my_yellow = "#CBA135"
my_orange = "#BF6828"
my_red = "#B04C4C"
my_grey = "#8A8A8A"


# --------------------------------------------------------------------------------------------------------
# TITLE
# --------------------------------------------------------------------------------------------------------

# Remove Streamlit default padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center;">
    <span style="font-weight:bold; font-size:3.2em;">DelayOver</span>
</div>
""", unsafe_allow_html=True)

st.markdown("___")

st.markdown(f"""
<div style="text-align:center;">
    <span style="font-size:1.4em;"><i>Based on <b>August 2024 - July 2025</b> Bureau of Transportation Statistics (BTS) flight data between the <b>Top 40 U.S. Airports</b></i></span>
</div>
""", unsafe_allow_html=True)

st.markdown("___")


# --------------------------------------------------------------------------------------------------------
# DROPDOWNS
# --------------------------------------------------------------------------------------------------------

def generate_hour_labels():
    hours = []
    for h in range(24):
        start = datetime.time(h).strftime("%I:00 %p")
        end = datetime.time(h).strftime("%I:59 %p")
        label = f"{start} - {end}"
        hours.append(label)
    return hours

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

# --------------------------------------------------------------------------------------------------------
# APPLY DROPDOWN FILTERS
# --------------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------------
# COUNT CARD FIGURES
# --------------------------------------------------------------------------------------------------------

st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">Number of Flights</span>
    </div><br>
    """, unsafe_allow_html=True)

total = len(filtered_df)
delayed = filtered_df["if_delay"].sum()
diverted = filtered_df["if_diverted"].sum()
cancelled = filtered_df["if_cancelled"].sum()

card_style = f"""
<div style="
    background-color: #5D8199;
    border-radius:10px;
    padding:15px;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    min-height:100px;
    width:100%;
    word-wrap:break-word;
    overflow-wrap:break-word;
    margin-bottom: 15px;
">
    <p style="
        color: white;
        font-size:max(0.8vw, 1.4em);
        font-weight:350;
        margin:0;
        line-height:1.5em;
        text-align:center;
        word-break:break-word;
    ">{{title}}</p>
    <p style="
        color: white;
        font-size:max(0.8vw, 1.4em);
        font-weight:bold;
        margin:0;
        line-height:1.5em;
        text-align:center;
        word-break:break-word;
    ">{{metric}}</p>
</div>
"""

spacer_left, col1, col2, col3, col4, spacer_right = st.columns([2, 5, 5, 5, 5, 2], gap="small")
with col1: st.markdown(card_style.format(title="Total", metric=f"{total:,}"), unsafe_allow_html=True)
with col2: st.markdown(card_style.format(title="Delayed", metric=f"{delayed:,}"), unsafe_allow_html=True)
with col3: st.markdown(card_style.format(title="Cancelled", metric=f"{cancelled:,}"), unsafe_allow_html=True)
with col4: st.markdown(card_style.format(title="Diverted", metric=f"{diverted:,}"), unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------
# ON-TIME PERCENTAGE
# --------------------------------------------------------------------------------------------------------

total = 0 if filtered_df.empty else len(filtered_df)
other = 0 if filtered_df.empty else filtered_df["if_delay"].sum() + filtered_df["if_diverted"].sum() + filtered_df["if_cancelled"].sum()
on_time = total - other
on_time_percent = 0 if filtered_df.empty else round(100 * (on_time / total)) if total > 0 else 0
delayed_percent = 100 - on_time_percent

total_all = len(df)
other_all = df["if_delay"].sum() + df["if_diverted"].sum() + df["if_cancelled"].sum()
on_time_percent_all = round(100 * (total_all - other_all) / total_all)

if filtered_df.empty: colors = ["white", "#C7C7C7"]
elif on_time_percent >= on_time_percent_all: colors = [my_green, "#C7C7C7"]
elif on_time_percent >= 0.85 * on_time_percent_all: colors = [my_yellow, "#C7C7C7"]
else: colors = [my_red, "#C7C7C7"] 

chart_df = pd.DataFrame({"Category": ["On-Time", "Delayed, Diverted, or Cancelled"], "Value": [on_time, other]})
if chart_df["Value"].sum() > 0: chart_df["Percent"] = (chart_df["Value"] / chart_df["Value"].sum() * 100).round().astype(int)
else: chart_df["Percent"] = [0, 0]
chart_df["Label"] = ''
chart_df["Y"] = "All"

if filtered_df.empty:
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">On-Time Percentage: NA</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">No matching data was found with current filters</span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">On-Time Percentage: </span>
        <span style="color: {colors[0]}; font-weight:bold; font-size:1.7em;">{on_time_percent}%</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">Flights that landed within 15 minutes of scheduled arrival</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

fig = px.bar(
    chart_df,
    x="Percent",
    y="Y",
    color="Category",
    orientation="h",
    text="Label",
    color_discrete_sequence=colors
)

# Style chart
fig.update_layout(
    barmode="stack",
    showlegend=False,
    legend_title_text=None,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    xaxis_title="",
    yaxis=dict(showticklabels=False, showgrid=False, visible=False),
    height=100,
    margin=dict(l=10, r=10, t=10, b=10)
)

# Set hovertemplate per trace
for trace in fig.data:
    trace.hovertemplate = f"<b>{trace.name}</b><br>%{{x}}%<extra></extra>"

# Center chart in Streamlit
col1, col2, col3 = st.columns([4, 12, 3])
with col2:
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------------
# PERCENTILE CARD FIGURES
# --------------------------------------------------------------------------------------------------------

st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">Arrival Time Percentiles</span>
    </div><br>
    """, unsafe_allow_html=True)


def safe_quantile(series, q):
    val = series.quantile(q)
    return int(val) if pd.notnull(val) else 0

quant_90_all = safe_quantile(df['arrdelayminutes'], 0.90)
quant_95_all = safe_quantile(df['arrdelayminutes'], 0.95)
quant_99_all = safe_quantile(df['arrdelayminutes'], 0.99)

quant_90 = safe_quantile(filtered_df['arrdelayminutes'], 0.90)
quant_95 = safe_quantile(filtered_df['arrdelayminutes'], 0.95)
quant_99 = safe_quantile(filtered_df['arrdelayminutes'], 0.99)

if filtered_df.empty:
    quant_90_color = my_grey
    quant_95_color = my_grey
    quant_99_color = my_grey
else:
    quant_90_color = my_green if quant_90 <= quant_90_all else my_yellow if quant_90 <= 1.4 * (quant_90_all) else my_red
    quant_95_color = my_green if quant_95 <= quant_95_all else my_yellow if quant_95 <= 1.3 * (quant_95_all) else my_red
    quant_99_color = my_green if quant_99 <= quant_99_all else my_yellow if quant_99 <= 1.18 * (quant_99_all) else my_red


card_style = f"""
<div style="
    background-color: {{color}};
    border-radius:10px;
    padding:15px;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    min-height:100px;
    width:100%;
    word-wrap:break-word;
    overflow-wrap:break-word;
    margin-bottom: 15px;
">
    <p style="
        color: white;
        font-size:max(0.8vw, 1em);
        margin:0;
        line-height:1.2em;
        text-align:center;
        word-break:break-word;
    ">{{metric}} of flights land within</p>
    <p style="
        color: white;
        font-size:max(1.4vw, 1.8em);
        font-weight:bold;
        margin:0;
        line-height:2em;
        text-align:center;
        word-break:break-word;
    ">{{title}}</p>
    <p style="
        color: white;
        font-size:max(0.8vw, 1em);
        margin:0;
        line-height:1.2em;
        text-align:center;
        word-break:break-word;
    ">of scheduled arrival time</p>
</div>
    <p style="
        font-size:max(0.8vw, 1em);
        margin:0;
        line-height:1.2em;
        text-align:center;
        word-break:break-word;
        margin-bottom: 40px;
    ">{{national}}</p>
"""

spacer_left, col1, col2, col3, spacer_right = st.columns([2, 7, 7, 7, 2], gap="small")

with col1: st.markdown(card_style.format(title=f"{quant_90} min", metric="90%", national=f"National 90th Percentile: {quant_90_all} min", color=quant_90_color), unsafe_allow_html=True)
with col2: st.markdown(card_style.format(title=f"{quant_95} min", metric="95%", national=f"National 95th Percentile: {quant_95_all} min", color=quant_95_color), unsafe_allow_html=True)
with col3: st.markdown(card_style.format(title=f"{quant_99} min", metric="99%", national=f"National 99th Percentile: {quant_99_all} min", color=quant_99_color), unsafe_allow_html=True)

st.markdown("___")
st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------------------------------------------

def get_airline_code(airline):
    if airline == "Alaska Airlines": return "AS"
    if airline == "Allegiant Air": return "G4"
    if airline == "American Airlines": return "AA"
    if airline == "Delta Air Lines": return "DL"
    if airline == "Endeavor Air": return "9E"
    if airline == "Envoy Air": return "MQ"
    if airline == "Frontier Airlines": return "F9"
    if airline == "Hawaiian Airlines": return "HA"
    if airline == "JetBlue Airways": return "B6"
    if airline == "PSA Airlines": return "OH"
    if airline == "Republic Airline": return "YX"
    if airline == "SkyWest Airlines": return "OO"
    if airline == "Southwest Airlines": return "WN"
    if airline == "Spirit Air Lines": return "NK"
    if airline == "United Air Lines": return "UA"
    return ""

month = datetime.datetime.strptime(selected_month.strip(), "%B").month if selected_month != "All Months" else ""
dayofweek = datetime.datetime.strptime(selected_day.strip(), "%A").isoweekday() if selected_day != "All Days" else ""
origin = selected_origin[:3] if selected_origin != "All Airports" else ""
dest = selected_destination[:3] if selected_destination != "All Airports" else ""
reporting_airline = get_airline_code(selected_airline)
dep_hour =  datetime.datetime.strptime(selected_hour.strip().upper(), "%I:%M %p").hour if selected_hour != "All Hours" else ""

st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:1.7em;">Prediction</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">Likelihood of a 30+ minute delay</span>
    </div><br>
    """, unsafe_allow_html=True)

pred_style = f"""
<div style="
    background-color: {{color}};
    border-radius:10px;
    padding:15px;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    min-height:100px;
    width:100%;
    word-wrap:break-word;
    overflow-wrap:break-word;
    margin-bottom: 15px;
">
    <p style="
        color: white;
        font-size:max(0.8vw, 1.4em);
        font-weight:350;
        margin:0;
        line-height:1.5em;
        text-align:center;
        word-break:break-word;
    ">{{pred}}</p>
</div>
"""

input_data = pd.DataFrame({
        "month": [month],
        "dayofweek": [dayofweek],
        "origin": [origin.strip().upper()],
        "dest": [dest.strip().upper()],
        "reporting_airline": [reporting_airline.strip().upper()],
        "dep_hour": [dep_hour]
    })

prediction = pipeline.predict(input_data)[0]
try:
        pred_prob = pipeline.predict_proba(input_data)[0][1]
except Exception:
        pred_prob = None

def interpret_probability(prob, thresholds):
    if prob is None:
        return "No probability available"
    sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
    label = sorted_thresholds[0][0]
    for name, value in sorted_thresholds:
        if prob >= value:
            label = name
    return label

fields = [month, dayofweek, origin, dest, reporting_airline, dep_hour]

if any(x is not None and x != "" for x in fields):
    delay_label = interpret_probability(pred_prob, thresholds)
else: 
    delay_label = "Select at least one filter to view delay prediction"


# if all([month, dayofweek, origin, dest, reporting_airline, dep_hour]):

#     input_data = pd.DataFrame({
#         "month": [month],
#         "dayofweek": [dayofweek],
#         "origin": [origin.strip().upper()],
#         "dest": [dest.strip().upper()],
#         "reporting_airline": [reporting_airline.strip().upper()],
#         "dep_hour": [dep_hour]
#     })

#     prediction = pipeline.predict(input_data)[0]
#     try:
#         pred_prob = pipeline.predict_proba(input_data)[0][1]
#     except Exception:
#         pred_prob = None

#     def interpret_probability(prob, thresholds):
#         if prob is None:
#             return "No probability available"
#         sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
#         label = sorted_thresholds[0][0]
#         for name, value in sorted_thresholds:
#             if prob >= value:
#                 label = name
#         return label

#     delay_label = interpret_probability(pred_prob, thresholds)

# else:
#     delay_label = "Fill in all fields to see the delay prediction"

if delay_label == "Delay Very Unlikely":
    pred_color = my_green
elif delay_label == "Delay Unlikely":
    pred_color = my_lime
elif delay_label == "Delay Somewhat Likely":
    pred_color = my_yellow
elif delay_label == "Delay Likely":
    pred_color = my_orange
elif delay_label == "Delay Very Likely":
    pred_color = my_red
else:
    pred_color = my_grey

spacer_left, col1, spacer_right = st.columns([2, 21, 2], gap="small")

with col1: st.markdown(pred_style.format(pred=f"{delay_label}", color=pred_color), unsafe_allow_html=True)
