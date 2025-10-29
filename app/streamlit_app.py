
import os
import joblib # for future ML portion
import calendar
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# --------------------------------------------------------------------------------------------------------
# PRE-PROCESSING
# --------------------------------------------------------------------------------------------------------

# Set page layout
st.set_page_config(layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'summary_dataset.parquet')

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"Data file not found at {path}")
        st.stop()
    return pd.read_parquet(path)

df = load_data(DATA_FILE)

# --------------------------------------------------------------------------------------------------------
# COLORS
# --------------------------------------------------------------------------------------------------------

my_green = "#3AA655"
my_yellow = "#CBA135"
my_red = "#B04C4C"


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

# st.markdown("<br>", unsafe_allow_html=True)


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


# --------------------------------------------------------------------------------------------------------
# FILTER SUMMARY (UX)
# --------------------------------------------------------------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True) # comment if using filter summary below

# st.markdown("<hr style='border:1px solid;'>", unsafe_allow_html=True)

# origin_summary = f' from <b>{selected_origin.split(" (")[0]}</b>' if selected_origin != "All Airports" else ''
# destination_summary = f' to <b>{selected_destination.split(" (")[0]}</b>' if selected_destination != "All Airports" else ''
# airline_summary = f' with <b>{selected_airline}</b>' if selected_airline != "All Airlines" else ''
# month_summary = f' in <b>{selected_month}</b>' if selected_month != "All Months" else ''
# day_summary = f' on <b>{selected_day}s</b> ' if selected_day != "All Days" else ''
# hour_ub = selected_hour.split(":")[0] + ":59 " + selected_hour.split(" ")[1]
# hour_summary = f' departing between <b>{selected_hour}</b> and <b>{hour_ub}</b>' if selected_hour != "All Hours" else ''

# filter_summary = 'Flights' + origin_summary + destination_summary + airline_summary + day_summary + month_summary + hour_summary
# filter_ux = (
#     'No matching data was found with current filters'
#     if filtered_df.empty
#     else 'All Flights from July 2024 - June 2025'
#     if filtered_df.equals(df)
#     else filter_summary
# )
# st.markdown(f"""
#     <div style="text-align:center;">
#         <span style="font-size:1.3em; line-height: 1em;">{filter_ux}</span>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<hr style='border:1px solid;'>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------------
# COUNT CARD FIGURES
# --------------------------------------------------------------------------------------------------------

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
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------------
# PERCENTILE CARD FIGURES
# --------------------------------------------------------------------------------------------------------

def safe_quantile(series, q):
    val = series.quantile(q)
    return int(val) if pd.notnull(val) else 0

quant_90_all = safe_quantile(df['arrdelayminutes'], 0.90)
quant_95_all = safe_quantile(df['arrdelayminutes'], 0.95)
quant_99_all = safe_quantile(df['arrdelayminutes'], 0.99)

quant_90 = safe_quantile(filtered_df['arrdelayminutes'], 0.90)
quant_95 = safe_quantile(filtered_df['arrdelayminutes'], 0.95)
quant_99 = safe_quantile(filtered_df['arrdelayminutes'], 0.99)

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
    ">{{national}}</p>
"""

spacer_left, col1, col2, col3, spacer_right = st.columns([2, 7, 7, 7, 2], gap="small")

with col1: st.markdown(card_style.format(title=f"{quant_90} min", metric="90%", national=f"National 90th Percentile: {quant_90_all} min", color=quant_90_color), unsafe_allow_html=True)
with col2: st.markdown(card_style.format(title=f"{quant_95} min", metric="95%", national=f"National 95th Percentile: {quant_95_all} min", color=quant_95_color), unsafe_allow_html=True)
with col3: st.markdown(card_style.format(title=f"{quant_99} min", metric="99%", national=f"National 99th Percentile: {quant_99_all} min", color=quant_99_color), unsafe_allow_html=True)