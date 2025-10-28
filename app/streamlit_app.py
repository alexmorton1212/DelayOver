
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# --------------------------------------------------------------------------------------------------------
# PRE-PROCESSING
# --------------------------------------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')

# Load data
df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, 'summary_dataset.parquet'))


# --------------------------------------------------------------------------------------------------------
# COLORS
# --------------------------------------------------------------------------------------------------------

my_green = "#3AA655"
my_yellow = "#CBA135"
my_red = "#B04C4C"


# --------------------------------------------------------------------------------------------------------
# TITLE
# --------------------------------------------------------------------------------------------------------

# Set page layout
st.set_page_config(layout="wide")

### App title
st.markdown(f"""
<div style="text-align:center;">
    <span style="font-weight:bold; font-size:3em;">DelayOver</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------------
# DROPDOWNS
# --------------------------------------------------------------------------------------------------------

st.markdown("""<style>.centered-label {text-align: center !important;}</style>""", unsafe_allow_html=True)

with st.container():
    left, col1, col2, col3, col4, col5, right = st.columns([3, 5, 5, 5, 5, 5, 3], gap="small")
    with col1:
        st.markdown('<div class="centered-label">Airport (From)</div>', unsafe_allow_html=True)
        origin_options = ["All Airports"] + sorted(df['origin_ui'].unique().tolist())
        selected_origin = st.selectbox("", origin_options, key="dd1")
    with col2:
        st.markdown('<div class="centered-label">Airport (To)</div>', unsafe_allow_html=True)
        destination_options = ["All Airports"] + sorted(df['destination_ui'].unique().tolist())
        selected_destination = st.selectbox("", destination_options, key="dd2")
    with col3:
        st.markdown('<div class="centered-label">Airline</div>', unsafe_allow_html=True)
        airline_options = ["All Airlines"] + sorted(df['airline_ui'].unique().tolist())
        selected_airline = st.selectbox("", airline_options, key="dd3")
    with col4:
        st.markdown('<div class="centered-label">Date</div>', unsafe_allow_html=True)
        st.selectbox("", ["All Dates"] + ["Option A", "Option B", "Option C"], key="dd4")
    with col5:
        st.markdown('<div class="centered-label">Time</div>', unsafe_allow_html=True)
        st.selectbox("", ["All Times"] + ["Option A", "Option B", "Option C"], key="dd5")


# --------------------------------------------------------------------------------------------------------
# APPLY DROPDOWN FILTERS
# --------------------------------------------------------------------------------------------------------

filtered_df = df.copy()

if selected_origin != "All Airports":
    filtered_df = filtered_df[filtered_df["origin_ui"] == selected_origin]
if selected_destination != "All Airports":
    filtered_df = filtered_df[filtered_df["destination_ui"] == selected_destination]
if selected_airline != "All Airlines":
    filtered_df = filtered_df[filtered_df["airline_ui"] == selected_airline]

st.markdown("<br><br>", unsafe_allow_html=True)

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
with col1:
    st.markdown(card_style.format(title="Total", metric=f"{total:,}"), unsafe_allow_html=True)
with col2:
    st.markdown(card_style.format(title="Delayed", metric=f"{delayed:,}"), unsafe_allow_html=True)
with col3:
    st.markdown(card_style.format(title="Cancelled", metric=f"{cancelled:,}"), unsafe_allow_html=True)
with col4:
    st.markdown(card_style.format(title="Diverted", metric=f"{diverted:,}"), unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------
# ON-TIME PERCENTAGE
# --------------------------------------------------------------------------------------------------------

if filtered_df.empty:
    total = 0
    delayed_diverted_cancelled = 0
    on_time = 0
    on_time_percent = 0
    delayed_percent = 100 - on_time_percent
else:
    total = len(filtered_df)
    delayed_diverted_cancelled = filtered_df["if_delay"].sum() + filtered_df["if_diverted"].sum() + filtered_df["if_cancelled"].sum()
    on_time = total - delayed_diverted_cancelled
    on_time_percent = round(100 * (on_time / total)) if total > 0 else 0
    delayed_percent = 100 - on_time_percent

if filtered_df.empty:
    colors = ["white", "#C7C7C7"]
elif on_time_percent > 78:
    colors = [my_green, "#C7C7C7"]
elif on_time_percent > 65:
    colors = [my_yellow, "#C7C7C7"]
else:
    colors = [my_red, "#C7C7C7"] 


chart_df = pd.DataFrame({"Category": ["On-Time", "Delayed, Diverted, or Cancelled"], "Value": [on_time, delayed_diverted_cancelled]})
if chart_df["Value"].sum() > 0:
    chart_df["Percent"] = (chart_df["Value"] / chart_df["Value"].sum() * 100).round().astype(int)
else:
    chart_df["Percent"] = [0, 0]
chart_df["Label"] = ''
chart_df["Y"] = "All"

if filtered_df.empty:
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-weight:bold; font-size:2em;">On-Time Percentage: NA</span>
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
        <span style="font-weight:bold; font-size:2em;">On-Time Percentage: </span>
        <span style="color: {colors[0]}; font-weight:bold; font-size:2em;">{on_time_percent}%</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="font-size:1em;">Percentage of flights arriving within 15 minutes of scheduled arrival</span>
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
    height=120,
    margin=dict(l=10, r=10, t=10, b=10)
)

# Set hovertemplate per trace
for trace in fig.data:
    trace.hovertemplate = f"<b>{trace.name}</b><br>%{{x}}%<extra></extra>"

# Center chart in Streamlit
col1, col2, col3 = st.columns([4, 12, 3])
with col2:
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------------------------------------
# PERCENTILE CARD FIGURES
# --------------------------------------------------------------------------------------------------------

st.markdown(f"""
<div style="text-align:center;">
    <span style="font-weight:bold; font-size:2em;">Arrival Times</span>
</div><br>
""", unsafe_allow_html=True)

# Card styling
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
    ">{{metric}} of flights arrive within</p>
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
"""

spacer_left, col1, col2, col3, spacer_right = st.columns([2, 7, 7, 7, 2], gap="small")

# Example metrics
with col1:
    st.markdown(card_style.format(title="27 min", metric="90%", national="National 90th Percentile: 36 min", color=my_green), unsafe_allow_html=True)

with col2:
    st.markdown(card_style.format(title="42 min", metric="95%", national="National 95th Percentile: 36 min", color=my_red), unsafe_allow_html=True)

with col3:
    st.markdown(card_style.format(title="232 min", metric="99%", national="National 99th Percentile: 36 min", color=my_yellow), unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)