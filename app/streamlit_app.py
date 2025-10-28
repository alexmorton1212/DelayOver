
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

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

# --- Custom CSS to center selectbox labels ---
st.markdown("""
<style>
.centered-label {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# --- Container for dropdowns ---
with st.container():
    left, col1, col2, col3, col4, col5, right = st.columns([3, 5, 5, 5, 5, 5, 3], gap="medium")

    with col1:
        st.markdown('<div class="centered-label">Airport (From)</div>', unsafe_allow_html=True)
        st.selectbox("", [""] + ["Option A", "Option B", "Option C"], key="dd1")
    with col2:
        st.markdown('<div class="centered-label">Airport (To)</div>', unsafe_allow_html=True)
        st.selectbox("", [""] + ["Option A", "Option B", "Option C"], key="dd2")
    with col3:
        st.markdown('<div class="centered-label">Airline</div>', unsafe_allow_html=True)
        st.selectbox("", [""] + ["Option A", "Option B", "Option C"], key="dd3")
    with col4:
        st.markdown('<div class="centered-label">Date</div>', unsafe_allow_html=True)
        st.selectbox("", [""] + ["Option A", "Option B", "Option C"], key="dd4")
    with col5:
        st.markdown('<div class="centered-label">Time</div>', unsafe_allow_html=True)
        st.selectbox("", [""] + ["Option A", "Option B", "Option C"], key="dd5")


st.markdown("<br><br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------
# COUNT CARD FIGURES
# --------------------------------------------------------------------------------------------------------

# Card styling
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

# Example metrics
with col1:
    st.markdown(card_style.format(title="Total", metric="2,800,312"), unsafe_allow_html=True)

with col2:
    st.markdown(card_style.format(title="Delayed", metric="677,422"), unsafe_allow_html=True)

with col3:
    st.markdown(card_style.format(title="Diverted", metric="4,068"), unsafe_allow_html=True)

with col4:
    st.markdown(card_style.format(title="Cancelled", metric="10,068"), unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------
# ON-TIME / DELAY PERCENTAGE FIGURES
# --------------------------------------------------------------------------------------------------------

# Test data
df = pd.DataFrame({"Category": ["On-Time", "Delayed, Diverted, or Cancelled"], "Value": [2800312, 677422]})
df["Percent"] = (df["Value"] / df["Value"].sum() * 100).round().astype(int)
df["Label"] = ''
# df["Label"] = df.apply(lambda x: f"{x['Category']}", axis=1)
df["Y"] = "All"

st.markdown(f"""
<div style="text-align:center;">
    <span style="font-weight:bold; font-size:2em;">On-Time Percentage: </span>
    <span style="color: #2ECC71; font-weight:bold; font-size:2em;">77%</span>
</div>
""", unsafe_allow_html=True)


st.markdown(f"""
<div style="text-align:center;">
    <span style="color: grey; font-size:1em;">Percentage of flights arriving within 15 minutes of scheduled arrival</span>
</div>
""", unsafe_allow_html=True)

colors = ["#2ECC71", "#c7c7c7"]
# colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]

# Build horizontal stacked bar
fig = px.bar(
    df,
    x="Percent",
    y="Y",
    color="Category",
    orientation="h",
    text="Label",  # shows percentage inside the bar
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
    height=max(120, 80 * len(df)),
    margin=dict(l=10, r=10, t=10, b=10)
)

# Set hovertemplate per trace
for trace in fig.data:
    trace.hovertemplate = f"<b>{trace.name}</b><br>%{{x}}%<extra></extra>"


# Center chart in Streamlit
col1, col2, col3 = st.columns([2, 24, 1])
with col2:
    st.plotly_chart(fig, use_container_width=True)

# st.markdown("<br>", unsafe_allow_html=True)


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
    st.markdown(card_style.format(title="27 min", metric="90%", national="National 90th Percentile: 36 min", color="#3AA655"), unsafe_allow_html=True)

with col2:
    st.markdown(card_style.format(title="42 min", metric="95%", national="National 95th Percentile: 36 min", color="#CBA135"), unsafe_allow_html=True)

with col3:
    st.markdown(card_style.format(title="232 min", metric="99%", national="National 99th Percentile: 36 min", color="#B04C4C"), unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# # Test data
# df = pd.DataFrame({"Category": ["On-Time", "Delayed", "Diverted or Cancelled"], "Value": [2800312, 677422, 140680]})
# df["Percent"] = (df["Value"] / df["Value"].sum() * 100).round().astype(int)
# df["Label"] = ''
# # df["Label"] = df.apply(lambda x: f"{x['Category']}", axis=1)
# df["Y"] = "All"

# st.markdown(f"""
# <div style="text-align:center;">
#     <span style="font-weight:bold; font-size:2em;">On-Time Percentage: </span>
#     <span style="color: #2ECC71; font-weight:bold; font-size:2em;">77%</span>
# </div>
# """, unsafe_allow_html=True)


# st.markdown(f"""
# <div style="text-align:center;">
#     <span style="color: grey; font-size:1em;">Percentage of flights arriving within 15 minutes of scheduled arrival</span>
# </div>
# """, unsafe_allow_html=True)

# colors = ["#2ECC71", "#F1C40F", "#E74C3C"]
# # colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]

# # Build horizontal stacked bar
# fig = px.bar(
#     df,
#     x="Percent",
#     y="Y",
#     color="Category",
#     orientation="h",
#     text="Label",  # shows percentage inside the bar
#     color_discrete_sequence=colors
# )

# # Style chart
# fig.update_layout(
#     barmode="stack",
#     showlegend=False,
#     legend_title_text=None,
#     xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
#     xaxis_title="",
#     yaxis=dict(showticklabels=False, showgrid=False, visible=False),
#     height=max(120, 50 * len(df)),
#     margin=dict(l=10, r=10, t=10, b=10),
#     # plot_bgcolor="#f5f5f5",
#     # paper_bgcolor="#e0e0e0",
# )

# # Set hovertemplate per trace
# for trace in fig.data:
#     trace.hovertemplate = f"<b>{trace.name}</b><br>%{{x}}%<extra></extra>"


# # Center chart in Streamlit
# col1, col2, col3 = st.columns([2, 24, 1])
# with col2:
#     st.plotly_chart(fig, use_container_width=True)







# --------------------------------------------------------------------------------------------------------
# PERCENTILE CARD FIGURES
# --------------------------------------------------------------------------------------------------------

# st.markdown(f"""
# <div style="text-align:center;">
#     <span style="font-weight:bold; font-size:2em;">Arrival Times</span>
# </div>
# """, unsafe_allow_html=True)

# # Card styling
# card_style = f"""

#     <p style="
#         color: #2ECC71;
#         font-size:max(1.5vw, 2em);
#         font-weight:bold;
#         margin:0;
#         line-height:2em;
#         text-align:center;
#         word-break:break-word;
#     ">{{title}}</p>

# <div style="
#     background-color: #5D8199;
#     border-radius:10px;
#     padding:15px;
#     box-shadow:0 2px 10px rgba(0,0,0,0.1);
#     display:flex;
#     flex-direction:column;
#     justify-content:center;
#     align-items:center;
#     text-align:center;
#     min-height:100px;
#     width:100%;
#     word-wrap:break-word;
#     overflow-wrap:break-word;
#     margin-bottom: 15px;
# ">
#     <p style="
#         color: white;
#         font-size:max(0.8vw, 1.2em);
#         font-weight:bold;
#         margin:0;
#         line-height:1.2em;
#         text-align:center;
#         word-break:break-word;
#     ">{{metric}}</p>
# </div>
#     <p style="
#         color: grey;
#         font-size:max(0.5vw, 0.8em);
#         margin:0;
#         line-height:1em;
#         text-align:center;
#         word-break:break-word;
#     ">{{national}}</p><br>


# """

# spacer_left, col1, col2, col3, spacer_right = st.columns([2, 7, 7, 7, 2], gap="small")

# # Example metrics
# with col1:
#     st.markdown(card_style.format(title="27 min", metric="90% of flights arrive within 27 minutes of scheduled arrival", national="National 90th Percentile: 36 min"), unsafe_allow_html=True)

# with col2:
#     st.markdown(card_style.format(title="42 min", metric="95% of flights arrive within 42 minutes of scheduled arrival", national="National 95th Percentile: 36 min"), unsafe_allow_html=True)

# with col3:
#     st.markdown(card_style.format(title="79 min", metric="99% of flights arrive within 79 minutes of scheduled arrival", national="National 99th Percentile: 36 min"), unsafe_allow_html=True)

# st.markdown("<br><br>", unsafe_allow_html=True)
