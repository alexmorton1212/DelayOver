
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import streamlit as st
import plotly.graph_objects as go

# Example value
value = 0.9
anti_value = 1 - value
percent = value * 100
anti_percent = anti_value * 100

# Dynamic color
color = (
    "#E82E2E" if value <= 0.25 else
    "#E89A2E" if value <= 0.45 else
    "#EDE54C" if value <= 0.6 else
    "#11BD27"
)

# Create the top half donut
fig_ontime_perc = go.Figure(data=[
    go.Pie(
        values=[value, 1 - value, 1],  # filled, unfilled, invisible (bottom)
        rotation=270,  # start at top center
        hole=0.65,
        direction="clockwise",
        marker=dict(colors=[color, "#e8e8e8", "rgba(0,0,0,0)"]),
        textinfo="none",
        hoverinfo="skip",
        sort=False
    )
]).update_layout(
    showlegend=False,
    margin=dict(t=0, b=0, l=0, r=0),
    height=275,
    annotations=[
        dict(
            text=f"<b>{percent:.0f}%</b>",
            x=0.5, y=0.6,  # centered at bottom of the half-donut
            font_size=38,
            showarrow=False
        ),
        dict(
            text="On-Time Arrivals",
            x=0.5, y=0.35,  # a bit lower
            font_size=28,
            showarrow=False
        ),
        dict(
            text="Percentage of flights arriving within 15",
            x=0.5, y=0.20,  # a bit lower
            font_size=14,
            font_color="gray",
            showarrow=False
        ),
        dict(
            text="minutes of scheduled arrival",
            x=0.5, y=0.14,  # a bit lower
            font_size=14,
            font_color="gray",
            showarrow=False
        )
    ]
)

# Example counts
counts = [30, 50, 20]  # counts for A, B, C
labels = ["A", "B", "C"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # customize as needed

# Create top half donut
fig_delay_perc = go.Figure(data=[
    go.Pie(
        values=counts + [sum(counts)],  # last part invisible bottom half
        rotation=270,  # start at top center
        hole=0.65,
        direction="clockwise",
        marker=dict(colors=colors + ["rgba(0,0,0,0)"]),
        textinfo="none",
        hoverinfo="label+percent",
        sort=False
    )
]).update_layout(
    showlegend=False,
    margin=dict(t=0, b=0, l=0, r=0),
    height=275,
    annotations=[
        dict(
            text=f"<b>{anti_percent:.0f}%</b>",
            x=0.5, y=0.6,  # centered at bottom of the half-donut
            font_size=38,
            showarrow=False
        ),
        dict(
            text="Non-Weather Delays",
            x=0.5, y=0.35,  # a bit lower
            font_size=28,
            font_color="gray",
            showarrow=False
        ),
        dict(
            text="Percentage of flights delayed by factors other",
            x=0.5, y=0.20,  # a bit lower
            font_size=14,
            font_color="gray",
            showarrow=False
        ),
        dict(
            text="than weather (late, airline, etc)",
            x=0.5, y=0.14,  # a bit lower
            font_size=14,
            font_color="gray",
            showarrow=False
        )
    ]
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_ontime_perc, use_container_width=True)
with col2:
    st.plotly_chart(fig_delay_perc, use_container_width=True)






### USE THIS LATER

# # --------------------------------------------------------------------------------------------------------
# # Paths
# # --------------------------------------------------------------------------------------------------------

# DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "summary_dataset.parquet")
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")

# # --------------------------------------------------------------------------------------------------------
# # Summary Dataframe
# # --------------------------------------------------------------------------------------------------------

# df = pd.read_parquet(DATA_PATH)

# # --------------------------------------------------------------------------------------------------------
# # Model Loading
# # --------------------------------------------------------------------------------------------------------

# with open(MODEL_PATH, "rb") as file:
#     model = joblib.load(file)

# # --------------------------------------------------------------------------------------------------------
# # Streamlit app
# # --------------------------------------------------------------------------------------------------------

# ### App title
# st.title("Flight Delay Prediction Dashboard")

# ### Dropdown for Airlines
# available_airlines = sorted(df['reporting_airline'].unique())
# selected_airline = st.selectbox("Select Airline", available_airlines)

# ### Dropdown for Origin Airports
# origin_airports = sorted(df['origin'].unique())
# selected_origin = st.selectbox("Select Origin Airport", origin_airports)

# ### Dropdown for Destination Airports
# destination_airports = sorted(df['dest'].unique())
# selected_dest = st.selectbox("Select Destination Airport", destination_airports)

# ### Date selector
# selected_date = st.date_input("Select Flight Date")
# selected_month = selected_date.month
# selected_dayofweek = selected_date.weekday() + 1

# ### Time selector (hour and minute)
# selected_time = st.time_input("Select Flight Time")
# selected_dephour = selected_time.hour

# ### Holiday proximity bucket (based on date)
# selected_holiday_bucket = 5

# ### Inputs for model prediction
# model_input = {
#     'month': [selected_month],
#     'dayofweek': [selected_dayofweek],
#     'origin': [selected_origin],
#     'dest': [selected_dest],
#     'reporting_airline': [selected_airline],
#     'dep_hour': [selected_dephour],
#     'holiday_proximity_bucket': [selected_holiday_bucket]
# }

# # Delay Categories
# def get_delay_label(prob):
#     if prob <= 0.25: return "Delay Very Unlikely"
#     elif prob <= 0.35: return "Delay Unlikely"
#     elif prob <= 0.45: return "Delay Somewhat Likely"
#     elif prob <= 0.6: return "Delay Likely"
#     elif prob <= 0.75: return "Delay Very Likely"
#     else: return "Delay Almost Certain (Yikes)"

# X_test = pd.DataFrame(model_input)
# prob = model.predict_proba(X_test)[0][1]
# label = get_delay_label(prob)

# st.write(f"Predicted probability of delay: {prob:.3f}")
# st.write(f"{label}")






# print("\n=== Model type ===")
# print(type(model))

# # If it's a pipeline, list the steps
# if hasattr(model, "named_steps"):
#     print("\n=== Pipeline steps ===")
#     for name, step in model.named_steps.items():
#         print(f"- {name}: {type(step)}")

#     # If there's a preprocessor, explore it
#     preprocessor = model.named_steps.get("preprocessor", None)
#     if preprocessor is not None:
#         print("\n=== Preprocessor details ===")
#         print(preprocessor)

#         # Try to print feature names if available
#         try:
#             feature_names = preprocessor.get_feature_names_out()
#             print("\n=== Feature names after preprocessing ===")
#             for f in feature_names:
#                 print(f)
#         except AttributeError:
#             print("\n[!] Preprocessor does not expose feature names directly.")
# else:
#     print("\n[!] This model is not a Pipeline object.")

# # Some sklearn models (1.0+) also store original feature names
# if hasattr(model, "feature_names_in_"):
#     print("\n=== Original training feature names ===")
#     print(model.feature_names_in_)
# else:
#     print("\n[!] No 'feature_names_in_' attribute found.")

# print("\n=== Done inspecting model ===")
