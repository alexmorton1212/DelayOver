
import os
import joblib
import streamlit as st
import pandas as pd

# --------------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "summary_dataset.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")

# --------------------------------------------------------------------------------------------------------
# Summary Dataframe
# --------------------------------------------------------------------------------------------------------

df = pd.read_parquet(DATA_PATH)

# --------------------------------------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = joblib.load(file)

# --------------------------------------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------------------------------------

### App title
st.title("Flight Delay Prediction Dashboard")

### Dropdown for Airlines
available_airlines = sorted(df['reporting_airline'].unique())
selected_airline = st.selectbox("Select Airline", available_airlines)

### Dropdown for Origin Airports
origin_airports = sorted(df['origin'].unique())
selected_origin = st.selectbox("Select Origin Airport", origin_airports)

### Dropdown for Destination Airports
destination_airports = sorted(df['dest'].unique())
selected_dest = st.selectbox("Select Destination Airport", destination_airports)

### Date selector
selected_date = st.date_input("Select Flight Date")
selected_month = selected_date.month
selected_dayofweek = selected_date.weekday() + 1

### Time selector (hour and minute)
selected_time = st.time_input("Select Flight Time")
selected_dephour = selected_time.hour

### Holiday proximity bucket (based on date)
selected_holiday_bucket = 5

### Inputs for model prediction
model_input = {
    'month': [selected_month],
    'dayofweek': [selected_dayofweek],
    'origin': [selected_origin],
    'dest': [selected_dest],
    'reporting_airline': [selected_airline],
    'dep_hour': [selected_dephour],
    'holiday_proximity_bucket': [selected_holiday_bucket]
}

# Delay Categories
def get_delay_label(prob):
    if prob <= 0.25: return "Delay Very Unlikely"
    elif prob <= 0.35: return "Delay Unlikely"
    elif prob <= 0.45: return "Delay Somewhat Likely"
    elif prob <= 0.6: return "Delay Likely"
    elif prob <= 0.75: return "Delay Very Likely"
    else: return "Delay Almost Certain (Yikes)"

X_test = pd.DataFrame(model_input)
prob = model.predict_proba(X_test)[0][1]
label = get_delay_label(prob)

st.write(f"Predicted probability of delay: {prob:.3f}")
st.write(f"{label}")






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
