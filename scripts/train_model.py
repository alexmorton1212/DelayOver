
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(script_dir, '..', 'data', 'processed')

df_og = pd.read_parquet(processed_data_dir + '/ml_dataset.parquet')

# Take a random 10% sample (with fixed seed for reproducibility)
df = df_og.sample(frac=0.01, random_state=42)

target = 'arrdelayminutes'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

categorical_features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline']
numeric_features = ['deptime_mins', 'arrtime_mins', 'days_from_holiday']
binary_features = ['if_near_holiday']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features),
        ('bin', 'passthrough', binary_features)
    ])

regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))