import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ---------------------------------------------
# Load data
# ---------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(script_dir, '..', 'data', 'processed')

df = pd.read_parquet(os.path.join(processed_data_dir, 'ml_dataset.parquet'))
df_final = df.sample(frac=0.5, random_state=42)

# ---------------------------------------------
# Prepare features & target
# ---------------------------------------------
features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 'holiday_proximity_bucket']
target = 'arrdelayminutes'

X = df_final[features]
y = (df_final[target] > 30).astype(int)  # Class 1 = delay over 30 mins

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

categorical_features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'holiday_proximity_bucket']
numeric_features = ['dep_hour']

# ---------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# ---------------------------------------------
# XGBoost model with initial parameters
# ---------------------------------------------

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=10,  # class imbalance
    random_state=42,
    n_jobs=-1
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb)
])

# model = pipeline
# model.fit(X_train, y_train)

# ---------------------------------------------
# Light hyperparameter tuning
# ---------------------------------------------

param_grid = {
    'classifier__max_depth': [4, 5, 6],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__n_estimators': [100],
    'classifier__scale_pos_weight': [2, 3]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='recall',  # focuses on capturing class 1
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Use best model from tuning
model = grid_search.best_estimator_

# ---------------------------------------------
# Predict and apply threshold
# ---------------------------------------------

threshold = 0.25  # adjust for desired false positive / false negative trade-off
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

# ---------------------------------------------
# Evaluation
# ---------------------------------------------
print(f"\nClassification Report (threshold = {threshold:.2f}):")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)
fn_rate = 1 - recall

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"FN Rate:   {fn_rate:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

# ---------------------------------------------
# How many were predicted as NOT delayed
# ---------------------------------------------
n_total = len(y_pred)
n_predicted_not_delayed = np.sum(y_pred == 0)
percent_not_delayed = n_predicted_not_delayed / n_total * 100

print(f"\nFlights predicted as NOT DELAYED: {n_predicted_not_delayed} ({percent_not_delayed:.2f}%)")


# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import xgboost as xgb

# # Paths (adjust as needed)
# script_dir = os.path.dirname(os.path.abspath(__file__))
# processed_data_dir = os.path.join(script_dir, '..', 'data', 'processed')

# # Load data
# df = pd.read_parquet(processed_data_dir + '/ml_dataset.parquet')

# # Use 10% sample
# df_final = df.sample(frac=0.2, random_state=42)

# # Features and target
# features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 'holiday_proximity_bucket']
# target = 'arrdelayminutes'

# X = df_final[features]
# y = (df_final[target] > 30).astype(int)  # Binary target: delay > 30 mins

# # Train-test split with stratify to preserve imbalance ratio
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Define categorical and numeric features
# categorical_features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'holiday_proximity_bucket']
# numeric_features = ['dep_hour']

# # Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#         ('num', 'passthrough', numeric_features)
#     ])

# # Calculate imbalance ratio for scale_pos_weight
# scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# # Initialize XGBoost classifier with imbalance handling
# xgb_clf = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     use_label_encoder=False,
#     n_jobs=-1,
#     random_state=42,
#     scale_pos_weight=scale_pos_weight,
#     verbosity=1
# )

# # Create pipeline
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', xgb_clf)
# ])

# # Train the model
# model_pipeline.fit(X_train, y_train)

# # Predict probabilities on test set
# y_probs = model_pipeline.predict_proba(X_test)[:, 1]

# # Apply adjusted threshold (e.g., 0.2) to increase recall
# threshold = 0.35
# y_pred_adj = (y_probs >= threshold).astype(int)

# # Evaluate
# print("Classification Report (threshold = {:.2f}):\n".format(threshold), classification_report(y_test, y_pred_adj))

# accuracy = accuracy_score(y_test, y_pred_adj)
# precision = precision_score(y_test, y_pred_adj)
# recall = recall_score(y_test, y_pred_adj)
# f1 = f1_score(y_test, y_pred_adj)
# roc_auc = roc_auc_score(y_test, y_probs)

# print(f"Accuracy:  {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall:    {recall:.4f}")
# print(f"FN Rate:   {1-recall:.4f}")
# print(f"F1 Score:  {f1:.4f}")
# print(f"ROC AUC:   {roc_auc:.4f}")


# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


# script_dir = os.path.dirname(os.path.abspath(__file__))
# processed_data_dir = os.path.join(script_dir, '..', 'data', 'processed')

# df = pd.read_parquet(processed_data_dir + '/ml_dataset.parquet')
# df_final = df.sample(frac=0.1, random_state=42)


# # Assuming df_final is your DataFrame and already cleaned/filtered
# # Features and target
# features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 'holiday_proximity_bucket']
# target = 'arrdelayminutes'

# X = df_final[features]

# # Create binary target: 1 if delay > 30 mins else 0
# y = (df_final[target] > 30).astype(int)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Define categorical and numeric features
# categorical_features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'holiday_proximity_bucket']
# numeric_features = ['dep_hour']

# # Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#         ('num', 'passthrough', numeric_features)
#     ])

# # Model pipeline
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
# ])

# # Train the classifier
# model_pipeline.fit(X_train, y_train)

# # Predict on test set
# y_pred = model_pipeline.predict(X_test)
# y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# # Evaluate
# print("Classification Report:\n", classification_report(y_test, y_pred))

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_proba)

# print(f"Accuracy:  {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall:    {recall:.4f}")
# print(f"F1 Score:  {f1:.4f}")
# print(f"ROC AUC:   {roc_auc:.4f}")