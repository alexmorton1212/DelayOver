
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ---------------------------------------------
# Load data function
# ---------------------------------------------

def load_data(processed_data_dir, frac=0.5, random_state=42):
    df = pd.read_parquet(os.path.join(processed_data_dir, 'ml_dataset.parquet'))
    df_sampled = df.sample(frac=frac, random_state=random_state)
    return df_sampled

# ---------------------------------------------
# Preprocessing pipeline function
# ---------------------------------------------

def get_preprocessor(categorical_features, numeric_features):
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

# ---------------------------------------------
# Evaluate metrics function
# ---------------------------------------------

def evaluate_metrics(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    support_0 = tn + fp
    support_1 = tp + fn

    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    roc_auc = roc_auc_score(y_true, y_probs)
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    percent_predicted_not_delayed = np.sum(y_pred == 0) / len(y_pred) * 100

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision_1': precision_1,
        'recall_1': recall_1,
        'f1_1': f1_1,
        'support_1': support_1,
        'precision_0': precision_0,
        'recall_0': recall_0,
        'f1_0': f1_0,
        'support_0': support_0,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'percent_predicted_not_delayed': percent_predicted_not_delayed,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

# ---------------------------------------------
# Cross-validation and evaluation function
# ---------------------------------------------

def cross_validate_models(X, y, param_grid, thresholds, preprocessor, cv_splits=3, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = []

    model_id = 1  # Counter to assign unique labels to parameter combinations

    for max_depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for spw in param_grid['scale_pos_weight']:
                for subsample in param_grid['subsample']:

                    model_label = f"model_{model_id}"
                    model_id += 1

                    # Store metrics for each threshold
                    fold_metrics_per_threshold = {thr: [] for thr in thresholds}

                    for train_idx, val_idx in cv.split(X, y):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                        model = XGBClassifier(
                            objective='binary:logistic',
                            eval_metric='logloss',
                            use_label_encoder=False,
                            n_estimators=200,
                            learning_rate=lr,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=0.9,
                            scale_pos_weight=spw,
                            random_state=random_state,
                            n_jobs=-1
                        )

                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', model)
                        ])

                        pipeline.fit(X_train, y_train)
                        y_probs = pipeline.predict_proba(X_val)[:, 1]

                        # Evaluate across all thresholds
                        for thr in thresholds:
                            metrics = evaluate_metrics(y_val, y_probs, thr)
                            fold_metrics_per_threshold[thr].append(metrics)

                    # Average metrics across folds per threshold
                    for thr in thresholds:
                        avg_metrics = pd.DataFrame(fold_metrics_per_threshold[thr]).mean().to_dict()
                        results.append({
                            'model_label': model_label,
                            'threshold': thr,
                            'max_depth': max_depth,
                            'learning_rate': lr,
                            'scale_pos_weight': spw,
                            'subsample': subsample,
                            **avg_metrics
                        })

    return results


# ---------------------------------------------
# Main execution block
# ---------------------------------------------

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir, '..', 'data', 'processed')

    df = load_data(processed_data_dir)

    features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 'holiday_proximity_bucket']
    target = 'arrdelayminutes'

    X = df[features]
    y = (df[target] > 30).astype(int)

    categorical_features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'holiday_proximity_bucket']
    numeric_features = ['dep_hour']

    preprocessor = get_preprocessor(categorical_features, numeric_features)

    param_grid = {
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [3, 5],
        'subsample': [0.8, 1.0]
    }

    thresholds = [0.15, 0.2, 0.25, 0.3, 0.35]

    results = cross_validate_models(X, y, param_grid, thresholds, preprocessor)

    results_df = pd.DataFrame(results)

    output_path = os.path.join(processed_data_dir, 'xgb_model_cv_eval_results_with_thresholds.csv')
    results_df.to_csv(output_path, index=False)

    print(f"\n✅ Cross-validated results with multiple thresholds saved to:\n{output_path}")


# ---------------------------------------------
# Identify and retrain the best model
# ---------------------------------------------

# Custom scoring: update this line as needed
results_df['custom_score'] = (0.55 * results_df['recall_1']) + (0.05 * results_df['precision_0']) + (0.35 * results_df['recall_0']) + (0.05 * results_df['roc_auc'])

# Select best model row
best_row = results_df.loc[results_df['custom_score'].idxmax()]
best_params = {
    'max_depth': int(best_row['max_depth']),
    'learning_rate': float(best_row['learning_rate']),
    'scale_pos_weight': float(best_row['scale_pos_weight']),
    'subsample': float(best_row['subsample']),
}
best_threshold = float(best_row['threshold'])
best_model_label = best_row['model_label']

print(f"\n🏆 Best Model: {best_model_label} @ threshold {best_threshold} with custom score = {best_row['custom_score']:.4f}")

# Retrain best model on full data
final_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200,
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    subsample=best_params['subsample'],
    colsample_bytree=0.9,
    scale_pos_weight=best_params['scale_pos_weight'],
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', final_model)
])

pipeline.fit(X, y)

# ---------------------------------------------
# Threshold Evaluation: 0.05 to 1.0
# ---------------------------------------------
thresholds_to_test = np.arange(0.05, 1.01, 0.05)
y_probs = pipeline.predict_proba(X)[:, 1]

threshold_metrics = []

for thr in thresholds_to_test:
    y_pred = (y_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    precision_1 = precision_score(y, y_pred, zero_division=0)
    recall_1 = recall_score(y, y_pred, zero_division=0)
    f1_1 = f1_score(y, y_pred, zero_division=0)

    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    roc_auc = roc_auc_score(y, y_probs)
    fp_rate = fp / (fp + tn) if (fp + tn) else 0
    fn_rate = fn / (fn + tp) if (fn + tp) else 0
    percent_predicted_not_delayed = np.sum(y_pred == 0) / len(y_pred) * 100

    threshold_metrics.append({
        "threshold": round(thr, 2),
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "percent_predicted_not_delayed": percent_predicted_not_delayed,
        "roc_auc": roc_auc,
        "accuracy": (tp + tn) / (tp + tn + fp + fn)
    })

threshold_df = pd.DataFrame(threshold_metrics)

output_path = os.path.join(processed_data_dir, 'best_model_thresholds.csv')
threshold_df.to_csv(output_path, index=False)

# ---------------------------------------------
# Save model & metadata
# ---------------------------------------------
models_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'final_model.pkl')
metadata_path = os.path.join(models_dir, 'model_metadata.json')
log_path = os.path.join(models_dir, 'model_training_log.csv')

# Save model
joblib.dump(pipeline, model_path)

# Save metadata
metadata = {
    "model_label": best_model_label,
    "selected_threshold": best_threshold,
    "parameters": best_params,
    "custom_score": best_row['custom_score'],
    "thresholds": {
        "unlikely": 0.0,
        "somewhat_likely": 0.3,
        "very_likely": 0.6
    },
    "generated_at": datetime.utcnow().isoformat()
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

# ---------------------------------------------
# Log model selection for audit/history
# ---------------------------------------------
log_entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "model_label": best_model_label,
    "threshold": best_threshold,
    **best_params,
    "custom_score": best_row['custom_score']
}

# Append or create log file
log_df = pd.DataFrame([log_entry])

if os.path.exists(log_path):
    existing_log = pd.read_csv(log_path)
    log_df = pd.concat([existing_log, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False)

print(f"\n✅ Final model saved to: {model_path}")
print(f"📋 Metadata saved to: {metadata_path}")
print(f"🗒️ Log updated at: {log_path}")