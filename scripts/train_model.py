
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, UTC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


#############################################################################################################
### DIRECTORIES & GLOBAL VARIABLES
#############################################################################################################

# Random State for Replication

SEED = 42

### Directories

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# Flight Delay Features

CATEGORICAL_FEATURES = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 'holiday_proximity_bucket']
NUMERIC_FEATURES = []

# Preprocessor (Transformations for XGBoost)

PREPROCESSOR = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ('num', 'passthrough', NUMERIC_FEATURES)
    ]
)

#############################################################################################################
### FUNCTIONS
#############################################################################################################

### -------------------------------------------------------------------------------------------------

def load_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, 'ml_dataset.parquet'))
    df_sampled = df.sample(frac=0.5, random_state=SEED)
    return df_sampled

### -------------------------------------------------------------------------------------------------

### NEED TO FIX THIS, "DEP_HOUR" SHOULD BE CATEGORICAL

def prepare_data():

    df = load_data()

    features = ['month', 'dayofweek', 'origin', 'dest', 'reporting_airline', 'dep_hour', 'holiday_proximity_bucket']
    target = 'arrdelayminutes'

    X = df[features]
    y = (df[target] > 30).astype(int)

    return X, y

### -------------------------------------------------------------------------------------------------

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

### -------------------------------------------------------------------------------------------------

def cross_validate_models(X, y, param_grid, thresholds):

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    results = []
    model_id = 1 

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
                            n_estimators=200,
                            learning_rate=lr,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=0.9,
                            scale_pos_weight=spw,
                            random_state=SEED,
                            n_jobs=-1
                        )

                        pipeline = Pipeline([
                            ('preprocessor', PREPROCESSOR),
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

### -------------------------------------------------------------------------------------------------

def tune_and_evaluate_models(X, y):

    param_grid = {
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [3, 5],
        'subsample': [0.8, 1.0]
    }

    thresholds = [0.15, 0.2, 0.25, 0.3, 0.35] # thresholds near proposed boundary
    results = cross_validate_models(X, y, param_grid, thresholds)

    results_df = pd.DataFrame(results)
    output_path = os.path.join(MODELS_DIR, 'model_xgb_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Cross-validated results with multiple thresholds saved to:\n{output_path}")

    return results_df

### -------------------------------------------------------------------------------------------------

def select_and_train_best_model(X, y, results_df):

    results_df['custom_score'] = (
        0.55 * results_df['recall_1'] +
        0.35 * results_df['recall_0'] +
        0.05 * results_df['precision_0'] +
        0.05 * results_df['roc_auc']
    )

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

    final_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=200,
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=0.9,
        scale_pos_weight=best_params['scale_pos_weight'],
        random_state=SEED,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', PREPROCESSOR),
        ('classifier', final_model)
    ])

    pipeline.fit(X, y)

    return pipeline, best_model_label, best_threshold, best_params, best_row

### -------------------------------------------------------------------------------------------------

def evaluate_thresholds_on_full_data(pipeline, X, y):

    thresholds_to_test = np.arange(0.05, 1.01, 0.05)
    y_probs = pipeline.predict_proba(X)[:, 1]

    threshold_metrics = []

    for thr in thresholds_to_test:
        metrics = evaluate_metrics(y, y_probs, thr)
        metrics['threshold'] = round(thr, 2)
        threshold_metrics.append(metrics)

    threshold_df = pd.DataFrame(threshold_metrics)
    output_path = os.path.join(MODELS_DIR, 'model_thresholds.csv')
    threshold_df.to_csv(output_path, index=False)
    print(f"\n📊 Threshold evaluations saved to:\n{output_path}")

### -------------------------------------------------------------------------------------------------

def save_model_and_metadata(pipeline, best_model_label, best_threshold, best_params, best_row):

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, 'final_model.pkl')
    metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')

    # Save model
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        "model_label": best_model_label,
        "selected_threshold": best_threshold,
        "parameters": best_params,
        "custom_score": best_row['custom_score'],
        "thresholds": {
            "Delay Very Unlikely": 0.25,
            "Delay Unlikely": best_threshold,
            "Delay Somewhat Likely": 0.45,
            "Delay Likely": 0.6,
            "Delay Very Likely": 0.75
        },
        "generated_at": datetime.now(UTC).isoformat()
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Final model saved to: {model_path}")
    print(f"📋 Metadata saved to: {metadata_path}")

### -------------------------------------------------------------------------------------------------

def log_model_training(best_model_label, best_threshold, best_params, best_row):

    log_path = os.path.join(MODELS_DIR, 'model_training_log.csv')

    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_label": best_model_label,
        "threshold": best_threshold,
        **best_params,
        "custom_score": best_row['custom_score']
    }

    log_df = pd.DataFrame([log_entry])

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        log_df = pd.concat([existing_log, log_df], ignore_index=True)

    log_df.to_csv(log_path, index=False)
    print(f"🗒️ Log updated at: {log_path}")


#############################################################################################################
### CALL MAIN
#############################################################################################################

if __name__ == '__main__':

    X, y = prepare_data()
    results_df = tune_and_evaluate_models(X, y)
    pipeline, best_model_label, best_threshold, best_params, best_row = select_and_train_best_model(X, y, results_df)
    evaluate_thresholds_on_full_data(pipeline, X, y)
    save_model_and_metadata(pipeline, best_model_label, best_threshold, best_params, best_row)
    log_model_training(best_model_label, best_threshold, best_params, best_row)