import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import src.config as config
from src.dataloader import load_data
from src.pipeline import build_preprocessing_pipeline
from src.validation import ExpandingWindowSplitter
from src.utils import predict_residual_stage, train_predict_residual_fold

# v1.4 Using blending model (XGBoost and LightGBM)
# v1.6 PLUS prophet model (with tune Expanding Window)
def train_final_system():
    """
    Refactored training logic using Time-Series Cross-Validation.
    Ensures local validation closely mimics Kaggle's time-based testing.
    """
    # 1. Load data
    X, y_casual, y_registered, _ = load_data()
    train_meta = X[['datetime']].copy()
    train_meta['hour'] = train_meta['datetime'].dt.hour

    # 2. Setup TimeSeriesSplit
    # v1.5 Expanding Window with Gap
    cv = ExpandingWindowSplitter(
        n_splits=5,
        gap_size=168, # gap 1 week
        test_size=168
    )

    all_fold_models = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        # Split data for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_c_train, y_c_val = y_casual.iloc[train_idx], y_casual.iloc[val_idx]
        y_r_train, y_r_val = y_registered.iloc[train_idx], y_registered.iloc[val_idx]

        # Train Casual
        pred_c, trees_c, prophet_c = train_predict_residual_fold(
            X_train, y_c_train, X_val, build_preprocessing_pipeline
        )
        pred_r, trees_r, prophet_r = train_predict_residual_fold(
            X_train, y_r_train, X_val, build_preprocessing_pipeline
        )

        # Save model
        all_fold_models.append((prophet_c, trees_c, prophet_r, trees_r))

        # Score
        total_pred = pred_c + pred_r
        total_actual = y_c_val + y_r_val
        score = np.sqrt(mean_squared_error(np.log1p(total_actual), np.log1p(total_pred)))
        print(f"Fold {fold + 1} RMSLE: {score:.4f}")
        gc.collect()

    return all_fold_models

def generate_submission():
    # 1. Get model
    all_fold_models = train_final_system()

    # 2. Predict & Average
    _, _, _, test_df = load_data()

    # Predict
    fold_preds_matrix = np.zeros((len(test_df), len(all_fold_models)))

    for i, (prophet_c, trees_c, prophet_r, trees_r) in enumerate(all_fold_models):
        p_c = predict_residual_stage(prophet_c, trees_c, test_df)
        p_r = predict_residual_stage(prophet_r, trees_r, test_df)
        fold_preds_matrix[:, i] = p_c + p_r

    # Calculate average (Raw Predictions)
    final_test_preds = np.mean(fold_preds_matrix, axis=1)

    # 3. Save
    submission = pd.DataFrame({
        "datetime": test_df["datetime"],
        "count": final_test_preds,
    })

    submission.to_csv(config.SUBMISSION_PATH, index=False)

if __name__ == "__main__":
    generate_submission()
