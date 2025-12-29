import numpy as np
import pandas as pd
from prophet import Prophet

from src import config
from src.pipeline import build_preprocessing_pipeline, build_model

# v1.6 Prophet model
def fit_prophet(df_train, train_y):
    df = df_train.copy()
    df['ds'] = df['datetime']
    df['y'] = train_y

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
    )

    # Regressors
    for col in config.PROPHET_REGRESSORS:
        if col in df.columns:
            m.add_regressor(col)

    m.fit(df)
    return m

def predict_prophet_with_regressors(m, df_future):
    """
    The Prophet helper function predicts when Regressors are available.
    The regressor information needs to be merged into the future dataframe.
    """
    future = pd.DataFrame({'ds': df_future['datetime']})

    for col in config.PROPHET_REGRESSORS:
        if col in df_future.columns:
            future[col] = df_future[col].values

    # Forecast
    forecast = m.predict(future)
    return forecast['yhat'].values

def train_predict_residual_fold(X_train, y_train, X_val, pipeline_builder):
    """
    Residual Learning with Seed Averaging & Prophet Regressors.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train['datetime'] = pd.to_datetime(X_train['datetime'])
    X_val['datetime'] = pd.to_datetime(X_val['datetime'])

    # 1. PROPHET (BASE)
    y_train_log = np.log1p(y_train)
    m = fit_prophet(X_train, y_train_log)

    # Predict In-sample & Out-sample
    train_pred_prophet = predict_prophet_with_regressors(m, X_train)
    val_pred_prophet = predict_prophet_with_regressors(m, X_val)
    residuals_train = y_train_log - train_pred_prophet

    # 2. ENSEMBLE WITH SEED AVERAGING
    # Split Internal Train for Early Stopping (90% Train, 10% Watchlist)
    split_idx = int(len(X_train) * 0.9)
    y_tr_resid = residuals_train.iloc[:split_idx]
    y_watch_resid = residuals_train.iloc[split_idx:]

    seeds = [42, 2024, 999]
    model_types = ['xgboost', 'lightgbm', 'catboost']

    ensemble_full_pipelines = {mt: [] for mt in model_types}

    val_resid_accum = np.zeros(len(X_val))

    for mt in model_types:
        # a. Build & Fit Preprocessor
        preprocessor = build_preprocessing_pipeline(model_type=mt)

        # Fit
        X_train_trans = preprocessor.fit_transform(X_train)
        X_val_trans = preprocessor.transform(X_val)

        # Create data for early stopping
        X_tr_int_trans = X_train_trans[:split_idx]
        X_watch_trans = X_train_trans[split_idx:]

        for seed in seeds:
            # b. Get Model Instance (with seed)
            model = build_model(mt, seed=seed)

            # c. Fit with Early Stopping
            fit_params = {'eval_set': [(X_watch_trans, y_watch_resid)]}

            # LightGBM sklearn wrapper fix
            if mt == 'xgboost':
                fit_params['verbose'] = False

            elif mt == 'lightgbm':
                from lightgbm import early_stopping, log_evaluation
                fit_params['callbacks'] = [early_stopping(stopping_rounds=50), log_evaluation(0)]
                fit_params['eval_metric'] = 'rmse'

            elif mt == 'catboost':
                fit_params['early_stopping_rounds'] = 50

            model.fit(X_tr_int_trans, y_tr_resid, **fit_params)

            # e. Save "Pipeline"
            ensemble_full_pipelines[mt].append((preprocessor, model))

            # f. Predict Val
            pred = model.predict(X_val_trans)
            val_resid_accum += pred

    # Average Validation
    val_pred_residuals_avg = val_resid_accum / (len(model_types) * len(seeds))

    # Combine
    final_log = val_pred_prophet + val_pred_residuals_avg
    final_log = np.clip(final_log, -1e-6, 10.0)
    final_pred = np.expm1(final_log)

    return np.maximum(0, final_pred), ensemble_full_pipelines, m

def predict_residual_stage(prophet_model, ensemble_pipelines_dict, X_df):
    X_df = X_df.copy()
    X_df['datetime'] = pd.to_datetime(X_df['datetime'])
    X_df['hour'] = X_df['datetime'].dt.hour

    # 1. Prophet Predict (Trend) - Log Space
    future = pd.DataFrame({'ds': X_df['datetime']})
    for col in config.PROPHET_REGRESSORS:
        if col in X_df.columns:
            future[col] = X_df[col].values

    base_pred_log = prophet_model.predict(future)['yhat'].values

    # 2. Tree Ensemble
    # Helper to average list of (preprocessor, model)
    def get_avg_pred(pipeline_list, X_raw):
        preds = []
        for preproc, model in pipeline_list:
            X_trans = preproc.transform(X_raw)
            preds.append(model.predict(X_trans))
        return np.mean(preds, axis=0)

    # Calculate Averages
    avg_cat = get_avg_pred(ensemble_pipelines_dict['catboost'], X_df)
    avg_xgb = get_avg_pred(ensemble_pipelines_dict['xgboost'], X_df)
    avg_lgb = get_avg_pred(ensemble_pipelines_dict['lightgbm'], X_df)

    # Apply Weights
    weighted_resid = (0.5 * avg_cat) + (0.25 * avg_xgb) + (0.25 * avg_lgb)

    # 3. Combine
    final_log = base_pred_log + weighted_resid
    return np.expm1(np.clip(final_log, 0, 10))
