from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, RegressorMixin

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.features import TemporalFeatureExtractor, WindspeedImputer, BikeInteractionTransformer, BikeRushHourTransformer, WeatherLagTransformer, DayNightTransformer
import src.config as config

# Use Wrapper Class to fix Catboost + Sklearn 1.6+ errors.
class CatBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, iterations=1000, learning_rate=0.05, depth=6,
                 loss_function='RMSE', verbose=False, allow_writing_files=False,
                 random_state=42, thread_count=-1,
                 l2_leaf_reg=3.0, subsample=0.8,
                 **kwargs):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.random_state = random_state
        self.thread_count = thread_count
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.kwargs = kwargs
        self.model = None
        self.fitted_ = False

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
            random_seed=self.random_state,
            thread_count=self.thread_count,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            **self.kwargs
        )

        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self.model.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)

# v1.4 Using blending model (XGBoost + LightGBM + Catboost(v1.9))
def build_model(model_type='xgboost', seed=42):
    if model_type == 'xgboost':
        return XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            sub_sample=0.8,
            n_jobs=-1,
            random_state=seed,
            early_stopping_rounds=50
        )
    elif model_type == 'lightgbm':
        return LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
            verbose=-1
        )
    # v1.9 Catboost
    elif model_type == 'catboost':
        return CatBoostWrapper(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            random_state=seed,
            l2_leaf_reg=3.0,
            subsample=0.8
        )

def build_preprocessing_pipeline(model_type):
    """
    Assembles the complete preprocessing pipeline.
    Modular stages for easy debugging.
    """
    # 1. Temporal Stage: Extract time features first
    temporal_stage = Pipeline(steps=[
        ('weather_lags', WeatherLagTransformer()),
        ('day_night', DayNightTransformer()),
        ('extractor', TemporalFeatureExtractor(date_column=config.FEATURES_TEMPORAL)),
    ])

    # 2. Imputation Stage: WindspeedImputer BEFORE ColumnTransformer
    imputation_stage = WindspeedImputer(method='mean')

    # 3. Interactions Stage
    interaction_stage = BikeInteractionTransformer(interaction_pairs=config.INTERACTION_PAIRS)

    # 4. Scaling & Encoding Stage
    if model_type == 'catboost':
        # CatBoost: Ordinal (0, 1, 2) hơn là One-Hot (0, 0, 1...)
        cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        # XGB/LGB: One-Hot
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), config.FEATURES_NUMERICAL),
            ('cat', cat_transformer, config.FEATURES_CATEGORICAL),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")

    # 5. Final Preprocessing Pipeline including Interaction Terms
    full_pipeline = Pipeline(steps=[
        ('temporal', temporal_stage),
        ('imputer', imputation_stage),
        ('interactions', interaction_stage),
        ('rush_hour', BikeRushHourTransformer()),
        ('col_transform', preprocessor),
    ])

    return full_pipeline
