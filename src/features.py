import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# v1.3 Add 'time_index' and cyclical encoding (replaced by prophet model)
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from the datetime column.
    Enhanced Temporal Extractor with Cyclical Encoding and Time Index.
    """
    def __init__(self, date_column='datetime'):
        self.date_column = date_column

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        dt_series = pd.to_datetime(X[self.date_column])

        X['hour'] = dt_series.dt.hour
        X['month'] = dt_series.dt.month
        X['weekday'] = dt_series.dt.weekday

        # Drop original column to avoid redundancy in the pipeline
        X = X.drop(columns=[self.date_column])

        return X

# v1.8 Using WeatherLag to helps the model understand weather inertia
# for example: a sharp drop in temperature 2 hours ago will affect current visitor numbers
class WeatherLagTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Đảm bảo sort theo thời gian
        if 'datetime' in X.columns:
            # Nếu datetime là cột
            X['dt_temp'] = pd.to_datetime(X['datetime'])
            X = X.sort_values('dt_temp')
        elif isinstance(X.index, pd.DatetimeIndex):
            X = X.sort_index()

        # --- FEATURE ENGINEERING ---

        # 1. Temp Lags (Nhiệt độ quá khứ)
        # Shift 1 dòng = 1 giờ (do dữ liệu liên tục)
        X['temp_lag_1'] = X['temp'].shift(1).bfill()
        X['temp_lag_2'] = X['temp'].shift(2).bfill()

        # 2. Weather Rolling (Xu hướng thời tiết)
        # Độ ẩm trung bình 3 giờ gần nhất
        X['humidity_roll_3'] = X['humidity'].rolling(window=3, min_periods=1).mean()
        # Gió mạnh nhất trong 3 giờ qua
        X['windspeed_max_3'] = X['windspeed'].rolling(window=3, min_periods=1).max()

        # 3. Weather Change (Sự thay đổi thời tiết)
        # 1 giờ trước mưa (3), giờ tạnh (1) -> Diff = -2 (Tín hiệu tốt)
        X['weather_diff'] = X['weather'].diff().fillna(0)

        # Xóa cột tạm nếu có
        if 'dt_temp' in X.columns:
            X = X.drop(columns=['dt_temp'])

        return X

class DayNightTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, night_hours=None):
        self.is_fitted_ = False
        if night_hours is None:
            self.night_hours = [0, 1, 2, 3, 4, 5]
        else:
            self.night_hours = night_hours

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        if 'datetime' in X.columns:
            # Chuyển đổi an toàn
            dt_col = pd.to_datetime(X['datetime'])
            # Feature 1: Is Night (Boolean/Int)
            X['is_night'] = dt_col.dt.hour.isin(self.night_hours).astype(int)
            # Feature 2: Hour (Raw hour is important for trees)
            X['hour_raw'] = dt_col.dt.hour
        return X

class WindspeedImputer(BaseEstimator, TransformerMixin):
    """
    Handles 0.0 values in windspeed using a group-based mean strategy.
    Prevents data leakage by calculating means only during 'fit'.
    """
    def __init__(self, method='mean'):
        self.method = method
        self.fill_values_ = None
        self.global_mean_ = None

    def fit(self, X, y=None):
        # We use 'season' and 'hour' as groups to calculate more accurate means
        # Assuming temporal features are extracted before this step
        self.fill_values_ = X[X['windspeed'] > 0].groupby(['season', 'hour'])['windspeed'].mean()
        self.global_mean_ = X[X['windspeed'] > 0]['windspeed'].mean()
        return self

    def transform(self, X):
        X = X.copy()

        # Logic to fill zeros based on (season, hour) mapping
        zero_mask = (X['windspeed'] == 0)
        fill_map = X.loc[zero_mask, ['season', 'hour']].apply(tuple, axis=1).map(self.fill_values_)
        X.loc[zero_mask, 'windspeed'] = fill_map.fillna(self.global_mean_)

        return X

class BikeInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Creates interaction terms defined in config.py.
    Example: hour * workingday
    """
    def __init__(self, interaction_pairs):
        self.interaction_pairs = interaction_pairs

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        for feat_a, feat_b in self.interaction_pairs:
            new_col_name = f"{feat_a}_{feat_b}"
            X[new_col_name] = X[feat_a].astype(float) * X[feat_b].astype(float)

        return X

class BikeRushHourTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        # Rush Hour for Office Workers (Registered)
        X['is_rush_hour'] = (((X['hour'] >= 7) & (X['hour'] <= 9)) |
                            ((X['hour'] >= 17) & (X['hour'] <= 19))) & \
                            (X['workingday'] == 1)
        X['is_rush_hour'] = X['is_rush_hour'].astype(int)

        return X
