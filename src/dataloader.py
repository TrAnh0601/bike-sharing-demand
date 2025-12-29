import pandas as pd
from src import config

def load_data():
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # Sort by datetime
    train_df["datetime"] = pd.to_datetime(train_df["datetime"])
    train_df = train_df.sort_values("datetime").reset_index(drop=True)

    X = train_df.drop(columns=[config.TARGET_TOTAL, config.TARGET_CASUAL, config.TARGET_REGISTERED])
    y_casual = train_df[config.TARGET_CASUAL]
    y_registered = train_df[config.TARGET_REGISTERED]

    return X, y_casual, y_registered, test_df
