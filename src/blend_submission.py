import pandas as pd
import src.config as config

BEST_SUB_PATH = config.SUBMISSION_DIR / 'best_submission.csv'
CURRENT_SUB_PATH = config.SUBMISSION_PATH
OUTPUT_PATH = config.SUBMISSION_DIR / 'final_submission.csv'

def main():
    # 2. Load submission file
    df_best = pd.read_csv(BEST_SUB_PATH)
    df_curr = pd.read_csv(CURRENT_SUB_PATH)

    # 3. Normalize Datetime & Sort
    df_best['datetime'] = pd.to_datetime(df_best['datetime'])
    df_best = df_best.sort_values('datetime').reset_index(drop=True)

    df_curr['datetime'] = pd.to_datetime(df_curr['datetime'])
    df_curr = df_curr.sort_values('datetime').reset_index(drop=True)

    # Length check
    if len(df_best) != len(df_curr):
        print("LỖI: Hai file submissions không cùng số lượng dòng!")
        return

    # 4. Post-processing
    curr_counts = df_curr['count'].copy()

    # Rolling Median
    curr_smooth = curr_counts.rolling(window=3, center=True, min_periods=1).median()

    # Blend nội bộ: 80% Gốc + 20% Mượt (Để giữ độ nhạy nhưng giảm nhiễu)
    df_curr['count_processed'] = (0.8 * curr_counts) + (0.2 * curr_smooth)

    # 5. Ensembling
    final_blend = (0.65 * df_best['count']) + (0.35 * df_curr['count_processed'])

    # 6. Save
    submission = pd.DataFrame({
        "datetime": df_best["datetime"],
        "count": final_blend
    })

    # Đảm bảo không âm
    submission['count'] = submission['count'].apply(lambda x: max(0, x))
    submission.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()