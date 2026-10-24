"""Bonus: a very simple monitoring report (data drift check).

Compares new data with the training data profile saved by train.py.
Usage:  python -m src.monitor path/to/new_data.csv
If no file is given, the test split is used as the "new" data.
"""

import os
import sys
import pandas as pd

from src import data_preprocessing as dp
from src.train import REPORT_DIR


def drift_report(new_df, profile_path=os.path.join(REPORT_DIR, "training_data_profile.csv")):
    profile = pd.read_csv(profile_path, index_col=0)
    rows = []
    for col in dp.NUMERIC_COLUMNS:
        train_mean = profile.loc[col, "mean"]
        train_std = profile.loc[col, "std"]
        new_mean = new_df[col].mean()
        shift = (new_mean - train_mean) / train_std   # shift measured in std units
        rows.append({
            "feature": col,
            "train_mean": round(train_mean, 3),
            "new_mean": round(new_mean, 3),
            "shift_in_std": round(shift, 3),
            "missing_pct": round(new_df[col].isna().mean() * 100, 2),
            "status": "CHECK" if abs(shift) > 0.25 else "ok",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = pd.read_csv(sys.argv[1])
    else:
        X, y = dp.get_X_y(dp.load_data())
        data = dp.split_data(X, y)[1]
    report = drift_report(data)
    print(report.to_string(index=False))
    report.to_csv(os.path.join(REPORT_DIR, "monitoring_report.csv"), index=False)
