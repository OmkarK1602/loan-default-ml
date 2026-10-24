"""Data loading, validation and preprocessing for the loan default project."""

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "loan_default_finance_dataset.csv")

# Settings used everywhere (so every run is reproducible)
RANDOM_SEED = 42
TEST_SIZE = 0.2

TARGET = "Default"
ID_COLUMN = "ApplicantID"

NUMERIC_COLUMNS = [
    "Age", "AnnualIncome", "EmploymentYears", "LoanAmount", "LoanTermMonths",
    "InterestRatePct", "CreditScore", "ExistingLoans", "PastDelinquencies",
    "DebtToIncomeRatio",
]
CATEGORICAL_COLUMNS = [
    "EmploymentType", "LoanPurpose", "HomeOwnership", "Region",
    "BankRelationship", "PaymentHistory",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Allowed ranges, used by the validation functions
VALID_RANGES = {
    "Age": (18, 100),
    "AnnualIncome": (0, 10_000_000),
    "EmploymentYears": (0, 60),
    "LoanAmount": (1, 100_000_000),
    "LoanTermMonths": (1, 480),
    "InterestRatePct": (0, 100),
    "CreditScore": (300, 900),
    "ExistingLoans": (0, 50),
    "PastDelinquencies": (0, 50),
    "DebtToIncomeRatio": (0, 1),
}


def load_data(path=DATA_PATH):
    """Read the csv file into a DataFrame."""
    return pd.read_csv(path)


def validate_data(df):
    """Check that the training data looks the way we expect.

    Raises ValueError with a clear message if something is wrong.
    """
    # 1. all columns must exist
    needed = FEATURE_COLUMNS + [TARGET]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # 2. dataset should not be empty
    if len(df) == 0:
        raise ValueError("The dataset is empty")

    # 3. target must be only Yes / No
    bad_target = set(df[TARGET].dropna().unique()) - {"Yes", "No"}
    if bad_target or df[TARGET].isna().any():
        raise ValueError(f"Target column '{TARGET}' must only contain Yes/No (found {bad_target})")

    # 4. numbers must be in a sensible range (missing values are allowed here,
    #    they get imputed later)
    for col, (low, high) in VALID_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if (df[col].notna() & values.isna()).any():
            raise ValueError(f"Column '{col}' has non-numeric values")
        if ((values < low) | (values > high)).any():
            raise ValueError(f"Column '{col}' has values outside {low} to {high}")

    return True


def validate_input(row):
    """Check one applicant (a dict) before predicting. Returns a 1-row DataFrame."""
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")

    for col, (low, high) in VALID_RANGES.items():
        value = row[col]
        if value is None:
            continue  # missing is OK, the pipeline imputes it
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{col} must be a number")
        if value != value:  # NaN
            continue
        if value < low or value > high:
            raise ValueError(f"{col} must be between {low} and {high}")

    for col in CATEGORICAL_COLUMNS:
        if row[col] is not None and not isinstance(row[col], str):
            raise ValueError(f"{col} must be text")

    return pd.DataFrame([{c: row[c] for c in FEATURE_COLUMNS}])


def get_X_y(df):
    """Split the DataFrame into features (X) and a 0/1 target (y).

    ApplicantID is dropped here, it is only an identifier.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def split_data(X, y, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """Stratified train/test split so both parts keep the same default rate."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_preprocessor():
    """ColumnTransformer: impute + scale numbers, impute + one-hot encode categories.

    It is put inside a Pipeline together with the model, so it is fitted only
    on training data (no data leakage).
    """
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERIC_COLUMNS),
        ("cat", categorical_steps, CATEGORICAL_COLUMNS),
    ])


def build_pipeline(model):
    """Full pipeline = preprocessing + model."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
