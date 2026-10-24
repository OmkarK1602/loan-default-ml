"""Tests for validation, preprocessing, prediction and invalid inputs.

Run with:  python -m pytest -q
(The tests train their own small model, so you do not need to run train.py first.)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import data_preprocessing as dp
from src.predict import predict_one, risk_band


@pytest.fixture(scope="module")
def df():
    return dp.load_data()


@pytest.fixture
def good_applicant():
    return {
        "Age": 40, "EmploymentType": "Salaried", "AnnualIncome": 70000,
        "EmploymentYears": 10, "LoanAmount": 400000, "LoanTermMonths": 60,
        "InterestRatePct": 12.0, "CreditScore": 680, "ExistingLoans": 1,
        "PastDelinquencies": 0, "DebtToIncomeRatio": 0.3, "LoanPurpose": "Vehicle",
        "HomeOwnership": "Rent", "Region": "North", "BankRelationship": "1-3 years",
        "PaymentHistory": "Good",
    }


@pytest.fixture(scope="module")
def small_model(df):
    """A quick model trained inside the test so tests do not depend on train.py."""
    X, y = dp.get_X_y(df)
    X_train, X_test, y_train, y_test = dp.split_data(X, y)
    pipe = dp.build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X_train, y_train)
    return pipe


# ---------- schema validation ----------
def test_real_dataset_is_valid(df):
    assert dp.validate_data(df) is True


def test_missing_column_is_rejected(df):
    with pytest.raises(ValueError, match="Missing columns"):
        dp.validate_data(df.drop(columns=["CreditScore"]))


def test_bad_target_value_is_rejected(df):
    bad = df.copy()
    bad.loc[0, "Default"] = "Maybe"
    with pytest.raises(ValueError, match="Yes/No"):
        dp.validate_data(bad)


def test_out_of_range_value_is_rejected(df):
    bad = df.copy()
    bad.loc[0, "DebtToIncomeRatio"] = 5
    with pytest.raises(ValueError, match="DebtToIncomeRatio"):
        dp.validate_data(bad)


# ---------- preprocessing ----------
def test_applicant_id_is_not_a_feature(df):
    X, y = dp.get_X_y(df)
    assert "ApplicantID" not in X.columns
    assert "Default" not in X.columns
    assert set(y.unique()) == {0, 1}


def test_split_is_stratified(df):
    X, y = dp.get_X_y(df)
    X_train, X_test, y_train, y_test = dp.split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_preprocessor_removes_missing_values(df):
    X, y = dp.get_X_y(df)
    assert X.isna().sum().sum() > 0              # there are missing values before
    out = dp.build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(out).any()               # none after


def test_preprocessor_handles_unseen_category(df):
    X, y = dp.get_X_y(df)
    pre = dp.build_preprocessor().fit(X)
    new = X.head(1).copy()
    new["Region"] = "Mars"                       # category never seen in training
    pre.transform(new)                           # should not crash


# ---------- predictions ----------
def test_prediction_output(small_model, good_applicant):
    result = predict_one(small_model, good_applicant)
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in ("Default", "No default")
    assert result["risk_band"] in ("Low", "Medium", "High")


def test_prediction_is_repeatable(small_model, good_applicant):
    a = predict_one(small_model, good_applicant)["probability"]
    b = predict_one(small_model, good_applicant)["probability"]
    assert a == b


def test_missing_value_in_input_is_allowed(small_model, good_applicant):
    good_applicant["CreditScore"] = None
    assert 0 <= predict_one(small_model, good_applicant)["probability"] <= 1


def test_threshold_changes_label(small_model, good_applicant):
    low = predict_one(small_model, good_applicant, threshold=0.0)
    high = predict_one(small_model, good_applicant, threshold=1.01)
    assert low["prediction"] == "Default"
    assert high["prediction"] == "No default"


def test_risk_band():
    assert risk_band(0.05) == "Low"
    assert risk_band(0.3) == "Medium"
    assert risk_band(0.8) == "High"


# ---------- invalid inputs ----------
def test_missing_field_is_rejected(small_model, good_applicant):
    del good_applicant["Age"]
    with pytest.raises(ValueError, match="Missing input fields"):
        predict_one(small_model, good_applicant)


def test_negative_age_is_rejected(small_model, good_applicant):
    good_applicant["Age"] = -5
    with pytest.raises(ValueError, match="Age"):
        predict_one(small_model, good_applicant)


def test_text_in_number_field_is_rejected(small_model, good_applicant):
    good_applicant["AnnualIncome"] = "lots"
    with pytest.raises(ValueError, match="AnnualIncome"):
        predict_one(small_model, good_applicant)


def test_dti_above_one_is_rejected(small_model, good_applicant):
    good_applicant["DebtToIncomeRatio"] = 1.5
    with pytest.raises(ValueError, match="DebtToIncomeRatio"):
        predict_one(small_model, good_applicant)
