"""Train models, track everything in MLflow, tune one model and register the best.

Run from the project folder:
    python -m src.train
"""

import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src import data_preprocessing as dp
from src import evaluate as ev

TRACKING_URI = "sqlite:///" + os.path.join(dp.ROOT_DIR, "mlflow.db")
EXPERIMENT_NAME = "Loan_Default_Risk"
REGISTERED_MODEL_NAME = "LoanDefaultRiskModel"
SELECTION_METRIC = "cv_pr_auc"   # decided BEFORE looking at results
MODEL_PATH = os.path.join(dp.ROOT_DIR, "models", "best_model.joblib")
REPORT_DIR = os.path.join(dp.ROOT_DIR, "reports")


def train_one_run(run_name, model_name, model, params, data, tag_stage="baseline"):
    """Train one pipeline and log everything to MLflow. Returns a small summary dict."""
    X_train, X_test, y_train, y_test = data
    pipeline = dp.build_pipeline(model)

    with mlflow.start_run(run_name=run_name, nested=(tag_stage == "tuning")) as run:
        mlflow.set_tags({"model_type": model_name, "stage": tag_stage})

        # parameters, seed and split settings
        mlflow.log_params(params)
        mlflow.log_param("random_seed", dp.RANDOM_SEED)
        mlflow.log_param("test_size", dp.TEST_SIZE)
        mlflow.log_param("stratified_split", True)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))

        # cross-validation on the TRAINING data only (used to choose the model)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=dp.RANDOM_SEED)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision")
        mlflow.log_metric("cv_pr_auc", cv_scores.mean())
        mlflow.log_metric("cv_pr_auc_std", cv_scores.std())

        # fit on the full training set and evaluate on the untouched test set
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = ev.get_metrics(y_test, y_proba)
        for name, value in metrics.items():
            mlflow.log_metric("test_" + name, value)

        # artifacts (plots and text files) - saved in a temp folder, then uploaded
        with tempfile.TemporaryDirectory() as tmp:
            ev.save_confusion_matrix(y_test, y_proba, os.path.join(tmp, "confusion_matrix.png"))
            ev.save_roc_curve(y_test, y_proba, os.path.join(tmp, "roc_curve.png"))
            ev.save_pr_curve(y_test, y_proba, os.path.join(tmp, "pr_curve.png"))
            ev.save_classification_report(y_test, y_proba, os.path.join(tmp, "classification_report.txt"))
            ev.save_feature_importance(pipeline, os.path.join(tmp, "feature_importance.png"),
                                       os.path.join(tmp, "feature_importance.csv"))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        # the model itself
        signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
        # cloudpickle is the classic format; newer MLflow defaults to skops,
        # which refuses some numpy types inside our pipeline.
        mlflow.sklearn.log_model(pipeline, name="model", signature=signature,
                                 input_example=X_train.head(3),
                                 serialization_format="cloudpickle")

        print(f"{run_name:<28} CV PR-AUC={cv_scores.mean():.3f}  "
              f"test PR-AUC={metrics['pr_auc']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  "
              f"recall={metrics['recall']:.3f}  precision={metrics['precision']:.3f}")

        return {"run_id": run.info.run_id, "run_name": run_name, "pipeline": pipeline,
                "cv_pr_auc": cv_scores.mean(), **metrics}


def baseline_models():
    """The four required models with simple starting parameters."""
    seed = dp.RANDOM_SEED
    return [
        ("logistic_regression", "LogisticRegression",
         LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
         {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
        ("decision_tree", "DecisionTree",
         DecisionTreeClassifier(max_depth=5, min_samples_split=20, class_weight="balanced", random_state=seed),
         {"max_depth": 5, "min_samples_split": 20, "class_weight": "balanced"}),
        ("random_forest", "RandomForest",
         RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=seed),
         {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"}),
        ("gradient_boosting", "GradientBoosting",
         GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=seed),
         {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    ]


# Six different settings to try for Gradient Boosting (the tuning part)
TUNING_CONFIGS = [
    {"n_estimators": 50,  "learning_rate": 0.1,  "max_depth": 2},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.02, "max_depth": 3},
    {"n_estimators": 150, "learning_rate": 0.1,  "max_depth": 4},
    {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 2},
]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---- load, validate, split ----
    df = dp.load_data()
    dp.validate_data(df)
    X, y = dp.get_X_y(df)
    data = dp.split_data(X, y)
    print("Rows:", len(df), "| train:", len(data[0]), "| test:", len(data[1]))
    print("Default rate  train: %.3f  test: %.3f\n" % (data[2].mean(), data[3].mean()))

    results = []

    # ---- Part B: four baseline runs ----
    print("=== Baseline runs ===")
    for run_name, model_name, model, params in baseline_models():
        results.append(train_one_run(run_name, model_name, model, params, data))

    # ---- Hyperparameter tuning: Gradient Boosting, 6 configs as nested runs ----
    print("\n=== Tuning Gradient Boosting (nested runs) ===")
    tuned = []
    with mlflow.start_run(run_name="tuning_gradient_boosting") as parent:
        mlflow.set_tag("stage", "tuning_parent")
        for i, cfg in enumerate(TUNING_CONFIGS, start=1):
            model = GradientBoostingClassifier(random_state=dp.RANDOM_SEED, **cfg)
            res = train_one_run(f"gb_tune_{i}", "GradientBoosting", model, cfg, data, tag_stage="tuning")
            tuned.append(res)
        best_tuned = max(tuned, key=lambda r: r["cv_pr_auc"])
        baseline_gb = [r for r in results if r["run_name"] == "gradient_boosting"][0]
        mlflow.log_metric("baseline_cv_pr_auc", baseline_gb["cv_pr_auc"])
        mlflow.log_metric("best_tuned_cv_pr_auc", best_tuned["cv_pr_auc"])
        mlflow.log_metric("cv_pr_auc_improvement", best_tuned["cv_pr_auc"] - baseline_gb["cv_pr_auc"])
        mlflow.set_tag("best_tuning_run", best_tuned["run_name"])
    results += tuned

    # ---- Part C: pick the best run automatically using the predefined metric ----
    select_and_register(data, [r["run_id"] for r in results])


def select_and_register(data, run_ids):
    """Find the best run in MLflow, save reports, register the model."""
    X_train, X_test, y_train, y_test = data
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    # Ask MLflow for all finished runs (not the parent run) sorted by our metric
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage != 'tuning_parent'",
        order_by=[f"metrics.{SELECTION_METRIC} DESC"],
    )
    # the script may have been run before, so keep only the runs from this execution
    runs = runs[runs["run_id"].isin(run_ids)].reset_index(drop=True)

    table = runs[["tags.mlflow.runName", "tags.model_type", "tags.stage",
                  "metrics.cv_pr_auc", "metrics.test_pr_auc", "metrics.test_roc_auc",
                  "metrics.test_precision", "metrics.test_recall", "metrics.test_f1",
                  "metrics.test_accuracy"]].copy()
    table.columns = ["run", "model", "stage", "cv_pr_auc", "pr_auc", "roc_auc",
                     "precision", "recall", "f1", "accuracy"]
    table = table.round(4).reset_index(drop=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    table.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)
    print("\n=== All runs ranked by", SELECTION_METRIC, "===")
    print(table.to_string())

    best = runs.iloc[0]
    best_run_id = best["run_id"]
    best_name = best["tags.mlflow.runName"]
    print(f"\nBest run: {best_name} ({best_run_id})")

    # Register the model in the Model Registry (works because we use a SQLite backend)
    model_uri = f"runs:/{best_run_id}/model"
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "selected_by", SELECTION_METRIC)
    client.update_model_version(
        REGISTERED_MODEL_NAME, version.version,
        description=f"{best_name}; CV PR-AUC={best['metrics.cv_pr_auc']:.3f}. "
                    "Synthetic educational data - NOT for real lending decisions.")
    print(f"Registered {REGISTERED_MODEL_NAME} version {version.version} (alias: champion)")

    # Save a local copy too (the Streamlit app can use it if the registry is missing)
    pipeline = mlflow.sklearn.load_model(model_uri)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # Bonus: threshold comparison for the selected model
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    thr = ev.compare_thresholds(y_test, y_proba)
    thr.to_csv(os.path.join(REPORT_DIR, "threshold_comparison.csv"), index=False)
    print("\n=== Threshold comparison (selected model, test set) ===")
    print(thr.to_string(index=False))
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(os.path.join(REPORT_DIR, "threshold_comparison.csv"), artifact_path="evaluation")

    # Bonus: simple monitoring reference - training feature means, used by monitor.py
    X_train.describe().T.to_csv(os.path.join(REPORT_DIR, "training_data_profile.csv"))


if __name__ == "__main__":
    main()
