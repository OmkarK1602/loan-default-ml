# Loan Default Risk - MLflow End-to-End Project

> **Classroom project only.** The dataset is synthetic (made up). The model must **never** be used for real
> credit decisions, lending, underwriting or financial advice.

## 1. Problem
A lending analytics team wants to study what is linked to loan default. I built a binary classifier that predicts the
`Default` label (Yes/No), tracked all experiments with MLflow, registered the best model and made a small Streamlit demo.

## 2. Dataset
- `data/loan_default_finance_dataset.csv` - 2,200 synthetic applicants, 17 columns + target (see `data/README.md`)
- About **20.5% defaults** (imbalanced), no duplicate rows
- Missing values: AnnualIncome 22, CreditScore 18, DebtToIncomeRatio 20, PaymentHistory 15
- `ApplicantID` is only an ID and is **not** used as a feature

## 3. Project structure
```
loan-default-ml/
├── data/                 dataset + data README
├── notebooks/eda.ipynb   EDA (7 plots), findings, fairness notes
├── src/
│   ├── data_preprocessing.py   loading, validation, split, ColumnTransformer + Pipeline
│   ├── train.py                4 models + tuning + MLflow logging + registry
│   ├── evaluate.py             metrics, plots, threshold comparison
│   ├── predict.py              load model + predict one applicant
│   └── monitor.py              (bonus) simple data-drift report
├── tests/test_preprocessing.py 17 tests
├── app.py                Streamlit demo
├── reports/              model_comparison.csv, threshold_comparison.csv, training_log.txt, monitoring_report.csv
├── screenshots/          MLflow UI, registry and Streamlit screenshots
├── .github/workflows/tests.yml   (bonus) GitHub Actions
└── requirements.txt
```

## 4. Setup and how to run
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.train               # trains all models, logs to MLflow, registers best model
python -m pytest -q               # run the tests
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000   # then open http://localhost:5000
streamlit run app.py              # then open http://localhost:8501
python -m src.monitor             # (bonus) drift report
```
Run every command from the `loan-default-ml/` folder. Training takes about 2 minutes.
The tracking backend is **SQLite** (`mlflow.db`) because the Model Registry needs a database backend
(a database backend is the recommended way to use the registry; the old file store is limited there).
`mlflow.db`, `mlruns/` and `models/` are created by training, so they are not included in the zip - run `python -m src.train` first.

## 5. Preprocessing (Part A)
Everything is inside one scikit-learn `Pipeline`, so it is fitted on training data only (no leakage):

| Column type | Steps |
|---|---|
| Numeric (10 columns) | `SimpleImputer(median)` -> `StandardScaler` |
| Categorical (6 columns) | `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")` |

Split: stratified 80/20, `random_state=42` -> 1,760 train / 440 test, default rate 20.5% in both.
Trees do not need scaling, but one shared preprocessor keeps the code simple.

## 6. MLflow experiments (Part B)
- **Experiment** `Loan_Default_Risk` = a folder/group for one problem. **Run** = one single training attempt inside it
  (one set of parameters, its metrics, artifacts and model). One experiment has many runs, and we compare runs.
- Tracking URI: `sqlite:///mlflow.db`
- Each run logs: model parameters, `random_seed`, `test_size`, stratified split flag, row counts,
  test precision / recall / F1 / accuracy / ROC-AUC / PR-AUC, 5-fold CV PR-AUC (training data only),
  and artifacts: confusion matrix, ROC curve, PR curve, classification report, feature importance (png + csv), and the model.

### Baseline runs (test set, threshold 0.5)

| Run | CV PR-AUC | Test PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **logistic_regression** (C=1, class_weight=balanced) | **0.311** | 0.316 | 0.627 | 0.273 | 0.533 | 0.361 |
| decision_tree (depth 5) | 0.259 | 0.228 | 0.546 | 0.238 | 0.478 | 0.317 |
| random_forest (200 trees, depth 8) | 0.276 | 0.316 | 0.588 | 0.267 | 0.344 | 0.301 |
| gradient_boosting (100, lr 0.1) | 0.287 | 0.272 | 0.583 | 0.167 | 0.011 | 0.021 |

For reference, a random model would have PR-AUC around 0.205 (the default rate) and ROC-AUC 0.5.

### Hyperparameter tuning (Gradient Boosting, 6 nested runs under `tuning_gradient_boosting`)

| Run | n_estimators | learning_rate | max_depth | CV PR-AUC | Test PR-AUC |
|---|---|---|---|---|---|
| baseline | 100 | 0.10 | 3 | 0.287 | 0.272 |
| gb_tune_1 | 50 | 0.10 | 2 | 0.301 | 0.302 |
| **gb_tune_2** | 100 | 0.05 | 2 | **0.305** | 0.311 |
| gb_tune_3 | 200 | 0.05 | 3 | 0.289 | 0.254 |
| gb_tune_4 | 300 | 0.02 | 3 | 0.296 | 0.271 |
| gb_tune_5 | 150 | 0.10 | 4 | 0.283 | 0.239 |
| gb_tune_6 | 200 | 0.03 | 2 | 0.304 | 0.319 |

Tuning helped (CV PR-AUC 0.287 -> 0.305): **shallower trees and a lower learning rate** were better, so deeper trees were overfitting the noise.
Note the CV standard deviation is about 0.02, so differences of 0.01 are not very meaningful.

## 7. Model selection and registry (Part C)
- **Selection metric (decided before training): 5-fold cross-validated PR-AUC on the training set.**
  Why: the target is imbalanced, PR-AUC focuses on the rare "default" class, and it does not depend on a threshold.
  Selecting on training-CV (not on the test set) keeps the test set honest.
- `train.py` uses `mlflow.search_runs()` to pick the best run automatically (bonus: automated model selection).
- **Winner: Logistic Regression** (CV PR-AUC 0.311). It also gave the best recall (0.53) at the default threshold.
  Simple, fast and explainable - good for a weak-signal dataset where complex models only overfit.
- Registered as **`LoanDefaultRiskModel` version 1** with alias **`champion`** (SQLite backend supports the registry).
  Load it with `mlflow.sklearn.load_model("models:/LoanDefaultRiskModel@champion")`. The app falls back to
  `models/best_model.joblib` if the registry is not available. Evidence: `screenshots/mlflow_model_registry.png`.

### False positives vs false negatives
- **False positive** = we flag a borrower who would *not* default. The bank may reject or over-price a good customer (lost business, unfair treatment).
- **False negative** = we miss a borrower who *does* default. The bank loses the loan money.
- In lending, a false negative usually costs more money than a false positive, so recall matters. But too many false
  positives hurts customers and revenue, so I look at both. Accuracy alone is not used: always predicting "No default" would already give about 79% accuracy.

### Threshold comparison (bonus, selected model, test set)

| Threshold | Precision | Recall | F1 | False positives | False negatives |
|---|---|---|---|---|---|
| 0.2 | 0.203 | 0.956 | 0.335 | 338 | 4 |
| 0.3 | 0.213 | 0.889 | 0.344 | 295 | 10 |
| 0.4 | 0.245 | 0.744 | 0.368 | 207 | 23 |
| 0.5 | 0.273 | 0.533 | 0.361 | 128 | 42 |
| 0.6 | 0.356 | 0.344 | 0.350 | 56 | 59 |
| 0.7 | 0.344 | 0.122 | 0.180 | 21 | 79 |

Lowering the threshold catches more defaulters (higher recall) but flags many more good borrowers. The app has a threshold slider to show this trade-off.

## 8. Prediction app (Part D)
`app.py` (Streamlit) takes the 16 applicant features, validates them, runs the **same saved pipeline**
(imputing, scaling, encoding happen inside it) and shows the default probability, a risk band
(Low < 20%, Medium 20-40%, High >= 40%) and the label at the chosen threshold. A warning banner says it is a classroom demo.
See `screenshots/streamlit_app.png`.

## 9. Testing (Part E)
`python -m pytest -q` -> 17 tests covering schema validation, out-of-range values, bad target values, ApplicantID exclusion,
stratified split, no NaN after preprocessing, unseen categories, prediction output, repeatability, missing input values, and invalid inputs.

## 10. Bonus items done
- Automated model selection from MLflow runs (`select_and_register` in `train.py`)
- Threshold comparison (section 7)
- Simple monitoring report (`src/monitor.py`, mean-shift check vs training data)
- GitHub Actions workflow (`.github/workflows/tests.yml`)
- Not done: SHAP, retraining workflow (re-running `python -m src.train` retrains and registers a new version)

## 11. Key findings from EDA
1. Only ~20% of applicants default, so accuracy is misleading.
2. Credit score is the clearest signal: default rate about 30% (lowest quartile) vs 12% (highest).
3. Unemployed (37%) and Contract (24%) applicants default more than Salaried / Self-employed (18%).
4. Other features have weak links with the target, and `PaymentHistory` is not ordered as expected (Excellent is not better than Good).
   This is why all models have modest scores (ROC-AUC about 0.55-0.63).

Top Logistic Regression features: EmploymentType_Unemployed, CreditScore, other EmploymentType values, Region_South, HomeOwnership_Mortgage.

## 12. Limitations
- Weak model: ROC-AUC 0.63 and precision of only about 27%. Most flagged applicants would NOT default.
- Small test set (440 rows, ~90 defaulters), so scores are noisy; CV std is about 0.02-0.03.
- Synthetic labels: the model learns the simulator's rules, not real borrower behaviour.
- Only one random split / seed; no calibration of probabilities (the percentages should not be read as true probabilities).
- Model files use cloudpickle, only load model files you trust.

## 13. Fairness and responsible use
- **Age** is used as a feature - age-based decisions are illegal or restricted in many lending contexts.
- **Region** and **HomeOwnership** can act as proxies for community, ethnicity or wealth.
- **Unemployed** is the strongest feature; this could exclude people who can really repay.
- Differences between groups were not tested for equal error rates. A real project would compare false positive/negative rates
  across groups, remove or justify sensitive features, and get legal and human review.
- **Data leakage**: avoided by excluding `ApplicantID`, doing the split before fitting, and keeping all preprocessing inside the Pipeline. Model choice used training CV, not the test set.
- **Class imbalance**: handled with `class_weight="balanced"` (LR, tree, forest), and PR-AUC / recall / precision instead of accuracy.
  (sklearn's GradientBoosting has no `class_weight`, which is why its recall at 0.5 is near 0 even though its PR-AUC is fine.)
- This synthetic model does not authorise any real credit decision.
