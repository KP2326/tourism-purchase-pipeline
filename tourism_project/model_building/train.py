# Tune an XGBoost model, log every trial to MLflow and save the best one.
import logging
import os
import warnings
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"  # keeps the logs readable
warnings.filterwarnings("ignore", module="mlflow")  # signature hints, not relevant here

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logging.getLogger("mlflow").setLevel(logging.ERROR)  # set after the import, mlflow resets it otherwise

MODEL_PATH = "tourism_project/deployment/best_model.joblib"

# use the MLflow server when one is running (the workflow starts one),
# otherwise fall back to a local ./mlruns folder
uri = os.getenv("MLFLOW_TRACKING_URI")
if uri:
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.search_experiments()
    except Exception:
        print(f"MLflow server at {uri} not reachable, logging locally instead")
        mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-wellness-package")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

num_cols = Xtrain.select_dtypes(include="number").columns.tolist()
cat_cols = Xtrain.select_dtypes(include="object").columns.tolist()

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), cat_cols),
])

# the classes are imbalanced (about 4 non-buyers per buyer), so scale_pos_weight
# is part of the search
neg_per_pos = float(round((ytrain == 0).sum() / (ytrain == 1).sum(), 1))

model = Pipeline([
    ("prep", preprocess),
    ("xgb", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1)),
])

param_grid = {
    "xgb__n_estimators": [200, 400],
    "xgb__max_depth": [6, 8, 10],
    "xgb__learning_rate": [0.05, 0.1, 0.2],
    "xgb__colsample_bytree": [0.5, 0.7],
    "xgb__scale_pos_weight": [1, neg_per_pos],
}

with mlflow.start_run(run_name="xgboost_grid_search"):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=2)
    grid.fit(Xtrain, ytrain)

    # one child run per parameter combination so every trial is visible in the UI
    results = grid.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"trial_{i:02d}", nested=True):
            mlflow.log_params({k.replace("xgb__", ""): v for k, v in params.items()})
            mlflow.log_metric("cv_f1_mean", results["mean_test_score"][i])
            mlflow.log_metric("cv_f1_std", results["std_test_score"][i])

    best = grid.best_estimator_
    mlflow.log_params({f"best_{k.replace('xgb__', '')}": v for k, v in grid.best_params_.items()})
    mlflow.log_metric("best_cv_f1", grid.best_score_)

    # evaluate on both splits
    for name, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        pred = best.predict(X)
        proba = best.predict_proba(X)[:, 1]
        mlflow.log_metrics({
            f"{name}_accuracy": accuracy_score(y, pred),
            f"{name}_precision": precision_score(y, pred),
            f"{name}_recall": recall_score(y, pred),
            f"{name}_f1": f1_score(y, pred),
            f"{name}_roc_auc": roc_auc_score(y, proba),
        })
        print(f"\n{name.upper()} classification report")
        print(classification_report(y, pred, target_names=["No purchase", "Purchase"]))

    print("Best parameters:", grid.best_params_)
    print(f"Best CV F1: {grid.best_score_:.4f}")

    mlflow.sklearn.log_model(best, name="model")

    # the workflow commits this file back to the repo for the Streamlit app
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print("Saved model to", MODEL_PATH)
