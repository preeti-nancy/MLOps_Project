"""Train and tune a Random Forest model with MLflow experiment tracking."""
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

ARTIFACT_DIR = "tourism_project/model_building/artifacts"
DEPLOY_DIR = "tourism_project/deployment"


def main():
    X_train = pd.read_csv(f"{ARTIFACT_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{ARTIFACT_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{ARTIFACT_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{ARTIFACT_DIR}/y_test.csv").squeeze()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    mlflow.set_experiment("Tourism_Purchase_Prediction")

    with mlflow.start_run(run_name="random_forest_gridsearch"):
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=1,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        os.makedirs(DEPLOY_DIR, exist_ok=True)
        joblib.dump(best_model, f"{DEPLOY_DIR}/model.pkl")
        joblib.dump(list(X_train.columns), f"{DEPLOY_DIR}/feature_columns.pkl")

        print("Best hyperparameters:", grid_search.best_params_)
        print(f"Best CV F1: {grid_search.best_score_:.4f}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
