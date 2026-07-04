import pandas as pd
import numpy as np
import random
import pickle
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix)
import matplotlib.pyplot as plt

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# XGBOOST MODEL (PIPELINE-COMPATIBLE)
# ============================================================

class XGBoostModel:
    def __init__(self):
        print("USING UPDATED XGBOOST (PIPELINE-COMPATIBLE)")
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.best_params = None

    # ============================================================
    # LOAD PIPELINE DATA
    # ============================================================

    def load_pipeline_data(self):
        preprocess_dir = (Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline")
        pipeline_files = sorted(preprocess_dir.glob("preprocessing_pipeline_*.pkl"))

        if len(pipeline_files) == 0:
            raise FileNotFoundError("No preprocessing pipeline found.")

        pipeline_path = pipeline_files[-1]
        print(f"\nLoading pipeline: {pipeline_path.name}")

        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        self.pipeline = pipeline
        self.feature_names = pipeline["feature_names"]

        print("\nFEATURES USED BY THE MODEL")
        print("-" * 70)

        for feature in self.feature_names:
            print(feature)

        print("-" * 70)
        print(f"Total Features : {len(self.feature_names)}")

        X_train = pd.read_csv(preprocess_dir / "X_train.csv")
        X_test = pd.read_csv(preprocess_dir / "X_test.csv")
        y_train = (pd.read_csv(preprocess_dir / "y_train.csv").values.ravel())
        y_test = (pd.read_csv(preprocess_dir / "y_test.csv").values.ravel())

        print("\nDataset Information")
        print("-" * 70)
        print(f"Training Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")
        print(f"Features         : {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    # ============================================================
    # TRAIN MODEL
    # ============================================================

    def train(self, X_train, y_train, use_hyperparameter_tuning=True,):
        print("\nTraining XGBoost...")
        if use_hyperparameter_tuning:
            self.model = self.hyperparameter_tuning(X_train, y_train)
        else:
            self.model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=3,
                subsample=0.80,
                colsample_bytree=0.80,
                gamma=0.10,
                random_state=SEED,
                tree_method="hist",
                n_jobs=-1,
            )

            self.model.fit(X_train, y_train)

        print("Training completed.")
        print(f"Number of Trees : {self.model.n_estimators}")
        return self.model

    # ============================================================
    # HYPERPARAMETER TUNING
    # ============================================================

    def hyperparameter_tuning(self, X_train, y_train):
        print("\nPerforming Hyperparameter Tuning...")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.10],
            "max_depth": [3, 4, 5],
            "subsample": [0.80, 1.00],
            "colsample_bytree": [0.80, 1.00],
        }

        classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1)
        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring="recall", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        self.best_params = grid.best_params_
        self.model = grid.best_estimator_

        print("\nBest Parameters")
        print("-" * 70)

        for key, value in self.best_params.items():
            print(f"{key:<20}: {value}")

        print("-" * 70)
        print(f"Best CV Recall : {grid.best_score_:.4f}")
        return self.model

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate(self, X_test, y_test, threshold=0.50):
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv = (tp / (tp + fp) if (tp + fp) > 0 else 0)
        npv = (tn / (tn + fn) if (tn + fn) > 0 else 0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    def get_feature_importance(self):
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({"Feature": self.feature_names, "Importance": importance})
        importance_df = (importance_df.sort_values(by="Importance", ascending=False,).reset_index(drop=True))
        return importance_df

    # ============================================================
    # PERMUTATION FEATURE IMPORTANCE
    # ============================================================

    def permutation_feature_importance(self, X_test, y_test, repeats=10):
        print("\nCalculating Permutation Feature Importance...")
        result = permutation_importance(
            estimator=self.model,
            X=X_test,
            y=y_test,
            scoring="recall",
            n_repeats=repeats,
            random_state=SEED,
            n_jobs=-1,
        )

        importance_df = pd.DataFrame({"Feature": self.feature_names, "Importance": result.importances_mean, "Std": result.importances_std})
        importance_df = (importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True))
        print("\nPermutation Feature Importance")
        print(importance_df.head(15))
        return importance_df

    # ============================================================
    # PLOT FEATURE IMPORTANCE
    # ============================================================

    def plot_feature_importance(self, importance_df, top_n=15):
        plot_df = (importance_df.head(top_n).sort_values(by="Importance"))
        plt.figure(figsize=(10, 7))
        plt.barh(plot_df["Feature"], plot_df["Importance"])
        plt.xlabel("Importance")
        plt.title("XGBoost Feature Importance")
        save_path = (Path(__file__).resolve().parent / "xgboost_feature_importance.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved figure to {save_path}")

    # ============================================================
    # SAVE MODEL (GUI COMPATIBLE)
    # ============================================================

    def save_model(self):
        save_path = (Path(__file__).resolve().parents[1] / "09_GUI Application" / "saved_models")
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = (save_path / "XGBoost.pkl")
        model_data = {"model": self.model, "feature_names": self.feature_names}

        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to GUI folder: {file_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("XGBOOST - PIPELINE VERSION")
    print("=" * 70)

    xgb_model = XGBoostModel()

    (X_train, X_test, y_train, y_test) = xgb_model.load_pipeline_data()
    xgb_model.train(X_train, y_train, use_hyperparameter_tuning=True)
    train_predictions = xgb_model.model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    print("\n" + "=" * 70)
    print("TRAINING PERFORMANCE")
    print("=" * 70)
    print(f"Training Accuracy : {train_accuracy:.4f}")

    metrics = xgb_model.evaluate(X_test, y_test)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f}")
    print(f"Recall       : {metrics['recall']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(f"PPV          : {metrics['ppv']:.4f}")
    print(f"NPV          : {metrics['npv']:.4f}")
    print("\nCONFUSION MATRIX")
    print(f"TP = {metrics['tp']}")
    print(f"TN = {metrics['tn']}")
    print(f"FP = {metrics['fp']}")
    print(f"FN = {metrics['fn']}")

    xgb_model.save_model()

    print("\n" + "=" * 70)
    print("XGBOOST TRAINING COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()