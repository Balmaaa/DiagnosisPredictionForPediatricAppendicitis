import pandas as pd
import numpy as np
import random
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# GRADIENT BOOSTING MODEL (PIPELINE-COMPATIBLE)
# ============================================================

class GradientBoostingModel:
    def __init__(self):
        print("USING UPDATED GRADIENT BOOSTING (PIPELINE-COMPATIBLE)")
        self.model = None
        self.feature_names = None
        self.pipeline = None

    # ============================================================
    # LOAD PIPELINE DATA
    # ============================================================

    def load_pipeline_data(self):
        preprocess_dir = Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline"
        pipeline_files = sorted(preprocess_dir.glob("preprocessing_pipeline_*.pkl"))
        if not pipeline_files:
            raise FileNotFoundError("No preprocessing pipeline found.")

        pipeline_path = pipeline_files[-1]

        print(f"Loading pipeline: {pipeline_path.name}")
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        self.pipeline = pipeline
        self.feature_names = pipeline["feature_names"]

        print("\nFEATURES USED BY THE MODEL")
        print("-" * 70)

        for feature in self.feature_names:
            print(feature)

        print("-" * 70)
        print(f"Total Features: {len(self.feature_names)}")

        X_train = pd.read_csv(preprocess_dir / "X_train.csv")
        X_test = pd.read_csv(preprocess_dir / "X_test.csv")
        y_train = pd.read_csv(preprocess_dir / "y_train.csv").values.ravel()
        y_test = pd.read_csv(preprocess_dir / "y_test.csv").values.ravel()

        print("\nDataset Information")
        print("-" * 70)
        print(f"Training samples : {len(X_train)}")
        print(f"Testing samples  : {len(X_test)}")
        print(f"Features         : {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    # ============================================================
    # TRAIN MODEL
    # ============================================================

    def train(self, X_train, y_train, use_hyperparameter_tuning=False):
        print("\nTraining Gradient Boosting...")

        if use_hyperparameter_tuning:
            self.model = self.hyperparameter_tuning(X_train, y_train)
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=SEED
            )
            self.model.fit(X_train, y_train)

        print("Training completed.")
        print(f"Number of estimators: {self.model.n_estimators_}")
        return self.model

    # ============================================================
    # (OPTIONAL) SIMPLE TUNING
    # ============================================================

    def hyperparameter_tuning(self, X_train, y_train):
        print("Performing hyperparameter tuning...")
        param_grid = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5], "subsample": [0.8, 1.0]}
        gb = GradientBoostingClassifier(random_state=SEED)
        grid = GridSearchCV(gb, param_grid, cv=5, scoring="recall", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_

        print("\nBest Params:", grid.best_params_)
        print("Best CV Recall:", grid.best_score_)

        return self.model

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate(self, X_test, y_test, threshold=0.5):
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }

    # ============================================================
    # SAVE MODEL (GUI COMPATIBLE)
    # ============================================================

    def save_model(self):
        save_path = Path(__file__).resolve().parents[1] / "09_GUI Application" / "saved_models"
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / "GradientBoosting.pkl"
        model_data = {"model": self.model, "feature_names": self.feature_names}

        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to GUI folder: {file_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GRADIENT BOOSTING - PIPELINE VERSION")
    print("=" * 70)

    gb = GradientBoostingModel()
    X_train, X_test, y_train, y_test = gb.load_pipeline_data()
    gb.train(X_train, y_train, use_hyperparameter_tuning=True)

    train_predictions = gb.model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    print("\n" + "=" * 70)
    print("TRAINING PERFORMANCE")
    print("=" * 70)
    print(f"Training Accuracy : {train_accuracy:.4f}")

    metrics = gb.evaluate(X_test, y_test)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f}")
    print(f"Recall       : {metrics['recall']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(f"PPV          : {metrics['ppv']:.4f}")
    print(f"NPV          : {metrics['npv']:.4f}")
    print("CONFUSION MATRIX:")
    print(f"TP={metrics['tp']}")
    print(f"TN={metrics['tn']}")
    print(f"FP={metrics['fp']}")
    print(f"FN={metrics['fn']}")

    gb.save_model()


if __name__ == "__main__":
    main()