import pandas as pd
import numpy as np
import pickle
import random
from pathlib import Path
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# DECISION TREE MODEL (GUI VERSION)
# ============================================================

class DecisionTreeModel:
    def __init__(self):
        print("USING GUI-COMPATIBLE DECISION TREE MODEL")
        self.model = None
        self.feature_names = None

    # ============================================================
    # LOAD PREPROCESSED DATA (PIPELINE OUTPUT)
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

        self.feature_names = pipeline["feature_names"]

        X_train = pd.read_csv(preprocess_dir / "X_train.csv")
        X_test = pd.read_csv(preprocess_dir / "X_test.csv")
        y_train = pd.read_csv(preprocess_dir / "y_train.csv").values.ravel()
        y_test = pd.read_csv(preprocess_dir / "y_test.csv").values.ravel()

        return X_train, X_test, y_train, y_test

    # ============================================================
    # TRAIN MODEL
    # ============================================================

    def train(self, X_train, y_train):
        print("\nTraining Decision Tree...")

        self.model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED
        )

        self.model.fit(X_train, y_train)

        print("Training completed.")
        print(f"Tree depth: {self.model.get_depth()}")
        print(f"Number of leaves: {self.model.get_n_leaves()}")

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

        metrics = {
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

        return metrics, preds

    # ============================================================
    # SAVE FOR GUI (IMPORTANT)
    # ============================================================

    def save_model(self):
        model_data = {"model": self.model, "feature_names": self.feature_names, "model_type": "decision_tree", "threshold": 0.5}
        save_path = Path(__file__).resolve().parents[1] / "09_GUI Application" / "saved_models" / "DecisionTree.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved for GUI: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DECISION TREE - GUI COMPATIBLE VERSION")
    print("=" * 70)

    dt = DecisionTreeModel()
    X_train, X_test, y_train, y_test = dt.load_pipeline_data()
    dt.train(X_train, y_train)
    metrics, preds = dt.evaluate(X_test, y_test)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f}")
    print(f"Recall       : {metrics['recall']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(f"PPV          : {metrics['ppv']:.4f}")
    print(f"NPV          : {metrics['npv']:.4f}")

    dt.save_model()


if __name__ == "__main__":
    main()