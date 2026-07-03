import pandas as pd
import numpy as np
import random
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.inspection import permutation_importance

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# LOAD PIPELINE DATA
# ============================================================

def load_pipeline_data():
    preprocess_dir = Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline"
    X_test = pd.read_csv(preprocess_dir / "X_test.csv")
    y_test = pd.read_csv(preprocess_dir / "y_test.csv").values.ravel()
    return X_test, y_test


def get_feature_names():
    preprocess_dir = (Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline")
    pipeline_files = sorted(preprocess_dir.glob("preprocessing_pipeline_*.pkl"))
    if not pipeline_files:
        raise FileNotFoundError("No preprocessing pipeline found")

    pipeline_path = pipeline_files[-1]
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    return pipeline["feature_names"]


# ============================================================
# LOAD MODEL
# ============================================================

def load_gradient_boosting_model():
    model_path = (Path(__file__).resolve().parents[1] / "09_GUI Application" / "saved_models" / "GradientBoosting.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at:\n{model_path}\n\n" "Run GradientBoostingModel first to generate it.")

    print(f"Loading model: {model_path.name}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data.get("feature_names", None)
    return model, feature_names


# ============================================================
# PERMUTATION FEATURE IMPORTANCE (RECALL)
# ============================================================

def permutation_feature_importance(model, X_test, y_test, feature_names, top_n=15, n_repeats=30):
    print("\nComputing permutation feature importance (Recall)...")
    result = permutation_importance(estimator=model, X=X_test, y=y_test, scoring="recall", n_repeats=n_repeats, random_state=SEED, n_jobs=-1)
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean, "Std": result.importances_std})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    return importance_df.head(top_n)


# ============================================================
# PLOT IMPORTANCE
# ============================================================

def plot_permutation_importance(importance_df):
    plt.figure(figsize=(10, 8))
    plot_df = importance_df.iloc[::-1]
    plt.barh(plot_df["Feature"], plot_df["Importance"], xerr=plot_df["Std"], color="lightgreen", edgecolor="darkgreen", alpha=0.8)
    plt.xlabel("Mean Recall Decrease")
    plt.ylabel("Feature")
    plt.title("Gradient Boosting Permutation Feature Importance")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_path = Path(__file__).parent / "gradient_boosting_permutation_importance.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved figure to {save_path}")


# ============================================================
# PRINT RANKING
# ============================================================

def print_feature_ranking(df):
    print("\n")
    print("=" * 70)
    print("GRADIENT BOOSTING PERMUTATION FEATURE IMPORTANCE")
    print("=" * 70)

    for i, row in enumerate(df.itertuples()):
        print(f"{i+1:2d}. " f"{row.Feature:<35}" f"{row.Importance:.5f}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GRADIENT BOOSTING - PERMUTATION FEATURE IMPORTANCE")
    print("=" * 70)

    model, saved_feature_names = load_gradient_boosting_model()
    X_test, y_test = load_pipeline_data()

    if saved_feature_names is not None:
        feature_names = saved_feature_names
    else:
        feature_names = get_feature_names()

    print(f"\nNumber of features: {len(feature_names)}")

    importance_df = permutation_feature_importance(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, top_n=15, n_repeats=50)
    plot_permutation_importance(importance_df)
    print_feature_ranking(importance_df)
    csv_path = Path(__file__).parent / "gradient_boosting_permutation_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    print(f"\nCSV saved to:\n{csv_path}")


if __name__ == "__main__":
    main()