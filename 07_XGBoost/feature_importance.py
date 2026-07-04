import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

def load_xgboost_model():
    model_path = (Path(__file__).resolve().parents[1] / "09_GUI Application" / "saved_models" / "XGBoost.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    print(f"Loading model:\n{model_path.name}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data["feature_names"]
    return model, feature_names


# ============================================================
# LOAD PIPELINE (OPTIONAL CHECK)
# ============================================================

def load_pipeline():
    preprocess_dir = (Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline")
    pipeline_files = sorted(preprocess_dir.glob("preprocessing_pipeline_*.pkl"))

    if len(pipeline_files) == 0:
        raise FileNotFoundError("No preprocessing pipeline found.")

    pipeline_path = pipeline_files[-1]
    print(f"Loading pipeline:\n{pipeline_path.name}")

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    return pipeline


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model, feature_names):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    importance_df = (importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True))
    return importance_df


# ============================================================
# HORIZONTAL BAR GRAPH
# ============================================================

def plot_feature_importance(importance_df, top_n=20):
    plot_df = (importance_df.head(top_n).sort_values(by="Importance", ascending=True))
    plt.figure(figsize=(11, 8))
    bars = plt.barh(plot_df["Feature"], plot_df["Importance"])
    plt.title(f"Top {top_n} XGBoost Feature Importance", fontsize=14, fontweight="bold")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.30)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = (Path(__file__).resolve().parent / "xgboost_feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to\n{output_path}")


# ============================================================
# PRINT FEATURE RANKING
# ============================================================

def print_feature_ranking(importance_df, top_n=20):
    print("\n" + "=" * 70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 70)

    top_features = importance_df.head(top_n)

    print(top_features.to_string(index=False))
    print("=" * 70)

    csv_path = (Path(__file__).resolve().parent / "xgboost_feature_importance.csv")
    top_features.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to\n{csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 70)

    try:
        pipeline = load_pipeline()
        print(f"\nPipeline Features : {len(pipeline['feature_names'])}")
        model, feature_names = load_xgboost_model()
        print(f"Loaded Model : {type(model).__name__}")
        print(f"Loaded Features : {len(feature_names)}")

        if len(feature_names) != len(model.feature_importances_):
            raise ValueError(
                f"""
                Feature count mismatch!
                Model Importance Length : {len(model.feature_importances_)}
                Feature Name Length     : {len(feature_names)}
                """
            )

        importance_df = get_feature_importance(model, feature_names)
        print_feature_ranking(importance_df, top_n=20)
        plot_feature_importance(importance_df, top_n=20)
        print("\nAnalysis Completed Successfully.")

    except Exception as e:
        print("\nERROR")
        print("-" * 70)
        print(e)

        import traceback
        traceback.print_exc()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()