import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

SEED = 42
np.random.seed(SEED)


class DecisionTreeFeatureImportanceAnalyzer:
    def __init__(self):
        self.preprocess_dir = Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline"
        self.gui_dir = Path(__file__).resolve().parents[1] / "08_GUI Application" / "saved_models"
        self.model = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None

    def load_data_and_model(self):
        pipeline_files = sorted(self.preprocess_dir.glob("preprocessing_pipeline_*.pkl"))
        if not pipeline_files:
            raise FileNotFoundError("No preprocessing pipeline found.")
        
        with open(pipeline_files[-1], "rb") as f:
            pipeline = pickle.load(f)
        self.feature_names = pipeline["feature_names"]

        self.X_test = pd.read_csv(self.preprocess_dir / "X_test.csv")
        self.y_test = pd.read_csv(self.preprocess_dir / "y_test.csv").values.ravel()

        model_path = self.gui_dir / "DecisionTree.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Decision Tree model not found at {model_path}. Please run training first.")
            
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            
        self.model = model_data["model"]
        print("Decision Tree model and test datasets loaded successfully.")

    def compute_permutation_importance(self, n_repeats=10):
        print("\nComputing Permutation Feature Importance for Decision Tree (Recall Drop)...")
        result = permutation_importance(
            estimator=self.model,
            X=self.X_test,
            y=self.y_test,
            scoring="recall",
            n_repeats=n_repeats,
            random_state=SEED,
            n_jobs=-1
        )

        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": result.importances_mean,
            "Std": result.importances_std
        })
        
        # Group and deduplicate split features (taking max importance and mean std)
        df = df.groupby("Feature", as_index=False).agg({"Importance": "max", "Std": "mean"})
        df = df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df

    def plot_and_save(self, df, top_n=15):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.clear()
        
        top_df = df.head(top_n).iloc[::-1]

        bars = ax.barh(
            top_df["Feature"], 
            top_df["Importance"], 
            xerr=top_df["Std"], 
            color="skyblue", 
            edgecolor="steelblue",
            linewidth=1.2,
            alpha=0.9, 
            capsize=4
        )

        ax.set_xlabel("Mean Recall Drop (Permutation Importance)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Clinical Feature", fontsize=11, fontweight="bold")
        ax.set_title("Decision Tree Model - Feature Importance", fontsize=14, fontweight="bold", pad=15)

        max_val = top_df["Importance"].max()
        if max_val == 0:
            max_val = 0.01
            
        ax.set_xlim(0, max_val * 1.30)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar, row in zip(bars, top_df.itertuples()):
            text_val = f"{row.Importance:.4f}"
            y_center = bar.get_y() + bar.get_height() / 2
            ax.text(
                row.Importance + (max_val * 0.02), 
                y_center, 
                text_val, 
                va="center", 
                ha="left", 
                fontsize=9, 
                fontweight="semibold",
                color="black"
            )

        fig.tight_layout()

        script_dir = Path(__file__).parent
        img_path = script_dir / "decision_tree_feature_importance.png"
        csv_path = script_dir / "decision_tree_feature_importance.csv"

        if img_path.exists():
            img_path.unlink()

        fig.savefig(img_path, dpi=300, bbox_inches="tight")
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 60)
        print(f"Figure successfully saved to: {img_path}")
        print(f"CSV data successfully saved to: {csv_path}")
        print("=" * 60)

        plt.close(fig)


def main():
    analyzer = DecisionTreeFeatureImportanceAnalyzer()
    analyzer.load_data_and_model()
    importance_df = analyzer.compute_permutation_importance(n_repeats=10)
    analyzer.plot_and_save(importance_df, top_n=15)


if __name__ == "__main__":
    main()