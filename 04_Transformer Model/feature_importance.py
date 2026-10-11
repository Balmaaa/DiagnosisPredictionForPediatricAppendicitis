import sys
from pathlib import Path
import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score
from transformer_model import AdvancedTabularTransformer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class AppendicitisDatasetDict(Dataset):
    def __init__(self, features_dict, targets):
        self.features_dict = features_dict
        if hasattr(targets, "values"):
            targets = targets.values
        self.targets = torch.LongTensor(np.asarray(targets, dtype=np.int64))
        first_key = list(features_dict.keys())[0]
        self.length = features_dict[first_key].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.features_dict.items()}
        return item, self.targets[idx]


class TransformerFeatureImportanceAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.preprocess_dir = Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline"
        self.gui_dir = Path(__file__).resolve().parents[1] / "09_GUI Application" / "saved_models"
        
        self.pipeline = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.feature_order = []
        self.all_numerical = []

    def load_pipeline_and_data(self):
        pipeline_files = sorted(self.preprocess_dir.glob("transformer_pipeline_*.pkl"))
        if not pipeline_files:
            raise FileNotFoundError("No Transformer preprocessing pipeline found.")
        
        pipeline_file = pipeline_files[-1]
        print(f"Loading pipeline: {pipeline_file.name}")
        with open(pipeline_file, "rb") as f:
            self.pipeline = pickle.load(f)

        self.X_test = pd.read_csv(self.preprocess_dir / "transformer_X_test.csv")
        self.y_test = pd.read_csv(self.preprocess_dir / "transformer_y_test.csv").squeeze()

        numerical = self.pipeline["numerical_features"]
        categorical = self.pipeline["categorical_features"]
        missing_indicators = self.pipeline.get("missing_indicator_columns", [])
        
        self.all_numerical = numerical + [col for col in missing_indicators if col in self.X_test.columns]
        self.feature_order = self.all_numerical + categorical
        print(f"Loaded test samples: {len(self.X_test)} | Total features tracked: {len(self.feature_order)}")

    def load_trained_model(self):
        checkpoint_path = self.gui_dir / "Transformer.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Transformer checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = AdvancedTabularTransformer(
            feature_info=checkpoint["feature_info"],
            embed_dim=checkpoint["embed_dim"],
            num_heads=checkpoint["num_heads"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print("Transformer model loaded successfully.")

    def _dataframe_to_dict(self, dataframe):
        feature_dict = {}
        feature_dict["numerical"] = torch.tensor(dataframe[self.all_numerical].values, dtype=torch.float32)
        for feature in self.pipeline["categorical_features"]:
            if feature in dataframe.columns:
                feature_dict[feature] = torch.tensor(dataframe[feature].values, dtype=torch.long)
        return feature_dict

    def predict(self, feature_dict, batch_size=64):
        dummy_targets = np.zeros(len(feature_dict["numerical"]))
        dataset = AppendicitisDatasetDict(feature_dict, dummy_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for features, _ in loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                logits = self.model(features).squeeze(1)
                probabilities = torch.sigmoid(logits)
                preds = (probabilities >= 0.5).long()
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def compute_permutation_importance(self, n_repeats=10):
        print("\nComputing Permutation Feature Importance (Recall Drop)...")
        X_dict = self._dataframe_to_dict(self.X_test)
        y_true = np.array(self.y_test)
        
        baseline_preds = self.predict(X_dict)
        baseline_recall = recall_score(y_true, baseline_preds, zero_division=0)
        print(f"Baseline Recall (Sensitivity): {baseline_recall:.4f}\n")

        importance_records = []

        for feature in self.feature_order:
            scores = []
            for _ in range(n_repeats):
                permuted = copy.deepcopy(X_dict)

                if feature in self.all_numerical:
                    idx = self.all_numerical.index(feature)
                    shuffled = permuted["numerical"].clone()
                    permutation = torch.randperm(shuffled.size(0))
                    shuffled[:, idx] = shuffled[permutation, idx]
                    permuted["numerical"] = shuffled
                elif feature in permuted:
                    shuffled = permuted[feature].clone()
                    permutation = torch.randperm(shuffled.size(0))
                    shuffled = shuffled[permutation]
                    permuted[feature] = shuffled
                else:
                    scores.append(0.0)
                    continue

                preds = self.predict(permuted)
                recall = recall_score(y_true, preds, zero_division=0)
                scores.append(baseline_recall - recall)

            mean_drop = np.mean(scores) if scores else 0.0
            std_drop = np.std(scores) if scores else 0.0
            importance_records.append({"Feature": feature, "Importance": max(0.0, mean_drop), "Std": std_drop})
            print(f"Processed feature: {feature:<35} | Importance Drop: {mean_drop:.5f}")

        df = pd.DataFrame(importance_records)
        
        # FIX: Group by feature name to combine any split occurrences and take the max importance
        df = df.groupby("Feature", as_index=False).agg({"Importance": "max", "Std": "mean"})
        
        df = df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df

    def plot_and_save(self, df, top_n=15):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.clear()
        
        top_df = df.head(top_n).iloc[::-1]  # Reverse for bottom-up horizontal bar layout

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
        ax.set_title("Transformer Model - Feature Importance", fontsize=14, fontweight="bold", pad=15)

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
        img_path = script_dir / "transformer_feature_importance.png"
        csv_path = script_dir / "transformer_feature_importance.csv"
        
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
    analyzer = TransformerFeatureImportanceAnalyzer()
    analyzer.load_pipeline_and_data()
    analyzer.load_trained_model()
    
    importance_df = analyzer.compute_permutation_importance(n_repeats=10)
    analyzer.plot_and_save(importance_df, top_n=15)


if __name__ == "__main__":
    main()