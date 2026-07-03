import copy
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score
from transformer_model import AdvancedTabularTransformer


###############################################################################
# DATASET
###############################################################################

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
        item = {}
        for key, value in self.features_dict.items():
            item[key] = value[idx]
        return item, self.targets[idx]


###############################################################################
# LOAD PREPROCESSING PIPELINE
###############################################################################

def load_pipeline():
    preprocess_dir = (Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline")

    pipeline_files = sorted(preprocess_dir.glob("transformer_pipeline_*.pkl"))

    if len(pipeline_files) == 0:
        raise FileNotFoundError("No preprocessing pipeline found.")

    pipeline_file = pipeline_files[-1]
    print(f"Loading pipeline: {pipeline_file.name}")
    with open(pipeline_file, "rb") as f:
        pipeline = pickle.load(f)

    X_test = pd.read_csv(preprocess_dir / "transformer_X_test.csv")
    y_test = pd.read_csv(preprocess_dir / "transformer_y_test.csv").squeeze()
    return X_test, y_test, pipeline


###############################################################################
# LOAD TRAINED TRANSFORMER
###############################################################################

def load_model(device):
    checkpoint_path = (Path(__file__).resolve().parents[1] / "09_GUI Application" / "saved_models" / "Transformer.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AdvancedTabularTransformer(
        feature_info=checkpoint["feature_info"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Transformer loaded successfully.")
    return model


###############################################################################
# DATAFRAME -> MODEL INPUT
###############################################################################

def dataframe_to_dict(dataframe, numerical_features, categorical_features):
    feature_dict = {}
    feature_dict["numerical"] = torch.tensor(dataframe[numerical_features].values, dtype=torch.float32)

    for feature in categorical_features:
        feature_dict[feature] = torch.tensor(dataframe[feature].values, dtype=torch.long)

    return feature_dict


###############################################################################
# PREDICTION FUNCTION
###############################################################################

def predict(model, feature_dict, device="cpu", batch_size=64):
    dummy_targets = np.zeros(len(feature_dict["numerical"]))
    dataset = AppendicitisDatasetDict(feature_dict, dummy_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval()

    with torch.no_grad():
        for features, _ in loader:
            features = {k: v.to(device) for k, v in features.items()}
            logits = model(features).squeeze(1)
            probabilities = torch.sigmoid(logits)
            preds = (probabilities >= 0.5).long()
            predictions.extend(preds.cpu().numpy())

    return np.array(predictions)

# ============================================================
# PERMUTATION FEATURE IMPORTANCE
# ============================================================

def permutation_feature_importance(model, X_dict, y_true, feature_names, numerical_features, device="cpu", n_repeats=5):
    print("\nComputing permutation feature importance...")
    baseline_predictions = predict(model, X_dict, device)
    baseline_recall = recall_score(y_true, baseline_predictions)
    print(f"Baseline Recall : {baseline_recall:.4f}")
    importance_scores = []

    for feature in feature_names:
        scores = []
        for _ in range(n_repeats):
            permuted = copy.deepcopy(X_dict)

            if feature in numerical_features:
                idx = numerical_features.index(feature)
                shuffled = permuted["numerical"].clone()
                permutation = torch.randperm(shuffled.size(0))
                shuffled[:, idx] = shuffled[permutation, idx]
                permuted["numerical"] = shuffled

            ####################################################
            # Categorical feature
            ####################################################
            else:
                shuffled = permuted[feature].clone()
                permutation = torch.randperm(shuffled.size(0))
                shuffled = shuffled[permutation]
                permuted[feature] = shuffled

            predictions = predict(model, permuted, device)
            recall = recall_score(y_true, predictions)
            scores.append(baseline_recall - recall)

        importance_scores.append(np.mean(scores))

        print(f"{feature:<35}" f"{importance_scores[-1]:.5f}")

    importance_scores = np.array(importance_scores)
    importance_scores[importance_scores < 0] = 0

    if importance_scores.sum() > 0:
        importance_scores /= importance_scores.sum()

    return importance_scores


# ============================================================
# PLOT FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(importance, feature_names, top_n=15):

    df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    df = df.sort_values(by="Importance", ascending=False)
    top_df = df.head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(top_df["Feature"][::-1], top_df["Importance"][::-1], color="purple")
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.title("Transformer Feature Importance", fontsize=15, fontweight="bold")

    for i, value in enumerate(top_df["Importance"][::-1]):
        plt.text(value + 0.001, i, f"{value:.3f}", va="center")

    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    output_path = (Path(__file__).parent / "transformer_feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to:\n{output_path}")
    plt.show()
    return df


# ============================================================
# PRINT RANKING
# ============================================================

def print_ranking(df):
    print("\n")
    print("=" * 80)
    print("TRANSFORMER FEATURE IMPORTANCE")
    print("=" * 80)

    for i, row in enumerate(df.itertuples()):
        print(f"{i+1:2d}. " f"{row.Feature:<35}" f"{row.Importance:.4f}")

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("TRANSFORMER FEATURE IMPORTANCE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_test, y_test, pipeline = load_pipeline()
    numerical_features = pipeline["numerical_features"]
    categorical_features = pipeline["categorical_features"]
    feature_order = pipeline["feature_order"]
    X_test_dict = dataframe_to_dict(X_test, numerical_features, categorical_features)
    model = load_model(device)
    importance = permutation_feature_importance(
        model=model,
        X_dict=X_test_dict,
        y_true=np.array(y_test),
        feature_names=feature_order,
        numerical_features=numerical_features,
        device=device,
        n_repeats=5
    )

    ranking = plot_feature_importance(importance, feature_order, top_n=15)
    print_ranking(ranking)
    csv_path = (Path(__file__).parent / "transformer_feature_importance.csv")
    ranking.to_csv(csv_path, index=False)
    print(f"\nCSV saved to:\n{csv_path}")


if __name__ == "__main__":
    main()