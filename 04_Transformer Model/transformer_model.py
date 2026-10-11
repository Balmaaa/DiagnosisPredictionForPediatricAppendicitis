import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FeatureEmbedding(nn.Module):    
    def __init__(self, feature_info, embed_dim=64):
        super(FeatureEmbedding, self).__init__()
        
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        
        self.numerical_feature_order = [f for f, info in feature_info.items() if info["type"] == "numerical"]
        self.categorical_feature_order = [f for f, info in feature_info.items() if info["type"] == "categorical"]
        
        self.numerical_embeddings = nn.ModuleDict({feat: nn.Linear(1, embed_dim) for feat in self.numerical_feature_order})
        self.categorical_embeddings = nn.ModuleDict({feat: nn.Embedding(feature_info[feat]["unique_values"], embed_dim) for feat in self.categorical_feature_order})
        
        self.numerical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.categorical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        
        total_features = len(feature_info)
        self.feature_positional_encoding = nn.Parameter(torch.randn(total_features, embed_dim) * 0.1)
        
    def forward(self, x_dict):
        embeddings = []
        
        if 'numerical' in x_dict:
            numerical_data = x_dict['numerical']            
            for i, feat_name in enumerate(self.numerical_feature_order):
                if i < numerical_data.shape[1]:
                    feat_values = numerical_data[:, i:i+1]
                    if feat_name in self.numerical_embeddings:
                        num_embed = self.numerical_embeddings[feat_name](feat_values)
                        num_embed = num_embed.unsqueeze(1)
                        num_embed = num_embed + self.numerical_token.unsqueeze(0)
                        embeddings.append(num_embed)
        
        for feat_name in self.categorical_feature_order:
            if feat_name in x_dict:
                feat_values = x_dict[feat_name]
                cat_embed = self.categorical_embeddings[feat_name](feat_values)
                cat_embed = cat_embed.unsqueeze(1)
                cat_embed = cat_embed + self.categorical_token.unsqueeze(0)
                embeddings.append(cat_embed)
        
        if embeddings:
            x = torch.cat(embeddings, dim=1)
            seq_len = x.size(1)
            pos_enc = self.feature_positional_encoding[:seq_len].unsqueeze(0)
            x = x + pos_enc
        else:
            x = torch.zeros(x_dict["numerical"].size(0), 1, self.embed_dim, device=x_dict["numerical"].device)
        return x

class AdvancedTabularTransformer(nn.Module):    
    def __init__(self, feature_info, embed_dim=64, num_heads=8, num_layers=4, dropout=0.3):
        super(AdvancedTabularTransformer, self).__init__()
        
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_embedding = FeatureEmbedding(feature_info, embed_dim)
        self.num_heads = min(num_heads, max(1, embed_dim // 32))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(embed_dim // 4, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x_dict):
        x = self.feature_embedding(x_dict)
        x = self.transformer(x)
        x_pooled = torch.mean(x, dim=1)
        output = self.classifier(x_pooled)
        return output

class AppendicitisDatasetDict(Dataset):    
    def __init__(self, features_dict, targets):
        self.features_dict = features_dict
        if hasattr(targets, 'values'):
            targets = targets.values
        self.targets = torch.LongTensor(np.array(targets, dtype=np.int64))
        
        first_key = list(features_dict.keys())[0]
        self.batch_size = features_dict[first_key].size(0)
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        item = {}
        for key, value in self.features_dict.items():
            item[key] = value[idx]
        return item, self.targets[idx]

class TransformerTrainer:    
    def __init__(self, model, device='cpu', pos_weight=None):
        self.model = model.to(device)
        self.device = device
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
            print(f"[BCEWithLogitsLoss] pos_weight={pos_weight:.4f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print("[BCEWithLogitsLoss] no class weighting")
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.temperature = 1.0
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features_dict, targets in dataloader:
            targets = targets.to(self.device)
            features_dict = {k: v.to(self.device) for k, v in features_dict.items()}
            
            self.optimizer.zero_grad()
            logits = self.model(features_dict).squeeze(-1)
            targets_float = targets.float()
            loss = self.criterion(logits, targets_float)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).long()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        return total_loss / len(dataloader), 100 * correct / total
    
    def evaluate(self, dataloader, debug=False):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for features_dict, targets in dataloader:
                targets = targets.to(self.device)
                features_dict = {k: v.to(self.device) for k, v in features_dict.items()}
                
                logits = self.model(features_dict).squeeze(-1)
                targets_float = targets.float()
                loss = self.criterion(logits, targets_float)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits / self.temperature)
                predicted = (probs > 0.5).long()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        if debug:
            logits_arr = np.array(all_logits)
            probs_arr = np.array(all_probs)
            print(f"  [DEBUG] Logits - min:{logits_arr.min():.4f} max:{logits_arr.max():.4f} "
                  f"mean:{logits_arr.mean():.4f} std:{logits_arr.std():.4f}")
            print(f"  [DEBUG] Probs  - min:{probs_arr.min():.4f} max:{probs_arr.max():.4f} "
                  f"mean:{probs_arr.mean():.4f} std:{probs_arr.std():.4f}")
        
        return total_loss / len(dataloader), 100 * correct / total, all_predictions, all_targets
    

    def calibrate_temperature(self, val_loader):
        self.model.eval()
        nll_criterion = nn.BCEWithLogitsLoss()
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for features_dict, targets in val_loader:
                targets = targets.to(self.device)
                features_dict = {k: v.to(self.device) for k, v in features_dict.items()}
                logits = self.model(features_dict).squeeze(-1)
                all_logits.append(logits)
                all_targets.append(targets.float())

        if not all_logits:
            raise ValueError("Temperature calibration failed: validation loader is empty.")

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        log_temperature = torch.zeros(1, requires_grad=True, device=self.device)
        optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            scaled_logits = all_logits / temperature
            loss = nll_criterion(scaled_logits, all_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        temperature = torch.exp(log_temperature).item()

        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"Invalid calibrated temperature: {temperature}")

        self.temperature = temperature
        print(f"[CALIBRATION] Temperature scaled: {self.temperature:.4f}")
        return self.temperature

    def train(self, train_loader, val_loader, epochs=50, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        best_model_path = Path(__file__).resolve().parent / "best_advanced_transformer_model.pth"

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, debug=(epoch % 10 == 0))

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  [BEST MODEL] Validation loss improved to {best_val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if not best_model_path.exists():
            raise FileNotFoundError(f"Best Transformer checkpoint not found:\n{best_model_path}")

        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        print(f"\n[BEST CHECKPOINT RESTORED]\n{best_model_path}")
        print("\n[CALIBRATION] Running temperature scaling...")
        self.calibrate_temperature(val_loader)

        return training_history


def calculate_medical_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    sensitivity = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'accuracy': accuracy, 'precision': precision, 'sensitivity': sensitivity,
        'specificity': specificity, 'ppv': ppv, 'npv': npv,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def load_preprocessed_data():
    PREPROCESS_DIR = Path(__file__).resolve().parents[1] / "03_Preprocessing Pipeline"
    pipeline_files = sorted(PREPROCESS_DIR.glob("transformer_pipeline_*.pkl"))
    pipeline_file = pipeline_files[-1]

    with open(pipeline_file, "rb") as f:
        pipeline = pickle.load(f)

    X_train = pd.read_csv(PREPROCESS_DIR / "transformer_X_train.csv")
    X_val   = pd.read_csv(PREPROCESS_DIR / "transformer_X_val.csv")
    X_test = pd.read_csv(PREPROCESS_DIR / "transformer_X_test.csv")
    y_train = pd.read_csv(PREPROCESS_DIR / "transformer_y_train.csv").squeeze()
    y_val   = pd.read_csv(PREPROCESS_DIR / "transformer_y_val.csv").squeeze()
    y_test = pd.read_csv(PREPROCESS_DIR / "transformer_y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test, pipeline

def main():    
    print("=" * 80)
    print("ADVANCED TRANSFORMER MODEL FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, pipeline = load_preprocessed_data()
        
        feature_info = pipeline["feature_info"]
        numerical_features = pipeline["numerical_features"]
        categorical_features = pipeline["categorical_features"]
        missing_indicators = pipeline.get("missing_indicator_columns", [])
        
        # Include missing indicator columns into numerical features so they are processed correctly
        all_numerical = numerical_features + [col for col in missing_indicators if col in X_train.columns]

        X_train_dict = {}
        X_val_dict = {}
        X_test_dict = {}
            
        for feat in categorical_features:
            if feat in X_train.columns:
                X_train_dict[feat] = torch.tensor(X_train[feat].values, dtype=torch.long)
                X_val_dict[feat] = torch.tensor(X_val[feat].values, dtype=torch.long)
                X_test_dict[feat] = torch.tensor(X_test[feat].values, dtype=torch.long)
                    
        X_train_dict["numerical"] = torch.tensor(X_train[all_numerical].values, dtype=torch.float32)
        X_val_dict["numerical"] = torch.tensor(X_val[all_numerical].values, dtype=torch.float32)
        X_test_dict["numerical"] = torch.tensor(X_test[all_numerical].values, dtype=torch.float32)

        print(f"Numerical features (including missing indicators): {len(all_numerical)}")
        print(f"Categorical features: {len(X_train_dict) - 1}")
            
        y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        n_negative = int(np.sum(y_train_arr == 0))
        n_positive = int(np.sum(y_train_arr == 1))
        pos_weight = n_negative / max(n_positive, 1)
        print(f"Class distribution - No Appendicitis: {n_negative}, Appendicitis: {n_positive}")
        print(f"pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}")

        model_params = {'embed_dim': 128, 'num_heads': 8, 'num_layers': 6, 'dropout': 0.2}
        training_params = {'batch_size': 32, 'learning_rate': 0.0005, 'weight_decay': 1e-3}
            
        model = AdvancedTabularTransformer(feature_info=feature_info, **model_params)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        train_dataset = AppendicitisDatasetDict(X_train_dict, y_train)
        val_dataset = AppendicitisDatasetDict(X_val_dict, y_val)
        test_dataset = AppendicitisDatasetDict(X_test_dict, y_test)
            
        train_loader = DataLoader(train_dataset, batch_size=training_params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_params["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=training_params["batch_size"], shuffle=False)
            
        trainer = TransformerTrainer(model, device, pos_weight=pos_weight)
        trainer.optimizer = optim.AdamW(model.parameters(), lr=training_params['learning_rate'], weight_decay=training_params['weight_decay'])
            
        print(f"\nTraining advanced transformer model...")
        training_history = trainer.train(train_loader, val_loader, epochs=100, patience=15)

        print(f"\n[FINAL EVALUATION]")
        test_loss, test_acc, predictions, targets = trainer.evaluate(test_loader, debug=True)

        metrics = calculate_medical_metrics(targets, predictions)
            
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"\nMedical Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  PPV: {metrics['ppv']:.4f}")
        print(f"  NPV: {metrics['npv']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {metrics['tp']}")
        print(f"  True Negatives: {metrics['tn']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
            
        gui_checkpoint_path = Path(__file__).parent.parent / '09_GUI Application' / 'saved_models' / 'Transformer.pt'
        gui_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": model.state_dict(),
            "feature_info": feature_info,
            "feature_order": list(feature_info.keys()),
            "numerical_features": all_numerical,
            "categorical_features": categorical_features,
            "label_encoders": pipeline["label_encoders"],
            "scaler": pipeline["scaler"],
            "embed_dim": model_params["embed_dim"],
            "num_heads": model_params["num_heads"],
            "num_layers": model_params["num_layers"],
            "dropout": model_params["dropout"],
            "temperature": trainer.temperature,
            "pos_weight": pos_weight
        }, gui_checkpoint_path)
        print(f"GUI checkpoint saved to: {gui_checkpoint_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()