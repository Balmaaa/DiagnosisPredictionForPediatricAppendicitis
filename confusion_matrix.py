#!/usr/bin/env python3
"""
Confusion Matrix Generator for Pediatric Appendicitis Prediction System

This script loads the trained models and dataset to compute and display
confusion matrices for all models in the system.
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add model directories to path
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / "06_Decision_Trees"))
sys.path.insert(0, str(base_path / "07_Gradient_Boosting"))
sys.path.insert(0, str(base_path / "08_XGBoost"))
sys.path.insert(0, str(base_path / "05_Transformer_Model"))

# Import model classes
from decision_tree_model import DecisionTreeModel
from gradient_boosting_model import GradientBoostingModel
from xgboost_model import XGBoostModel

# Feature definitions (global 38-feature contract)
NUMERICAL_FEATURES = [
    'Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 'WBC_Count',
    'RBC_Count', 'Hemoglobin', 'RDW', 'Segmented_Neutrophils',
    'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite',
    'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 'Severity',
    'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness',
    'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine'
]

LAB_FIELDS = ['WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW',
              'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP',
              'Neutrophil_Percentage']

LAB_MISSING_INDICATORS = [f"{lab}_missing" for lab in LAB_FIELDS]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + LAB_MISSING_INDICATORS

 

# Transformer model classes (simplified versions)
class FeatureEmbedding(nn.Module):
    def __init__(self, feature_info, embed_dim=64):
        super().__init__()
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        numerical_features = [f for f, i in feature_info.items() if i['type'] == 'numerical']
        self.numerical_embeddings = nn.ModuleDict({f: nn.Linear(1, embed_dim) for f in numerical_features})
        self.categorical_embeddings = nn.ModuleDict({
            f: nn.Embedding(i['unique_values'], embed_dim)
            for f, i in feature_info.items() if i['type'] == 'categorical'
        })
        self.numerical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.categorical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.feature_positional_encoding = nn.Parameter(torch.randn(len(feature_info), embed_dim) * 0.1)

    def forward(self, x_dict):
        embeddings = []
        if 'numerical' in x_dict:
            data = x_dict['numerical']
            for i, feat in enumerate(self.numerical_embeddings.keys()):
                if i < data.shape[1]:
                    e = self.numerical_embeddings[feat](data[:, i:i+1]).unsqueeze(1)
                    embeddings.append(e + self.numerical_token.unsqueeze(0))
        for feat, val in x_dict.items():
            if feat != 'numerical' and feat in self.categorical_embeddings:
                e = self.categorical_embeddings[feat](val).unsqueeze(1)
                embeddings.append(e + self.categorical_token.unsqueeze(0))
        if embeddings:
            x = torch.cat(embeddings, dim=1)
            x = x + self.feature_positional_encoding[:x.size(1)].unsqueeze(0)
        else:
            x = torch.zeros(x_dict['numerical'].size(0), 1, self.embed_dim)
        return x

class AdvancedTabularTransformer(nn.Module):
    def __init__(self, feature_info, embed_dim=64, num_heads=8, num_layers=4, num_classes=2, dropout=0.3):
        super().__init__()
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_embedding = FeatureEmbedding(feature_info, embed_dim)
        self.num_heads = min(num_heads, embed_dim // 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=self.num_heads, dropout=dropout,
            dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, embed_dim // 4), nn.GELU(), nn.Dropout(dropout * 0.25),
            nn.Linear(embed_dim // 4, 1))

    def forward(self, x_dict):
        x = self.feature_embedding(x_dict)
        x = self.transformer(x)
        x_pooled = torch.mean(x, dim=1)
        return self.classifier(x_pooled)

class ConfusionMatrixGenerator:
    def __init__(self):
        self.models = {}
        self.X_test = None
        self.y_test = None
        self.feature_info = None
        
    def load_models(self):
        """Load all trained models from saved files"""
        print("Loading trained models...")
        
        # Model directory paths (same as in main system)
        model_dir = base_path / "09_GUI_Application" / "saved_models"
        
        # Load sklearn models from .pkl files
        pkl_models = {
            'Decision Tree': 'Decision_Tree.pkl',
            'Gradient Boosting': 'Gradient_Boosting.pkl',
            'XGBoost': 'XGBoost.pkl'
        }
        
        for model_name, filename in pkl_models.items():
            file_path = model_dir / filename
            if file_path.exists():
                try:
                    obj = pickle.load(open(file_path, 'rb'))
                    # Extract the actual model from the saved object
                    if isinstance(obj, dict):
                        model = None
                        for key in ['model', 'best_estimator_', 'trained_model']:
                            if key in obj:
                                model = obj[key]
                                break
                        if model is not None:
                            self.models[model_name] = model
                            print(f"[OK] Loaded {model_name}")
                    else:
                        self.models[model_name] = obj
                        print(f"[OK] Loaded {model_name}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {model_name}: {e}")
            else:
                print(f"[ERROR] Model file not found: {file_path}")
        
        # Load Transformer from .pt file
        transformer_path = model_dir / "Transformer.pt"
        if transformer_path.exists():
            try:
                ckpt = torch.load(transformer_path, map_location='cpu', weights_only=False)
                model = AdvancedTabularTransformer(
                    feature_info=ckpt['feature_info'], embed_dim=ckpt['embed_dim'],
                    num_heads=ckpt['num_heads'], num_layers=ckpt['num_layers'],
                    dropout=ckpt['dropout'], num_classes=2)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                self.models['Transformer'] = model
                self.feature_info = ckpt['feature_info']
                print(f"[OK] Loaded Transformer")
            except Exception as e:
                print(f"[ERROR] Failed to load Transformer: {e}")
        else:
            print(f"[ERROR] Transformer file not found: {transformer_path}")
        
        print(f"Models loaded: {list(self.models.keys())}")
    
    def load_dataset(self):
        """Load and preprocess the same dataset used for training"""
        print("\nLoading dataset...")
        
        try:
            # Use Decision Tree model to load unified data (same as in main system)
            dt_wrapper = DecisionTreeModel()
            X_train, X_test, y_train, y_test = dt_wrapper.load_unified_data()
            
            # Enforce the global 38-feature contract ordering
            X_train = X_train[ALL_FEATURES]
            X_test = X_test[ALL_FEATURES]
            
            # Store for use in predictions
            self.X_test = X_test
            self.y_test = y_test
            self.feature_columns = ALL_FEATURES
            
            print(f"[OK] Dataset loaded: X_test={X_test.shape}, y_test={y_test.shape}")
            print(f"[OK] Features: {len(self.feature_columns)} (38 unified features)")
            
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            raise
    
    def generate_predictions(self, threshold=0.5):
        """Generate predictions for all models using specified threshold"""
        print(f"\nGenerating predictions (threshold={threshold})...")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'Transformer':
                    # Transformer prediction
                    pred = self._predict_transformer(model, threshold)
                else:
                    # sklearn model prediction
                    pred = self._predict_sklearn(model, threshold)
                
                predictions[model_name] = pred
                print(f"[OK] {model_name}: {len(pred)} predictions")
                
            except Exception as e:
                print(f"[ERROR] {model_name} prediction failed: {e}")
                predictions[model_name] = None
        
        return predictions
    
    def _predict_sklearn(self, model, threshold):
        """Generate predictions for sklearn models"""
        # All sklearn models use the unified 38-feature vector in exact order
        X_test_38 = self.X_test[self.feature_columns]
        X_test_38_array = X_test_38.values
        
        # Get probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_38_array)
            prob_appendicitis = proba[:, 1]
        else:
            # Fallback to direct prediction
            pred = model.predict(X_test_38_array)
            return pred
        
        # Apply threshold
        pred = (prob_appendicitis >= threshold).astype(int)
        return pred
    
    def _predict_transformer(self, model, threshold):
        """Generate predictions for Transformer model"""
        # Build feature dict matching training format
        feat_to_idx = {f: i for i, f in enumerate(self.feature_columns)}
        
        # Extract and scale numerical features
        num_indices = [feat_to_idx[f] for f in NUMERICAL_FEATURES if f in feat_to_idx]
        num_block = self.X_test.iloc[:, num_indices].astype(np.float32)
        
        # Apply normalization (using simple standardization)
        num_block = (num_block - num_block.mean()) / (num_block.std() + 1e-8)
        
        X_dict = {'numerical': torch.FloatTensor(num_block.values)}
        
        # Extract categorical features
        for cat_feat in CATEGORICAL_FEATURES:
            if cat_feat in feat_to_idx and cat_feat in self.feature_info:
                val = self.X_test.iloc[:, feat_to_idx[cat_feat]].astype(int).values
                max_val = self.feature_info[cat_feat].get('unique_values', 2)
                val = np.clip(val, 0, max_val - 1)
                X_dict[cat_feat] = torch.LongTensor(val)
        
        # Get predictions
        with torch.no_grad():
            logits = model(X_dict).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            proba = torch.sigmoid(logits).numpy()
        
        # Apply threshold
        pred = (proba >= threshold).astype(int)
        return pred
    
    def compute_confusion_matrices(self, predictions):
        """Compute confusion matrices for all models"""
        print("\nComputing confusion matrices...")
        
        confusion_matrices = {}
        
        for model_name, pred in predictions.items():
            if pred is not None:
                try:
                    # Compute confusion matrix
                    cm = confusion_matrix(self.y_test, pred)
                    tn, fp, fn, tp = cm.ravel()
                    
                    confusion_matrices[model_name] = {
                        'TP': tp,
                        'TN': tn,
                        'FP': fp,
                        'FN': fn,
                        'cm': cm
                    }
                    
                    print(f"[OK] {model_name}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
                    
                except Exception as e:
                    print(f"[ERROR] {model_name} confusion matrix failed: {e}")
        
        return confusion_matrices
    
    def display_confusion_matrices(self, confusion_matrices):
        """Display confusion matrices in text format"""
        print("\n" + "="*80)
        print("CONFUSION MATRICES FOR ALL MODELS")
        print("="*80)
        
        for model_name, metrics in confusion_matrices.items():
            print(f"\nModel: {model_name}")
            print("\nConfusion Matrix:")
            print("Predicted")
            print("        Yes    No")
            print(f"Actual Yes     {metrics['TP']:<6} {metrics['FN']:<6}")
            print(f"       No      {metrics['FP']:<6} {metrics['TN']:<6}")
            print("-" * 30)
    
    def create_visualizations(self, confusion_matrices):
        """Create visualizations of confusion matrices"""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
        
        model_names = list(confusion_matrices.keys())
        
        for idx, (model_name, metrics) in enumerate(confusion_matrices.items()):
            if idx < 4:  # We have 4 models max
                ax = axes[idx // 2, idx % 2]
                
                # Create heatmap
                sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Appendicitis', 'Appendicitis'],
                           yticklabels=['No Appendicitis', 'Appendicitis'],
                           ax=ax)
                
                ax.set_title(f'{model_name}', fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Hide any unused subplots
        for idx in range(len(confusion_matrices), 4):
            axes[idx // 2, idx % 2].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = base_path / "confusion_matrices_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualization saved to: {plot_path}")
        
        plt.show()
    
    def run_analysis(self, threshold=0.5, create_plots=True):
        """Run complete confusion matrix analysis"""
        print("CONFUSION MATRIX ANALYSIS STARTED")
        print("="*50)
        
        # Step 1: Load models
        self.load_models()
        
        # Step 2: Load dataset
        self.load_dataset()
        
        # Step 3: Generate predictions
        predictions = self.generate_predictions(threshold)
        
        # Step 4: Compute confusion matrices
        confusion_matrices = self.compute_confusion_matrices(predictions)
        
        # Step 5: Display results
        self.display_confusion_matrices(confusion_matrices)
        
        # Step 6: Create visualizations (optional)
        if create_plots and confusion_matrices:
            self.create_visualizations(confusion_matrices)
        
        print("\n" + "="*50)
        print("CONFUSION MATRIX ANALYSIS COMPLETED")
        print("="*50)
        
        return confusion_matrices

def main():
    """Main function to run confusion matrix analysis"""
    generator = ConfusionMatrixGenerator()
    
    # Run analysis with threshold = 0.7 (same as in backend_predictor.py)
    confusion_matrices = generator.run_analysis(threshold=0.7, create_plots=True)
    
    return confusion_matrices

if __name__ == "__main__":
    main()
