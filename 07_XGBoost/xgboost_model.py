import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class XGBoostModel:
    """XGBoost model for appendicitis prediction"""
    
    def __init__(self):
        print("USING UPDATED XGBOOST MODEL")
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.best_params = None
        
    def load_unified_data(self):
        """Load unified data for fair comparison"""
        
        # Import unified data preprocessing
        import sys
        import os
        # Add project root to path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from unified_data_preprocessing import prepare_unified_data
        
        # Load unified data
        X_train, X_test, y_train, y_test, feature_names, encoders, scalers = prepare_unified_data('CSV')
        
        # Store encoders and scalers
        self.categorical_encoders = encoders
        self.numerical_scalers = scalers
        self.feature_names = feature_names
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, X, y):
        """Prepare data for XGBoost model"""
        
        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y.values.flatten()
        
        # Ensure both arrays have the same length
        min_length = min(len(X_array), len(y_array))
        X_array = X_array[:min_length]
        y_array = y_array[:min_length]
        
        # Find indices where target is not NaN
        valid_indices = ~pd.isna(y_array)
        
        # Filter both features and targets
        X_clean = X_array[valid_indices]
        y_clean = y_array[valid_indices]
        
        # Encode target variable
        label_encoder = LabelEncoder()
        
        # Convert to binary: appendicitis vs no appendicitis
        y_binary = np.array([1 if str(x).lower() == 'appendicitis' else 0 for x in y_clean])
        
        y_encoded = label_encoder.fit_transform(y_binary)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        print(f"Data after cleaning: {X_clean.shape}")
        print(f"Target distribution: {np.bincount(y_encoded)}")
        print(f"Target classes: {label_encoder.classes_}")
        
        return X_clean, y_encoded, label_encoder
    
    def split_data(self, X, y, test_size=0.4, random_state=42):
        """Split data into training and testing sets (60:40 ratio as per paper)"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        print(f"Training target distribution: {np.bincount(y_train)}")
        print(f"Testing target distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for XGBoost"""
        
        print("Performing hyperparameter tuning...")
        
        # Calculate scale_pos_weight for class imbalance
        from collections import Counter
        counter = Counter(y_train)
        scale_pos_weight = counter[0] / counter[1]  # majority / minority
        
        print(f"Class distribution: {dict(counter)}")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Define parameter grid - EXPANDED for better tuning effectiveness
        param_grid = {
            'n_estimators': [100, 300, 500, 700],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Create XGBoost classifier - ADDED: scale_pos_weight, SEED for reproducibility
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=SEED,
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight
        )
        
        # Perform Grid Search - CHANGED: Use f1 scoring
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        # TUNING EFFECTIVENESS DIAGNOSTICS
        print(f"\n{'='*60}")
        print("TUNING EFFECTIVENESS ANALYSIS")
        print(f"{'='*60}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Check if best params differ significantly from defaults
        default_n_est = 100
        default_lr = 0.3  # XGBoost default
        default_depth = 6  # XGBoost default
        best_n_est = self.best_params.get('n_estimators', default_n_est)
        best_lr = self.best_params.get('learning_rate', default_lr)
        best_depth = self.best_params.get('max_depth', default_depth)
        
        params_differ = (best_n_est != default_n_est or 
                        abs(best_lr - default_lr) > 0.05 or 
                        best_depth != default_depth)
        
        print(f"\nParameter Impact Analysis:")
        print(f"  - n_estimators: {best_n_est} (default: {default_n_est})")
        print(f"  - learning_rate: {best_lr} (default: {default_lr})")
        print(f"  - max_depth: {best_depth} (default: {default_depth})")
        print(f"  - Parameters differ from default: {params_differ}")
        
        # Show all CV results summary
        cv_results = grid_search.cv_results_
        mean_scores = cv_results['mean_test_score']
        print(f"  - Score range: {mean_scores.min():.4f} to {mean_scores.max():.4f}")
        print(f"  - Score std: {mean_scores.std():.4f}")
        
        if mean_scores.std() < 0.01:
            print(f"  WARNING: Low score variance - tuning may be ineffective")
            print(f"  WARNING: Suggests: Dataset bottleneck or model limitation")
        else:
            print(f"  OK: Score variance healthy - tuning is effective")
        print(f"{'='*60}\n")
        
        return self.model
    
    def train_model(self, X_train, y_train, use_hyperparameter_tuning=True):
        """Train XGBoost model"""
        
        # Calculate scale_pos_weight for class imbalance
        from collections import Counter
        counter = Counter(y_train)
        scale_pos_weight = counter[0] / counter[1]
        
        if use_hyperparameter_tuning:
            self.model = self.hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters - ADDED: scale_pos_weight, SEED for reproducibility
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=1,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=SEED,
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight
            )
            self.model.fit(X_train, y_train)
        
        print(f"Model trained successfully!")
        print(f"Number of estimators: {self.model.n_estimators}")
        print(f"Best score: {self.model.best_score if hasattr(self.model, 'best_score') else 'N/A'}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, threshold=0.5):
        """Evaluate XGBoost model with optional threshold adjustment"""
        
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Apply custom threshold for better sensitivity (default 0.5, can be lowered to 0.3-0.4)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        sensitivity = recall_score(y_test, y_pred, average='binary')  # Sensitivity = Recall
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate PPV and NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        return metrics, y_pred, y_pred_proba
    
    def calculate_medical_metrics(self, y_true, y_pred):
        """Calculate medical-specific evaluation metrics"""
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        sensitivity = recall_score(y_true, y_pred, average='binary')  # Sensitivity = Recall
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate PPV and NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history for XGBoost"""
        
        if self.model is None or not hasattr(self.model, 'evals_result'):
            print("No evaluation history available!")
            return
        
        # Get evaluation results
        results = self.model.evals_result()
        
        if not results:
            print("No evaluation results found!")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot training and validation loss
        for metric in results:
            for dataset in results[metric]:
                plt.plot(results[metric][dataset], label=f'{dataset} {metric}')
        
        plt.xlabel('Iterations')
        plt.ylabel('Metric Value')
        plt.title('XGBoost Training History')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from XGBoost"""
        
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        
        feature_importance = self.get_feature_importance(top_n)
        
        if feature_importance is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot horizontal bar chart
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - XGBoost')
        plt.gca().invert_yaxis()  # To display highest importance at top
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_tree(self, tree_index=0, save_path=None):
        """Plot individual XGBoost tree"""
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        plt.figure(figsize=(20, 12))
        
        # Plot tree
        xgb.plot_tree(
            self.model, 
            tree_index=tree_index,
            feature_names=self.feature_names[:50] if len(self.feature_names) > 50 else self.feature_names,
            ax=plt.gca()
        )
        
        plt.title(f'XGBoost Tree {tree_index}', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, filepath):
        """Save the trained model"""
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data.get('best_params')
        
        print(f"Model loaded from: {filepath}")
        return self.model

def main():
    """Main function for XGBoost model training and evaluation"""
    
    print("=" * 60)
    print("XGBOOST MODEL FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("[FAIR COMPARISON - UNIFIED DATA]")
    print("=" * 60)
    
    try:
        # Initialize XGBoost model
        xgb_model = XGBoostModel()
        
        # Load unified data (same as all other models)
        print(f"\n{'='*60}")
        print("LOADING UNIFIED DATA")
        print(f"{'='*60}")
        X_train, X_test, y_train, y_test = xgb_model.load_unified_data()
        
        print(f"\n{'='*60}")
        print(f"TRAINING XGBOOST MODEL")
        print(f"{'='*60}")
        
        # Train model
        xgb_model.train_model(X_train, y_train, use_hyperparameter_tuning=True)
        
        # DEBUG: Verify tuned parameters are actually used
        print(f"\n{'='*60}")
        print("MODEL PARAMETER VERIFICATION")
        print(f"{'='*60}")
        if xgb_model.best_params:
            print(f"Best parameters from tuning: {xgb_model.best_params}")
            print(f"Model is using tuned parameters: {xgb_model.model.get_params()['n_estimators'] != 100}")
        else:
            print("WARNING: No tuned parameters found - using defaults!")
        print(f"Final Model Parameters: n_estimators={xgb_model.model.get_params()['n_estimators']}, "
              f"learning_rate={xgb_model.model.get_params()['learning_rate']:.4f}, "
              f"max_depth={xgb_model.model.get_params()['max_depth']}, "
              f"subsample={xgb_model.model.get_params()['subsample']}")
        print(f"{'='*60}")
        
        # Evaluate model
        print(f"\nEvaluating model...")
        metrics, y_pred, y_pred_proba = xgb_model.evaluate_model(X_test, y_test)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"PPV (Positive Predictive Value): {metrics['ppv']:.4f}")
        print(f"NPV (Negative Predictive Value): {metrics['npv']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {metrics['tp']}")
        print(f"  True Negatives: {metrics['tn']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
        
        # Feature importance
        feature_importance = xgb_model.get_feature_importance(top_n=10)
        print(f"\nTop 10 Feature Importance:")
        print(feature_importance.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"xgboost_results_fair_{timestamp}.pkl"
        model_file = f"xgboost_model_fair_{timestamp}.pkl"
        
        # Save model and results
        xgb_model.save_model(model_file)
        
        results = {
            'dataset_name': 'CSV',
            'model_type': 'XGBoost',
            'data_type': 'unified_30_features',
            'best_params': xgb_model.best_params,
            'final_metrics': metrics,
            'feature_importance': feature_importance,
            'timestamp': timestamp
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Model saved to: {model_file}")
        
        print(f"\n{'='*60}")
        print("XGBOOST MODEL TRAINING COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
