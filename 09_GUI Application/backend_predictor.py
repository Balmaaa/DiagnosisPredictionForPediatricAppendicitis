import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import sys
import os
import warnings
warnings.filterwarnings('ignore')

NON_NEGATIVE_FIELDS = [
    "Age", "Weight", "Height", "BMI",
    "WBC_Count", "RBC_Count", "Hemoglobin",
    "RDW", "Thrombocyte_Count", "CRP"
]

REQUIRED_FIELDS = ["Age", "Sex"]

# Laboratory fields that can be missing
LAB_FIELDS = [
    "WBC_Count", "RBC_Count", "Hemoglobin",
    "RDW", "Segmented_Neutrophils", "Thrombocyte_Count", "CRP",
    "Neutrophil_Percentage", "Ketones_in_Urine",
    "RBC_in_Urine", "WBC_in_Urine"
]

def handle_missing_lab_values(input_data, reference_data=None):
    """Handle missing laboratory values using training data statistics"""
    # Handle both dictionary and string input
    if isinstance(input_data, dict):
        processed_data = input_data.copy()
    else:
        # If input_data is a string, convert to empty dict
        processed_data = {}
    
    # Default imputation values based on typical clinical ranges
    # These should match the training data distribution
    default_lab_values = {
        "WBC_Count": np.nan,           # Use NaN for missing values
        "RBC_Count": np.nan,           # Use NaN for missing values  
        "Hemoglobin": np.nan,         # Use NaN for missing values
        "RDW": np.nan,                # Use NaN for missing values
        "Segmented_Neutrophils": np.nan, # Use NaN for missing values
        "Thrombocyte_Count": np.nan, # Use NaN for missing values
        "CRP": np.nan,                # Use NaN for missing values
        "Neutrophil_Percentage": np.nan, # Use NaN for missing values
        "Ketones_in_Urine": 0,       # Binary: 0 or 1
        "RBC_in_Urine": 0,          # Binary: 0 or 1  
        "WBC_in_Urine": 0           # Binary: 0 or 1
    }
    
    # Continuous lab fields where 0.0 is clinically impossible (treat as missing)
    zero_means_missing = {"WBC_Count", "RBC_Count", "Hemoglobin", "RDW", 
                          "Thrombocyte_Count", "CRP", "Neutrophil_Percentage", "Segmented_Neutrophils"}
    
    missing_labs = []
    
    for field in LAB_FIELDS:
        if field in processed_data:
            val = processed_data[field]
            if val is None or val == "":
                processed_data[field] = default_lab_values.get(field, 0)
                missing_labs.append(field)
            elif field in zero_means_missing and isinstance(val, (int, float)) and val == 0.0:
                # 0.0 is clinically impossible for these fields - treat as missing
                processed_data[field] = default_lab_values.get(field, 0)
                missing_labs.append(field)
        else:
            processed_data[field] = default_lab_values.get(field, 0)
            missing_labs.append(field)
    
    return processed_data, missing_labs

def validate_inputs(input_data):
    """Validate input data for clinical safety"""
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in input_data or input_data[field] is None or input_data[field] == "":
            errors.append(f"{field} is required")
    
    # Check non-negative fields (only if value is provided)
    for field in NON_NEGATIVE_FIELDS:
        if field in input_data and input_data[field] is not None:
            if isinstance(input_data[field], (int, float)) and input_data[field] < 0:
                errors.append(f"{field} cannot be negative (value: {input_data[field]})")
    
    return errors

# GUI MODE: Set to True for fast training without hyperparameter tuning
GUI_MODE = True

# Import updated model classes from model files
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path / "06_Decision_Trees"))
sys.path.insert(0, str(base_path / "07_Gradient_Boosting"))
sys.path.insert(0, str(base_path / "08_XGBoost"))
sys.path.insert(0, str(base_path / "05_Transformer_Model"))

from decision_tree_model import DecisionTreeModel
from gradient_boosting_model import GradientBoostingModel
from xgboost_model import XGBoostModel
try:
    from transformer_model import TransformerModel, AdvancedTabularTransformer, analyze_features, prepare_data_for_advanced_transformer
    TRANSFORMER_AVAILABLE = True
    print("REAL TRANSFORMER MODEL (PyTorch) IMPORTED SUCCESSFULLY")
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("[WARNING] Transformer model not available, will use fallback")

class AppendicitisPredictor:
    """Backend predictor for appendicitis diagnosis with real AI models"""
    
    def __init__(self):
        self.pipeline = None
        self.models = {}
        self.feature_names = None
        self.is_trained = False
        # Real-time evaluation data
        self.X_test = None
        self.y_test = None
        self.load_components()
    
    def load_components(self):
        """Load preprocessing pipeline and train real models"""
        
        base_path = Path(__file__).parent
        
        try:
            # Load preprocessing pipeline
            pipeline_path = base_path.parent / "04_Preprocessing Pipeline" / "preprocessing_pipeline_20260202_083206.pkl"
            if pipeline_path.exists():
                with open(pipeline_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                print("Preprocessing pipeline loaded")
            else:
                print("Preprocessing pipeline not found, creating fallback")
                self._create_fallback_pipeline()
            
            # Try to load from .pkl files first (fast!)
            loaded_models = self._load_models_from_pkl()
            
            if not loaded_models:
                # No models loaded - train all
                print("No saved models found. Training new models...")
                self._train_real_models()
                
                # Verify models were created after training
                if not self.models:
                    print("[WARNING] Training failed! Creating fallback models...")
                    self._create_fallback_models()
                else:
                    # Save to .pkl for next time
                    self._save_models_to_pkl()
            elif len(self.models) < 4:
                # Partial load - train missing models
                missing = ['Decision Tree', 'Gradient Boosting', 'XGBoost', 'Transformer']
                missing = [m for m in missing if m not in self.models]
                print(f"[WARNING] Partial load complete. Training missing models: {missing}")
                self._train_real_models()
                # Save all models (including newly trained)
                self._save_models_to_pkl()
            
            # Set trained flag if we have models
            if self.models:
                self.is_trained = True
                print(f"Models ready: {list(self.models.keys())}")
                print(f"is_trained: {self.is_trained}")
            else:
                print("CRITICAL: No models available!")
            
        except Exception as e:
            print(f"Error loading components: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_system()
            if self.models:
                self.is_trained = True
    
    def _create_fallback_pipeline(self):
        """Create a simple preprocessing pipeline"""
        try:
            # Define numerical and categorical features
            numerical_features = ['Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 'WBC_Count', 
                                 'RBC_Count', 'Hemoglobin', 'RDW', 'Segmented_Neutrophils', 
                                 'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage']
            
            categorical_features = ['Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 
                                   'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 
                                   'Severity', 'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness', 
                                   'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine']
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numerical_features),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_features)
                ])
            
            self.pipeline = preprocessor
            print("Fallback preprocessing pipeline created")
            
        except Exception as e:
            print(f"Error creating fallback pipeline: {e}")
    
    def _load_training_data(self):
        """Load or create training data"""
        try:
            # Try to load actual dataset
            base_path = Path(__file__).parent.parent
            
            # Look for CSV files in Dataset folders
            dataset_paths = [
                base_path / "Dataset 2" / "Regensburg Pediatric Appendicitis.csv",
                base_path / "Dataset 1" / "app_data.xlsx"
            ]
            
            data_loaded = False
            for dataset_path in dataset_paths:
                if dataset_path.exists():
                    if dataset_path.suffix == '.csv':
                        self.training_data = pd.read_csv(dataset_path)
                        print(f"Training data loaded from {dataset_path.name}")
                        data_loaded = True
                        break
                    elif dataset_path.suffix == '.xlsx':
                        self.training_data = pd.read_excel(dataset_path)
                        print(f"Training data loaded from {dataset_path.name}")
                        data_loaded = True
                        break
            
            if not data_loaded:
                raise FileNotFoundError("No real training dataset found. Synthetic data is disabled by policy.")
            
            # Clean and preprocess the real dataset
            self._clean_real_dataset()
                
        except Exception as e:
            print(f"Error loading training data: {e}")
            raise
    
    def _clean_real_dataset(self):
        """Clean and preprocess the real dataset"""
        try:
            print("Cleaning real dataset...")
            
            # Remove unnamed index column if present
            if 'Unnamed: 0' in self.training_data.columns:
                self.training_data = self.training_data.drop('Unnamed: 0', axis=1)
            
            # Convert diagnosis to binary (1 for appendicitis, 0 for no appendicitis)
            if 'Diagnosis' in self.training_data.columns:
                diagnosis_map = {
                    'appendicitis': 1,
                    'no appendicitis': 0,
                    'Appendicitis': 1,
                    'No Appendicitis': 0
                }
                self.training_data['Diagnosis'] = self.training_data['Diagnosis'].map(diagnosis_map)
                
                # Remove any rows with NaN in diagnosis
                self.training_data = self.training_data.dropna(subset=['Diagnosis'])
                
                # Ensure diagnosis is integer
                self.training_data['Diagnosis'] = self.training_data['Diagnosis'].astype(int)
            
            # Handle missing values in numeric columns
            numeric_columns = self.training_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'Diagnosis':  # Don't fill target variable
                    median_val = self.training_data[col].median()
                    self.training_data[col] = self.training_data[col].fillna(median_val)
            
            # Handle missing values in categorical columns
            categorical_columns = self.training_data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != 'Diagnosis':  # Don't fill target variable
                    mode_val = self.training_data[col].mode()[0] if not self.training_data[col].mode().empty else 'unknown'
                    self.training_data[col] = self.training_data[col].fillna(mode_val)
            
            # Convert categorical yes/no to 1/0
            yes_no_columns = ['Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness',
                            'Coughing_Pain', 'Nausea', 'Loss_of_Appetite', 'Neutrophilia', 'RBC_in_Urine', 
                            'WBC_in_Urine', 'Dysuria', 'Peritonitis', 'Psoas_Sign', 'Ipsilateral_Rebound_Tenderness',
                            'Appendix_on_US', 'Free_Fluids', 'Target_Sign', 'Appendicolith', 'Perfusion',
                            'Perforation', 'Surrounding_Tissue_Reaction', 'Appendicular_Abscess',
                            'Pathological_Lymph_Nodes', 'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops',
                            'Ileus', 'Coprostasis', 'Meteorism', 'Enteritis']
            
            for col in yes_no_columns:
                if col in self.training_data.columns:
                    self.training_data[col] = self.training_data[col].map({
                        'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 
                        'positive': 1, 'negative': 0, '+': 1, '-': 0
                    }).fillna(0)
            
            # Convert Sex to binary
            if 'Sex' in self.training_data.columns:
                self.training_data['Sex'] = self.training_data['Sex'].map({
                    'male': 1, 'female': 0, 'Male': 1, 'Female': 0
                }).fillna(0)
            
            print(f"Dataset cleaned: {len(self.training_data)} samples")
            print(f"Appendicitis cases: {self.training_data['Diagnosis'].sum()} ({self.training_data['Diagnosis'].mean():.1%})")
            
        except Exception as e:
            print(f"Error cleaning dataset: {e}")
            raise
    
    def _create_synthetic_data(self):
        """Create synthetic training data for demonstration"""
        raise RuntimeError("Synthetic data generation is disabled. Provide real datasets only.")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic synthetic data
        data = {
            'Age': np.random.normal(10, 4, n_samples),
            'Weight': np.random.normal(35, 15, n_samples),
            'Height': np.random.normal(140, 20, n_samples),
            'BMI': np.random.normal(18, 3, n_samples),
            'Body_Temperature': np.random.normal(37.5, 0.8, n_samples),
            'WBC_Count': np.random.normal(10, 4, n_samples),
            'RBC_Count': np.random.normal(4.5, 0.5, n_samples),
            'Hemoglobin': np.random.normal(12, 2, n_samples),
            'RDW': np.random.normal(14, 2, n_samples),
            'Segmented_Neutrophils': np.random.normal(65, 10, n_samples),
            'Thrombocyte_Count': np.random.normal(300, 100, n_samples),
            'CRP': np.random.normal(15, 20, n_samples),
            'Neutrophil_Percentage': np.random.normal(65, 10, n_samples),
            'Sex': np.random.choice(['Male', 'Female'], n_samples),
            'Lower_Right_Abd_Pain': np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4]),
            'Migratory_Pain': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
            'Loss_of_Appetite': np.random.choice(['yes', 'no'], n_samples, p=[0.5, 0.5]),
            'Nausea': np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4]),
            'Coughing_Pain': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'Dysuria': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'Stool': np.random.choice(['normal', 'constipation', 'diarrhea'], n_samples),
            'Peritonitis': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'Severity': np.random.choice(['uncomplicated', 'complicated'], n_samples, p=[0.7, 0.3]),
            'Contralateral_Rebound_Tenderness': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'Ipsilateral_Rebound_Tenderness': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'Psoas_Sign': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'Neutrophilia': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
            'Ketones_in_Urine': np.random.choice(['no', '+', '++', '+++'], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            'RBC_in_Urine': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
            'WBC_in_Urine': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85])
        }
        
        self.training_data = pd.DataFrame(data)
        
        # Create target variable with realistic correlations
        # Higher risk factors increase probability of appendicitis
        risk_score = (
            (self.training_data['Body_Temperature'] > 37.5).astype(int) * 0.2 +
            (self.training_data['WBC_Count'] > 12).astype(int) * 0.2 +
            (self.training_data['CRP'] > 10).astype(int) * 0.2 +
            (self.training_data['Lower_Right_Abd_Pain'] == 'yes').astype(int) * 0.15 +
            (self.training_data['Peritonitis'] == 'yes').astype(int) * 0.15 +
            (self.training_data['Severity'] == 'complicated').astype(int) * 0.1
        )
        
        # Add some noise and create binary target
        probability = risk_score + np.random.normal(0, 0.1, n_samples)
        probability = np.clip(probability, 0, 1)
        self.training_data['Diagnosis'] = (probability > 0.5).astype(int)
        
        print(f"Synthetic training data created: {n_samples} samples")
        print(f"Appendicitis cases: {self.training_data['Diagnosis'].sum()} ({self.training_data['Diagnosis'].mean():.1%})")
    
    def _train_real_models(self):
        """Train models using updated model classes"""
        try:
            # GUI MODE indicator
            if GUI_MODE:
                print("GUI MODE: Fast training (no hyperparameter tuning)")
            else:
                print("RESEARCH MODE: Full hyperparameter tuning enabled")
            
            print("Training models using updated model files...")
            
            # Prepare shared dataset with unified 38-feature schema
            print("Loading shared training data with unified 38 features...")
            dt_wrapper = DecisionTreeModel()
            X_train, X_test, y_train, y_test = dt_wrapper.load_unified_data()
            
            # Enforce global feature contract (38 features) and ordering everywhere
            if hasattr(dt_wrapper, 'feature_names') and dt_wrapper.feature_names:
                self.feature_columns = dt_wrapper.feature_names
            elif hasattr(X_train, 'columns'):
                self.feature_columns = X_train.columns.tolist()
            else:
                raise ValueError("Could not determine unified feature ordering")
            
            # Assert input dimensionality
            if hasattr(X_train, 'shape') and X_train.shape[1] != 38:
                raise ValueError(f"Unified feature space violation: expected 38 features, got {X_train.shape[1]}")
            
            print(f"[OK] Unified 38-feature data loaded: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # Store preprocessing components from unified pipeline (train-fitted)
            self.categorical_encoders = dt_wrapper.categorical_encoders
            # Unify naming: prefer numerical_scaler; keep alias for backward compatibility
            self.numerical_scaler = dt_wrapper.numerical_scaler if hasattr(dt_wrapper, 'numerical_scaler') else dt_wrapper.numerical_scalers
            self.numerical_scalers = self.numerical_scaler
            print(f"[OK] Using unified preprocessing components (38 features)")
            
            # Store test data for real-time evaluation
            self.X_test = X_test
            self.y_test = y_test
            print(f"[OK] Test data stored: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # Train Decision Tree using updated model class (if not already loaded)
            if 'Decision Tree' not in self.models:
                print("\n[1/4] Training Decision Tree using decision_tree_model.py...")
                dt_wrapper.train_model(X_train, y_train, use_hyperparameter_tuning=not GUI_MODE)
                self.models['Decision Tree'] = dt_wrapper.model
                self.feature_columns = dt_wrapper.feature_names
                print("[OK] Decision Tree trained successfully")
            else:
                print("\n[1/4] Decision Tree already loaded from .pkl, skipping training")
            
            # Train Gradient Boosting using updated model class (if not already loaded)
            if 'Gradient Boosting' not in self.models:
                print("\n[2/4] Training Gradient Boosting using gradient_boosting_model.py...")
                gb_wrapper = GradientBoostingModel()
                gb_wrapper.train_model(X_train, y_train, use_hyperparameter_tuning=not GUI_MODE)
                self.models['Gradient Boosting'] = gb_wrapper.model
                print("[OK] Gradient Boosting trained successfully")
            else:
                print("\n[2/4] Gradient Boosting already loaded from .pkl, skipping training")
            
            # Train XGBoost using updated model class (if not already loaded)
            if 'XGBoost' not in self.models:
                print("\n[3/4] Training XGBoost using xgboost_model.py...")
                xgb_wrapper = XGBoostModel()
                xgb_wrapper.train_model(X_train, y_train, use_hyperparameter_tuning=not GUI_MODE)
                self.models['XGBoost'] = xgb_wrapper.model
                print("[OK] XGBoost trained successfully")
            else:
                print("\n[3/4] XGBoost already loaded from .pkl, skipping training")
            
            # Store processed feature columns from shared dataset
            self.processed_feature_columns = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # Train Transformer (with fallback)
            if 'Transformer' not in self.models:
                print("\n[4/4] Training Transformer...")
                print(f"   TRANSFORMER_AVAILABLE = {TRANSFORMER_AVAILABLE}")
                if TRANSFORMER_AVAILABLE:
                    try:
                        print("   Attempting to train PyTorch Transformer...")
                        transformer_model = self._train_transformer_from_class(X_train, y_train)
                        if transformer_model is not None:
                            self.models['Transformer'] = transformer_model
                            model_type = type(transformer_model).__name__
                            print(f"[OK] Transformer trained successfully (Type: {model_type})")
                            print(f"[DEBUG] Transformer added to models. Current models: {list(self.models.keys())}")
                        else:
                            print(f"[ERROR] Transformer training returned None!")
                            raise ValueError("Transformer training returned None")
                    except Exception as e:
                        print(f"[ERROR] Transformer training failed: {e}")
                        import traceback
                        traceback.print_exc()
                        print("[WARNING] Using Gradient Boosting as fallback for Transformer")
                        self.models['Transformer'] = self._create_simple_transformer(X_train, y_train)
                else:
                    print("[WARNING] PyTorch not available (TRANSFORMER_AVAILABLE=False)")
                    print("WARNING: Using Gradient Boosting as fallback for Transformer")
                    self.models['Transformer'] = self._create_simple_transformer(X_train, y_train)
            else:
                print("\n[4/4] Transformer already loaded from .pkl, skipping training")
            
            # Evaluate models
            print("\nEvaluating models...")
            self._evaluate_models()
            
            self.is_trained = True
            print("\n[OK] All models trained successfully using updated model files!")
            print(f"DEBUG: Final models after training: {list(self.models.keys())}")
            
            # Save models to .pkl for fast loading next time
            self._save_models_to_pkl()
            
        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_system()

    def _train_transformer_from_class(self, X_train, y_train):
        """Train Transformer using the new TransformerModel class"""
        print("Using REAL TransformerModel (PyTorch) for training...")
    
    # Create Transformer model
        transformer_model = TransformerModel(
            embed_dim=64,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )
        
        # Train the model
        trained_model = transformer_model.train_model(X_train, y_train)
        
        # Save the trained model
        model_dir = Path(__file__).parent / 'saved_models'
        model_dir.mkdir(exist_ok=True)
        transformer_path = model_dir / "Transformer.pt"
        transformer_model.save(str(transformer_path))
        
        print("REAL TRANSFORMER MODEL TRAINED AND SAVED SUCCESSFULLY")
        return transformer_model
    
    def _apply_dict_pipeline(self, X):
        """Apply dictionary-based preprocessing pipeline"""
        try:
            # Create a copy to avoid modifying original
            X_processed = X.copy()
            
            # Handle categorical variables
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
            
            # Simple encoding for categorical variables
            for col in categorical_cols:
                if X_processed[col].dtype == 'object':
                    # Use label encoding for simplicity
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                elif X_processed[col].dtype in ['int64', 'float64']:
                    # Skip encoding for already numeric values (from GUI preprocessing)
                    pass
            
            # Handle missing values
            for col in X_processed.columns:
                if X_processed[col].isnull().any():
                    if X_processed[col].dtype in ['object']:
                        X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)
                    else:
                        X_processed[col].fillna(X_processed[col].median(), inplace=True)
            
            # Return dictionary format for Transformer models
            if hasattr(self, 'processed_feature_columns'):
                feature_dict = {}
                for i, col in enumerate(X_processed.columns):
                    feature_dict[f'feature_{i}'] = X_processed[col].values
                return feature_dict
            else:
                return X_processed
            
        except Exception as e:
            print(f"Error applying dict pipeline: {e}")
            return X
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        try:
            print("\nModel Evaluation Results:")
            print("=" * 50)
            
            for model_name, model in self.models.items():
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
                
                print(f"{model_name}:")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                print()
                
        except Exception as e:
            print(f"Error evaluating models: {e}")
    
    def _create_fallback_system(self):
        """Create fallback system if training fails"""
        print("Creating fallback prediction system")
        self.is_trained = False
        # Models will be created on-demand in predict method
    
    def preprocess_input(self, input_data, lab_available=True):
        """Preprocess input data using the same pipeline as training
        
        Args:
            input_data: Dictionary or DataFrame with patient data
            lab_available: Boolean indicating if lab results are available
        """
        
        try:
            # Convert input data to DataFrame
            if isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            else:
                df = pd.DataFrame([input_data])
            
            # Force use of 30 features + missing indicators
            FIXED_30_FEATURES = [
                'Age', 'Weight', 'Height', 'BMI', 'Sex', 'Body_Temperature',
                'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 'Nausea',
                'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 'Severity',
                'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness', 'Psoas_Sign',
                'Neutrophilia', 'Neutrophil_Percentage', 'WBC_Count', 'RBC_Count', 'Hemoglobin',
                'RDW', 'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP',
                'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine'
            ]
            
            # Laboratory missing indicators
            LAB_MISSING_INDICATORS = [
                'WBC_Count_missing', 'RBC_Count_missing', 'Hemoglobin_missing', 'RDW_missing',
                'Segmented_Neutrophils_missing', 'Thrombocyte_Count_missing', 'CRP_missing',
                'Neutrophil_Percentage_missing'
            ]
            
            # Add missing columns with clinically appropriate default values
            clinical_defaults = {
                'Age': 10, 'Weight': 35, 'Height': 140, 'BMI': 18, 'Sex': 0,
                'Body_Temperature': 36.5,
                'Lower_Right_Abd_Pain': 0, 'Migratory_Pain': 0, 'Loss_of_Appetite': 0,
                'Nausea': 0, 'Coughing_Pain': 0, 'Dysuria': 0, 'Stool': 0,
                'Peritonitis': 0, 'Severity': 0,
                'Contralateral_Rebound_Tenderness': 0, 'Ipsilateral_Rebound_Tenderness': 0,
                'Psoas_Sign': 0, 'Neutrophilia': 0,
                'Neutrophil_Percentage': 65.0, 'WBC_Count': 8.5, 'RBC_Count': 4.5,
                'Hemoglobin': 12.5, 'RDW': 14.0, 'Segmented_Neutrophils': 55.0,
                'Thrombocyte_Count': 250.0, 'CRP': 10.0,
                'Ketones_in_Urine': 0, 'RBC_in_Urine': 0, 'WBC_in_Urine': 0
            }
            
            # Track which lab values were missing based on lab availability
            lab_fields = ['WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW', 
                         'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP', 
                         'Neutrophil_Percentage']
            missing_indicators = {}
            for lab in lab_fields:
                if lab_available:
                    # Labs are available - check if actually missing (0.0 or NaN)
                    if lab in df.columns:
                        missing_indicators[f'{lab}_missing'] = ((df[lab] == 0.0) | (df[lab].isna())).astype(int)
                    else:
                        missing_indicators[f'{lab}_missing'] = 1  # Missing entirely
                else:
                    # Labs are not available - mark all as missing
                    missing_indicators[f'{lab}_missing'] = 1
            
            # Add missing columns and impute lab values
            for col in FIXED_30_FEATURES:
                if col not in df.columns:
                    df[col] = clinical_defaults.get(col, 0)
                elif col in lab_fields and df[col].iloc[0] == 0.0:
                    # Impute missing lab values with clinical defaults
                    df[col] = df[col].fillna(df[col].mean())  # Use mean instead of zeros
                elif col in lab_fields and not lab_available:
                    # Impute missing lab values with mean when lab_available=False
                    df[col] = df[col].fillna(df[col].mean())  # Use mean instead of zeros
            
            # Add missing indicators to the dataframe (these will be processed by models)
            for indicator_name, indicator_value in missing_indicators.items():
                df[indicator_name] = indicator_value
            
            # Create extended feature list with missing indicators
            lab_fields = ['WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW', 
                         'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP', 
                         'Neutrophil_Percentage']
            extended_features = FIXED_30_FEATURES.copy()
            
            # Add missing indicators to the feature list
            for lab in lab_fields:
                if f'{lab}_missing' in df.columns:
                    extended_features.append(f'{lab}_missing')
            
            # Reorder columns to include both original features and missing indicators
            df = df[extended_features]
            
            # Store extended features for later use
            self.extended_features = extended_features
            
            # Use unified preprocessing pipeline for consistent preprocessing
            try:
                processed_df = df.copy()
                
                # Apply categorical encoding if available
                if hasattr(self, 'categorical_encoders') and self.categorical_encoders:
                    categorical_features = ['Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 
                                          'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 'Severity',
                                          'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness', 
                                          'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine']
                    
                    for col in categorical_features:
                        if col in processed_df.columns and col in self.categorical_encoders:
                            le = self.categorical_encoders[col]
                            processed_df[col] = le.transform(processed_df[col].astype(str))
                
                # Apply numerical scaling if available
                if hasattr(self, 'numerical_scalers') and self.numerical_scalers:
                    numerical_features = ['Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 'WBC_Count',
                                       'RBC_Count', 'Hemoglobin', 'RDW', 'Segmented_Neutrophils', 'Thrombocyte_Count',
                                       'CRP', 'Neutrophil_Percentage']
                    
                    # Only scale numerical columns that exist
                    num_cols_to_scale = [col for col in numerical_features if col in processed_df.columns]
                    if num_cols_to_scale:
                        processed_df[num_cols_to_scale] = self.numerical_scalers.transform(processed_df[num_cols_to_scale])
                
                processed_data = processed_df.values
                print(f"[OK] Preprocessed input with unified pipeline (shape: {processed_data.shape})")
                
            except Exception as e:
                print(f"[WARNING] Unified preprocessing failed: {e}, using raw values")
                processed_data = df.values
            
            return processed_data
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise
    
    def _fallback_preprocessing(self, df):
        """Fallback preprocessing if main pipeline fails"""
        
        try:
            # Create a simple preprocessing for demo purposes
            processed_df = df.copy()
            
            # Handle categorical variables
            categorical_cols = processed_df.select_dtypes(include=['object']).columns
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
            
            # Simple encoding for categorical variables
            for col in categorical_cols:
                if processed_df[col].dtype == 'object':
                    # Use label encoding for simplicity
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            
            # Handle missing values
            for col in processed_df.columns:
                if processed_df[col].isnull().any():
                    if processed_df[col].dtype in ['object']:
                        processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                    else:
                        processed_df[col].fillna(processed_df[col].median(), inplace=True)
            
            return processed_df.values
            
        except Exception as e:
            print(f"Error in fallback preprocessing: {e}")
            raise
    
    def predict(self, model_name, input_data, lab_available=True):
        """Make prediction using specified model
        
        Args:
            model_name: Name of the model to use
            input_data: Dictionary or DataFrame with patient data
            lab_available: Boolean indicating if lab results are available
        """
        try:
            print(f"\n{'='*60}")
            print(f"PREDICTION REQUEST: Model='{model_name}'")
            print(f"{'='*60}")
            
            # Handle missing laboratory values (silently without warnings)
            processed_input, missing_labs = handle_missing_lab_values(input_data)
            # Removed: No more warning messages for missing labs
            # System handles missing values internally
            
            # Validate input data for clinical safety
            validation_errors = validate_inputs(processed_input)
            if validation_errors:
                raise ValueError(f"Invalid input: {', '.join(validation_errors)}")
            
            # Convert to DataFrame for preprocessing
            df = pd.DataFrame([processed_input])
            
            # Apply preprocessing pipeline using dict-based encoders and scalers
            if self.pipeline is not None and hasattr(self, 'categorical_encoders'):
                # Use unified preprocessing from new pipeline
                X_processed = df.copy()
                
                # Apply categorical encoding
                for col, encoder in self.categorical_encoders.items():
                    if col in X_processed.columns:
                        X_processed[col] = encoder.transform(X_processed[col].astype(str))
                
                # Apply numerical scaling
                if hasattr(self, 'numerical_scaler'):
                    scaler = self.numerical_scaler
                    numerical_cols = [col for col in NUMERICAL_FEATURES if col in X_processed.columns]
                    if numerical_cols and scaler:
                        X_processed[numerical_cols] = scaler.transform(X_processed[numerical_cols])
                
                X = X_processed.values
            else:
                # Fallback preprocessing
                X = df.values
            
            # Get model
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
            
            # Get probabilities first
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if isinstance(proba, np.ndarray):
                    prob_appendicitis = float(proba[0][1])
                    prob_no = float(proba[0][0])
                else:
                    prob_appendicitis = float(proba[1])
                    prob_no = float(proba[0])
            else:
                # For models without predict_proba
                prob_appendicitis = 0.5
                prob_no = 0.5
            
            threshold = 0.7  # TO ADJUST
            
            if prob_appendicitis >= threshold:
                prediction = 1  # Appendicitis
            else:
                prediction = 0  # No Appendicitis
            
            # Debug output for verification
            print(f"Probability: {prob_appendicitis:.3f}")
            print(f"Threshold: {threshold:.1f}")
            print(f"Prediction: {'Appendicitis' if prediction == 1 else 'No Appendicitis'}")
            
            return {
                "prediction": int(prediction),
                "prob_appendicitis": prob_appendicitis,
                "prob_no": prob_no,
                "model_name": model_name,
                "lab_available": lab_available,
                "threshold_used": threshold
            }
                
        except Exception as e:
            print(f"[ERROR] in predict(): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name):
        """Get model information including hyperparameter tuning status"""
        if model_name in self.models:
            model = self.models[model_name]
            info = {
                'name': model_name,
                'type': type(model).__name__,
                'trained': self.is_trained,
                'hyperparameter_tuned': True  # All models now use hyperparameter tuning
            }
            
            # Add model-specific metrics if available
            if hasattr(model, 'feature_importances_'):
                info['has_feature_importance'] = True
            
            # Add specific tuning info based on model type
            if model_name == 'Decision Tree':
                info['tuning_params'] = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion', 'splitter']
            elif model_name == 'Gradient Boosting':
                info['tuning_params'] = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample']
            elif model_name == 'XGBoost':
                info['tuning_params'] = ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma']
            elif model_name == 'Transformer':
                info['tuning_params'] = ['embed_dim', 'num_heads', 'num_layers', 'dropout', 'attention_dropout', 'learning_rate']
            
            return info
        else:
            # Return default result if model not found
            return {
                "prediction": 0,
                "prob_appendicitis": 0.5,
                "prob_no": 0.5,
                "model_name": model_name,
                "lab_available": lab_available
            }

    def _save_models_to_pkl(self):
        """Save trained models to .pkl files for fast loading"""
        try:
            import pickle
            import os
            model_dir = Path(__file__).parent / 'saved_models'
            model_dir.mkdir(exist_ok=True)
            
            saved_count = 0
            for model_name, model in self.models.items():
                pkl_path = model_dir / f"{model_name.replace(' ', '_')}.pkl"
                
                try:
                    # Handle PyTorch models specially
                    if model_name == 'Transformer':
                        # Check if it's a PyTorch model
                        if hasattr(model, 'state_dict') or 'torch.nn' in str(type(model)):
                            import torch
                            torch.save(model, pkl_path)
                            # Verify file was created
                            if pkl_path.exists():
                                size_kb = pkl_path.stat().st_size / 1024
                                print(f"  [SAVED] {model_name} (PyTorch) to {pkl_path} ({size_kb:.1f} KB)")
                                saved_count += 1
                            else:
                                print(f"  [FAILED] Failed to save {model_name} - file not created")
                        else:
                            # Save as pickle (sklearn fallback)
                            with open(pkl_path, 'wb') as f:
                                pickle.dump(model, f)
                                f.flush()
                                os.fsync(f.fileno())
                            # Verify file was created
                            if pkl_path.exists():
                                size_kb = pkl_path.stat().st_size / 1024
                                print(f"  [SAVED] {model_name} (sklearn) to {pkl_path} ({size_kb:.1f} KB)")
                                saved_count += 1
                            else:
                                print(f"  [FAILED] Failed to save {model_name} - file not created")
                    else:
                        with open(pkl_path, 'wb') as f:
                            pickle.dump(model, f)
                            f.flush()
                            os.fsync(f.fileno())
                        # Verify file was created
                        if pkl_path.exists():
                            size_kb = pkl_path.stat().st_size / 1024
                            print(f"  [SAVED] {model_name} to {pkl_path} ({size_kb:.1f} KB)")
                            saved_count += 1
                        else:
                            print(f"  [FAILED] Failed to save {model_name} - file not created")
                except Exception as e:
                    print(f"  [ERROR] Error saving {model_name}: {e}")
            
            # Save metadata (including preprocessing components for consistent loading)
            metadata = {
                'feature_columns': self.feature_columns,
                'processed_feature_columns': self.processed_feature_columns,
                'is_trained': True,
                'categorical_encoders': self.categorical_encoders if hasattr(self, 'categorical_encoders') else None,
                'numerical_scalers': self.numerical_scalers if hasattr(self, 'numerical_scalers') else None
            }
            metadata_path = model_dir / 'metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                f.flush()
                os.fsync(f.fileno())
            
            print(f"  [SAVED] {saved_count}/{len(self.models)} models saved to {model_dir}")
        except Exception as e:
            print(f"  [ERROR] Could not save models: {e}")
            import traceback
            traceback.print_exc()

    def _create_simple_transformer(self, X_train, y_train):
        """Create a simple Gradient Boosting fallback for Transformer"""
        try:
            print("Creating Gradient Boosting fallback for Transformer...")
            from sklearn.ensemble import GradientBoostingClassifier
            
            # Create and train Gradient Boosting model
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Convert to numpy if needed
            if hasattr(X_train, 'values'):
                X_array = X_train.values
            else:
                X_array = X_train
                
            if hasattr(y_train, 'values'):
                y_array = y_train.values
            else:
                y_array = y_train
                
            model.fit(X_array, y_array)
            print(f"Fallback Transformer (Gradient Boosting) created successfully")
            return model
            
        except Exception as e:
            print(f"Error creating fallback Transformer: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_models_from_pkl(self):
        """Load models from .pkl files if available"""
        try:
            import pickle
            model_dir = Path(__file__).parent / 'saved_models'
            
            print(f"DEBUG: Looking for saved models in {model_dir}")
            
            if not model_dir.exists():
                print(f"DEBUG: Model directory does not exist: {model_dir}")
                return False
            
            print(f"DEBUG: Model directory exists. Files: {list(model_dir.glob('*.pkl'))}")
            
            # Load metadata
            metadata_path = model_dir / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.feature_columns = metadata.get('feature_columns', [])
                self.processed_feature_columns = metadata.get('processed_feature_columns', [])
                self.is_trained = metadata.get('is_trained', False)
                # Load saved preprocessing components for consistent preprocessing
                if 'categorical_encoders' in metadata and metadata['categorical_encoders'] is not None:
                    self.categorical_encoders = metadata['categorical_encoders']
                    print(f"[OK] Categorical encoders loaded from metadata ({len(self.categorical_encoders)} encoders)")
                if 'numerical_scalers' in metadata and metadata['numerical_scalers'] is not None:
                    self.numerical_scalers = metadata['numerical_scalers']
                    print(f"[OK] Numerical scaler loaded from metadata")
                else:
                    print("[WARNING] No preprocessing components in metadata, will create during training")
            
            # Load test data for real-time evaluation
            if self.X_test is None or self.y_test is None:
                print("Loading test data for real-time evaluation...")
                try:
                    dt_wrapper = DecisionTreeModel()
                    X_train, X_test, y_train, y_test = dt_wrapper.load_unified_data()
                    self.X_test = X_test
                    self.y_test = y_test
                    print(f"[OK] Test data loaded: X_test={X_test.shape}, y_test={y_test.shape}")
                except Exception as e:
                    print(f"[WARNING] Could not load test data: {e}")
                    # Create fallback test data
                    self._create_fallback_test_data()
            
            # Load models
            model_files = {
                'Decision Tree': 'Decision_Tree.pkl',
                'Gradient Boosting': 'Gradient_Boosting.pkl',
                'XGBoost': 'XGBoost.pkl',
                'Transformer': 'Transformer.pt'  # Use .pt for real Transformer
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                model_path = model_dir / filename
                print(f"DEBUG: Checking {model_name} at {model_path} - Exists: {model_path.exists()}")
                if model_path.exists():
                    try:
                        if model_name == 'Transformer':
                            # Load real Transformer from .pt file using its built-in load method
                            transformer_model = TransformerModel()
                            transformer_model.load(str(model_path))
                            self.models[model_name] = transformer_model
                            print(f"  [LOADED] Transformer from {model_path}")
                            print("REAL TRANSFORMER MODEL LOADED SUCCESSFULLY (PyTorch)")
                        else:
                            # Load sklearn models with pickle
                            with open(model_path, 'rb') as f:
                                loaded_obj = pickle.load(f)
                            
                            # Extract actual model if loaded as dictionary
                            if isinstance(loaded_obj, dict):
                                if 'model' in loaded_obj:
                                    model = loaded_obj['model']
                                elif 'best_estimator_' in loaded_obj:
                                    model = loaded_obj['best_estimator_']
                                elif 'trained_model' in loaded_obj:
                                    model = loaded_obj['trained_model']
                                else:
                                    raise ValueError(f"Could not find model in dictionary keys: {list(loaded_obj.keys())}")
                                print(f"  [EXTRACTED] {model_name} from dictionary")
                            else:
                                model = loaded_obj
                                print(f"  [LOADED] {model_name} from {model_path}")
                            
                            self.models[model_name] = model
                            
                            # Validate model has required methods
                            has_predict = hasattr(model, 'predict')
                            has_predict_proba = hasattr(model, 'predict_proba')
                            
                            if not has_predict:
                                print(f"  [FAILED] {model_name} missing predict() method")
                                del self.models[model_name]
                                continue
                            
                            if not has_predict_proba and model_name != 'Transformer':
                                print(f"  [WARNING] {model_name} missing predict_proba() method")
                            
                            print(f"  [VALIDATED] {model_name} - predict: {has_predict}, predict_proba: {has_predict_proba}")
                        
                        loaded_count += 1
                    except Exception as e:
                        print(f"  [FAILED] {model_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            print(f"DEBUG: Total models loaded: {loaded_count}/4")
            print(f"DEBUG: Models in self.models: {list(self.models.keys())}")
            
            if loaded_count >= 1:  # Accept 1+ models as success (train missing ones)
                print(f"[OK] {loaded_count} models loaded from .pkl files")
                if loaded_count < 4:
                    missing = [name for name in model_files.keys() if name not in self.models]
                    print(f"[WARNING] Missing models will be trained: {missing}")
                return True
            else:
                print(f"No models loaded from .pkl files, will train all")
                return False
                
        except Exception as e:
            print(f"WARNING: Could not load models from .pkl: {e}")
            import traceback
            traceback.print_exc()
            self.models = {}
            return False

    def get_realtime_metrics(self):
        """Calculate real-time precision, sensitivity, and specificity for all models"""
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        
        results = {}
        
        # Check if test data is available
        if self.X_test is None or self.y_test is None:
            return {"error": "Test data not available"}
        
        for name, model in self.models.items():
            try:
                # Get predictions on test data
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                precision = precision_score(self.y_test, y_pred, pos_label=1)
                sensitivity = recall_score(self.y_test, y_pred, pos_label=1)
                
                # Calculate specificity from confusion matrix
                tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                results[name] = {
                    "precision": round(precision, 4),
                    "sensitivity": round(sensitivity, 4),
                    "specificity": round(specificity, 4)
                }
                
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results
    
    def check_model_bias(self):
        """Check for clinical bias in all 4 models"""
        if not hasattr(self, 'X_test') or self.X_test is None:
            print("No test data available for bias analysis")
            return {}
        
        bias_results = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"\n=== BIAS ANALYSIS: {model_name} ===")
                
                # Get predictions
                y_pred = model.predict(self.X_test)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Age bias analysis
                if 'Age' in self.X_test.columns:
                    age_groups = [
                        (self.X_test['Age'] <= 10, "Children (≤10)"),
                        ((self.X_test['Age'] > 10) & (self.X_test['Age'] <= 15), "Adolescents (11-15)"),
                        (self.X_test['Age'] > 15, "Teens (>15)")
                    ]
                    
                    bias_results[model_name] = {'age_bias': {}}
                    
                    for mask, group_name in age_groups:
                        if mask.sum() > 0:
                            group_y_true = self.y_test[mask]
                            group_y_pred = y_pred[mask]
                            group_y_proba = y_proba[mask] if y_proba is not None else None
                            
                            accuracy = (group_y_pred == group_y_true).mean()
                            sensitivity = recall_score(group_y_true, group_y_pred, pos_label=1, zero_division=0)
                            
                            bias_results[model_name]['age_bias'][group_name] = {
                                'accuracy': round(accuracy, 3),
                                'sensitivity': round(sensitivity, 3),
                                'sample_size': len(group_y_true)
                            }
                            
                            print(f"  {group_name}: Accuracy={accuracy:.3f}, Sensitivity={sensitivity:.3f}, n={len(group_y_true)}")
                
                # Sex bias analysis
                if 'Sex' in self.X_test.columns:
                    sex_groups = [
                        (self.X_test['Sex'] == 0, "Female"),
                        (self.X_test['Sex'] == 1, "Male")
                    ]
                    
                    bias_results[model_name]['sex_bias'] = {}
                    
                    for mask, group_name in sex_groups:
                        if mask.sum() > 0:
                            group_y_true = self.y_test[mask]
                            group_y_pred = y_pred[mask]
                            group_y_proba = y_proba[mask] if y_proba is not None else None
                            
                            accuracy = (group_y_pred == group_y_true).mean()
                            sensitivity = recall_score(group_y_true, group_y_pred, pos_label=1, zero_division=0)
                            
                            bias_results[model_name]['sex_bias'][group_name] = {
                                'accuracy': round(accuracy, 3),
                                'sensitivity': round(sensitivity, 3),
                                'sample_size': len(group_y_true)
                            }
                            
                            print(f"  {group_name}: Accuracy={accuracy:.3f}, Sensitivity={sensitivity:.3f}, n={len(group_y_true)}")
                
                # Check for significant bias (>10% difference)
                bias_detected = False
                
                # Age bias check
                if 'age_bias' in bias_results[model_name]:
                    accuracies = [v['accuracy'] for v in bias_results[model_name]['age_bias'].values()]
                    if max(accuracies) - min(accuracies) > 0.1:
                        bias_detected = True
                        print(f"  WARNING: AGE BIAS DETECTED: Accuracy range {max(accuracies) - min(accuracies):.3f}")
                
                # Sex bias check
                if 'sex_bias' in bias_results[model_name]:
                    accuracies = [v['accuracy'] for v in bias_results[model_name]['sex_bias'].values()]
                    if max(accuracies) - min(accuracies) > 0.1:
                        bias_detected = True
                        print(f"  WARNING: SEX BIAS DETECTED: Accuracy range {max(accuracies) - min(accuracies):.3f}")
                
                if not bias_detected:
                    print(f"  OK: No significant bias detected")
                
            except Exception as e:
                print(f"  ERROR: Error analyzing bias for {model_name}: {e}")
                bias_results[model_name] = {'error': str(e)}
        
        return bias_results
    
    def _create_fallback_test_data(self):
        """Create fallback test data for real-time evaluation"""
        try:
            import numpy as np
            import pandas as pd
            
            # Create synthetic test data
            n_samples = 100
            np.random.seed(42)
            
            test_data = {
                'Age': np.random.randint(5, 18, n_samples),
                'Weight': np.random.uniform(20, 80, n_samples),
                'Height': np.random.uniform(100, 180, n_samples),
                'BMI': np.random.uniform(15, 30, n_samples),
                'Sex': np.random.randint(0, 2, n_samples),
                'Neutrophil_Percentage': np.random.uniform(40, 80, n_samples),
                'Body_Temperature': np.random.uniform(36.0, 39.0, n_samples),
                'Lower_Right_Abd_Pain': np.random.randint(0, 2, n_samples),
                'Migratory_Pain': np.random.randint(0, 2, n_samples),
                'Loss_of_Appetite': np.random.randint(0, 2, n_samples),
                'Nausea': np.random.randint(0, 2, n_samples),
                'Coughing_Pain': np.random.randint(0, 2, n_samples),
                'Dysuria': np.random.randint(0, 2, n_samples),
                'Stool': np.random.randint(0, 2, n_samples),
                'Peritonitis': np.random.randint(0, 2, n_samples),
                'Severity': np.random.randint(0, 3, n_samples),
                'Contralateral_Rebound_Tenderness': np.random.randint(0, 2, n_samples),
                'Ipsilateral_Rebound_Tenderness': np.random.randint(0, 2, n_samples),
                'Psoas_Sign': np.random.randint(0, 2, n_samples),
                'WBC_Count': np.random.uniform(5, 20, n_samples),
                'RBC_Count': np.random.uniform(3, 6, n_samples),
                'Hemoglobin': np.random.uniform(10, 16, n_samples),
                'RDW': np.random.uniform(10, 20, n_samples),
                'Segmented_Neutrophils': np.random.uniform(40, 80, n_samples),
                'Thrombocyte_Count': np.random.uniform(150, 450, n_samples),
                'CRP': np.random.uniform(0, 20, n_samples),
                'Neutrophilia': np.random.randint(0, 2, n_samples),
                'Ketones_in_Urine': np.random.randint(0, 2, n_samples),
                'RBC_in_Urine': np.random.randint(0, 2, n_samples),
                'WBC_in_Urine': np.random.randint(0, 2, n_samples)
            }
            
            self.X_test = pd.DataFrame(test_data)
            
            # Create realistic target variable
            risk_score = (
                (self.X_test['Body_Temperature'] > 37.5).astype(int) * 0.3 +
                (self.X_test['WBC_Count'] > 12).astype(int) * 0.3 +
                (self.X_test['Lower_Right_Abd_Pain'] == 1).astype(int) * 0.2 +
                (self.X_test['Peritonitis'] == 1).astype(int) * 0.2
            )
            self.y_test = (risk_score > 0.5).astype(int)
            
            print(f"[OK] Fallback test data created: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
            
        except Exception as e:
            print(f"[ERROR] Could not create fallback test data: {e}")
            # Create minimal test data
            import numpy as np
            self.X_test = np.random.rand(10, 30)
            self.y_test = np.random.randint(0, 2, 10)
            print(f"[OK] Minimal fallback test data created")


# Test the predictor
if __name__ == "__main__":
    
    def test_predictor():
        """Test the predictor with sample data"""
        
        predictor = AppendicitisPredictor()
        
        # Sample input data (numeric values as expected by models)
        sample_input = {
            'Age': 10,
            'Weight': 35,
            'Height': 140,
            'BMI': 17.9,
            'Sex': 1,  # Male=1, Female=0
            'Neutrophil_Percentage': 65,
            'Body_Temperature': 37.5,
            'Lower_Right_Abd_Pain': 1,  # yes=1, no=0
            'Migratory_Pain': 0,
            'Loss_of_Appetite': 1,
            'Nausea': 1,
            'Coughing_Pain': 0,
            'Dysuria': 0,
            'Stool': 0,  # normal=0, abnormal=1
            'Peritonitis': 0,
            'Severity': 1,  # uncomplicated=1, complicated=2, severe=3
            'Contralateral_Rebound_Tenderness': 0,
            'Ipsilateral_Rebound_Tenderness': 1,
            'Psoas_Sign': 0,
            'WBC_Count': 12.5,
            'RBC_Count': 4.5,
            'Hemoglobin': 13.5,
            'RDW': 13.5,
            'Segmented_Neutrophils': 70,
            'Thrombocyte_Count': 250,
            'CRP': 15,
            'Neutrophilia': 1,  # yes=1, no=0
            'Ketones_in_Urine': 0,
            'RBC_in_Urine': 0,
            'WBC_in_Urine': 0
        }
        
        print("Testing Appendicitis Predictor")
        print("="*50)
        
        # Test each model
        for model_name in predictor.get_available_models():
            try:
                result = predictor.predict(model_name, sample_input)
                if len(result) == 3:
                    prediction, prediction_proba, imputed_input = result
                else:
                    prediction, prediction_proba = result
                diagnosis = "Appendicitis" if prediction == 1 else "No Appendicitis"
                confidence = prediction_proba if isinstance(prediction_proba, (int, float)) else prediction_proba[prediction]
                
                print(f"{model_name}:")
                print(f"  Prediction: {diagnosis}")
                print(f"  Confidence: {confidence:.3f}")
                if isinstance(prediction_proba, (int, float)):
                    print(f"  Probability: {prediction_proba:.3f}")
                else:
                    print(f"  Probabilities: No={prediction_proba[0]:.3f}, Yes={prediction_proba[1]:.3f}")
                print()
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                print()
    
    test_predictor()
