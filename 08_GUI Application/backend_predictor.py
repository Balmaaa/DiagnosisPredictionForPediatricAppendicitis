import warnings
warnings.filterwarnings("ignore")
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '04_Transformer Model')))
from transformer_model import AdvancedTabularTransformer

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------

REQUIRED_FIELDS = [
    "Age",
    "Sex"
]

NON_NEGATIVE_FIELDS = [
    "Age",
    "Weight",
    "Height",
    "BMI",
    "Body_Temperature",
    "WBC_Count",
    "RBC_Count",
    "Hemoglobin",
    "RDW",
    "Segmented_Neutrophils",
    "Thrombocyte_Count",
    "CRP",
    "Neutrophil_Percentage"
]

LAB_FIELDS = [
    "WBC_Count",
    "RBC_Count",
    "Hemoglobin",
    "RDW",
    "Segmented_Neutrophils",
    "Thrombocyte_Count",
    "CRP",
    "Neutrophil_Percentage"
]


# -------------------------------------------------------
# VALIDATION
# -------------------------------------------------------

def validate_inputs(input_data):
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in input_data:
            errors.append(f"{field} is required")
            continue
        if input_data[field] is None:
            errors.append(f"{field} is required")

    for field in NON_NEGATIVE_FIELDS:
        if field not in input_data:
            continue

        value = input_data[field]
        if value is None:
            continue
        if isinstance(value, (int, float)):
            if value < 0:
                errors.append(f"{field} cannot be negative")

    return errors


# -------------------------------------------------------
# MISSING LAB HANDLING
# -------------------------------------------------------

def handle_missing_lab_values(input_data):
    patient = input_data.copy()
    missing_labs = []
    default_values = {
        "WBC_Count": np.nan,
        "RBC_Count": np.nan,
        "Hemoglobin": np.nan,
        "RDW": np.nan,
        "Segmented_Neutrophils": np.nan,
        "Thrombocyte_Count": np.nan,
        "CRP": np.nan,
        "Neutrophil_Percentage": np.nan
    }

    for lab in LAB_FIELDS:
        if lab not in patient:
            patient[lab] = default_values[lab]
            missing_labs.append(lab)
        elif patient[lab] is None:
            patient[lab] = default_values[lab]
            missing_labs.append(lab)
        elif patient[lab] == "":
            patient[lab] = default_values[lab]
            missing_labs.append(lab)
        elif isinstance(patient[lab], (int, float)):
            if patient[lab] == 0:
                patient[lab] = default_values[lab]
                missing_labs.append(lab)

    return patient, missing_labs


# -------------------------------------------------------
# TRANSFORMER INFERENCE WRAPPER
# -------------------------------------------------------

class TransformerInference:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.temperature = checkpoint.get("temperature", 1.0)
        self.feature_info = checkpoint["feature_info"]
        self.feature_order = checkpoint["feature_order"]
        self.numerical_features = checkpoint["numerical_features"]
        self.categorical_features = checkpoint["categorical_features"]
        self.model = AdvancedTabularTransformer(
            feature_info=self.feature_info,
            embed_dim=checkpoint["embed_dim"],
            num_heads=checkpoint["num_heads"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)


    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_order)
        else:
            df = X.copy()
        x_dict = {}
        for feat in self.categorical_features:
            x_dict[feat] = torch.LongTensor(df[feat].values).to(self.device)

        x_dict["numerical"] = torch.FloatTensor(df[self.numerical_features].values).to(self.device)

        with torch.no_grad():
            logits = self.model(x_dict).squeeze(-1)
            probs = torch.sigmoid(logits / self.temperature)

        p = probs.cpu().numpy()[0]
        return np.array([[1-p, p]])


# -------------------------------------------------------
# APPENDICITIS BACKEND
# -------------------------------------------------------

class AppendicitisPredictor:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.models = {}
        self.thresholds = {"Decision Tree": 0.40, "Gradient Boosting": 0.40, "XGBoost": 0.40, "Transformer": 0.70}
        self.ml_pipeline = None
        self.transformer_pipeline = None
        self.feature_columns = None
        self.is_loaded = False
        self.load_pipeline()
        self.load_models()
        self.verify_transformer_pipeline()


    def verify_transformer_pipeline(self):
        if "Transformer" not in self.models:
            return

        transformer_model = self.models["Transformer"]

        if self.transformer_feature_order != transformer_model.feature_order:
            raise RuntimeError("Transformer preprocessing feature order does not match the trained model.")
        if self.transformer_numerical != transformer_model.numerical_features:
            raise RuntimeError("Transformer numerical feature list does not match the trained model.")
        if self.transformer_categorical != transformer_model.categorical_features:
            raise RuntimeError("Transformer categorical feature list does not match the trained model.")

        print("[OK] Transformer pipeline verified.")


    def load_pipeline(self):
        pipeline_folder = self.base_path.parent / "03_Preprocessing Pipeline"
        ml_files = sorted(pipeline_folder.glob("preprocessing_pipeline_*.pkl"))
        transformer_files = sorted(pipeline_folder.glob("transformer_pipeline_*.pkl"))

        if len(ml_files) == 0:
            raise FileNotFoundError("No preprocessing pipeline found.")
        if len(transformer_files) == 0:
            raise FileNotFoundError("No transformer pipeline found.")

        with open(ml_files[-1], "rb") as f:
            pipeline = pickle.load(f)
            self.ml_pipeline = pipeline["preprocessor"]
        
        with open(transformer_files[-1], "rb") as f:
            pipeline = pickle.load(f)

        self.transformer_feature_order = pipeline["feature_order"]
        self.transformer_feature_info = pipeline["feature_info"]

        self.transformer_numerical = pipeline["numerical_features"]
        self.transformer_categorical = pipeline["categorical_features"]

        self.transformer_num_imputer = pipeline["numerical_imputer"]
        self.transformer_cat_imputer = pipeline["categorical_imputer"]

        self.transformer_scaler = pipeline["scaler"]
        self.transformer_label_encoders = pipeline["label_encoders"]

        print("[OK] ML preprocessing loaded")
        print("[OK] Transformer preprocessing loaded")

        if hasattr(self.ml_pipeline, "feature_names_in_"):
            self.feature_columns = list(self.ml_pipeline.feature_names_in_)
        elif "feature_names" in pipeline:
            self.feature_columns = pipeline["feature_names"]
        else:
            raise RuntimeError("Unable to determine ML preprocessing feature order.")


    def load_models(self):
        model_dir = self.base_path / "saved_models"
        model_files = {
            "Decision Tree": "DecisionTree.pkl",
            "Gradient Boosting": "GradientBoosting.pkl",
            "XGBoost": "XGBoost.pkl",
            "Transformer": "Transformer.pt"
        }

        loaded = 0

        for model_name, filename in model_files.items():
            model_path = model_dir / filename
            if not model_path.exists():
                print(f"[WARNING] {filename} not found.")
                continue
            try:
                if model_name == "Transformer":
                    self.models[model_name] = TransformerInference(model_path)
                else:
                    with open(model_path, "rb") as f:
                        loaded_object = pickle.load(f)

                    if isinstance(loaded_object, dict):
                        if "model" in loaded_object:
                            model = loaded_object["model"]
                            if "threshold" in loaded_object:
                                model.threshold = loaded_object["threshold"]
                            self.models[model_name] = model
                        else:
                            raise RuntimeError(f"{filename} is a dictionary but contains no 'model' key.")
                    else:
                        self.models[model_name] = loaded_object
                loaded += 1
                print(f"[OK] Loaded {model_name}")

            except Exception as e:
                print(f"[ERROR] Could not load {model_name}: {e}")
        self.is_loaded = loaded > 0
        print(f"[INFO] Loaded {loaded} model(s).")


    def preprocess_ml(self, patient):
        df = pd.DataFrame([patient])
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df.reindex(columns=self.feature_columns)
        df = df.replace("", np.nan)
        df = df.replace({None: np.nan})
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in self.feature_columns:
            if col in df.columns and col not in numeric_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except Exception:
                    pass
        return self.ml_pipeline.transform(df)


    def preprocess_transformer(self, patient):
        df = pd.DataFrame([patient])
        for col in self.transformer_feature_order:
            if col not in df.columns:
                df[col] = np.nan

        df = df.reindex(columns=self.transformer_feature_order)
        df = df.replace("", np.nan)
        df = df.replace({None: np.nan})
        for col in self.transformer_numerical:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in self.transformer_categorical:
            df[col] = df[col].astype("object")
        df[self.transformer_numerical] = self.transformer_num_imputer.transform(df[self.transformer_numerical])
        df[self.transformer_categorical] = self.transformer_cat_imputer.transform(df[self.transformer_categorical])
        df[self.transformer_numerical] = self.transformer_scaler.transform(df[self.transformer_numerical])
        for feature in self.transformer_categorical:
            encoder = self.transformer_label_encoders[feature]
            mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
            unknown = len(encoder.classes_)
            df[feature] = (df[feature].astype(str).map(mapping).fillna(unknown).astype(int))

        return df


    def get_available_models(self):
        return list(self.models.keys())


    def get_model_info(self, model_name):
        if model_name not in self.models:
            return None
        model = self.models[model_name]
        print("---------------------------")
        print("Requested:", model_name)
        print("Stored models:", self.models.keys())
        print("Loaded object:", type(self.models[model_name]))
        print(self.models[model_name])
        print("---------------------------")
        return {"name": model_name, "type": type(model).__name__, "loaded": True, "supports_probability": hasattr(model, "predict_proba")}


    def predict(self, model_name, input_data):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        errors = validate_inputs(input_data)

        if len(errors) > 0:
            raise ValueError("\n".join(errors))

        patient, missing_labs = handle_missing_lab_values(input_data)
        
        if model_name == "Transformer":
            X = self.preprocess_transformer(patient)
        else:
            X = self.preprocess_ml(patient)
        
        model = self.models[model_name]
        probabilities = model.predict_proba(X)
        prob_no = float(probabilities[0][0])
        prob_appendicitis = float(probabilities[0][1])
        threshold = getattr(model, "threshold", self.thresholds.get(model_name, 0.50))
        prediction = int(prob_appendicitis >= threshold)
        diagnosis = ("Appendicitis" if prediction == 1 else "No Appendicitis")
        lab_available = len(missing_labs) == 0
        
        return {
            "prediction": prediction,
            "diagnosis": diagnosis,
            "prob_appendicitis": round(prob_appendicitis, 4),
            "prob_no_appendicitis": round(prob_no, 4),
            "threshold": threshold,
            "missing_laboratory_fields": missing_labs,
            "laboratory_available": lab_available,
            "model": model_name
        }


    def predict_all(self, input_data):
        errors = validate_inputs(input_data)
        if errors:
            raise ValueError("\n".join(errors))
        results = {}
        for model_name in self.get_available_models():
            results[model_name] = self.predict(model_name=model_name, input_data=input_data)
        return self.build_consensus(results)



    def build_consensus(self, model_results):
        positive = 0
        negative = 0
        highest_probability = -1
        highest_model = None
        missing_fields = set()

        for model_name, result in model_results.items():
            if result["prediction"] == 1:
                positive += 1
            else:
                negative += 1

            confidence = max(result["prob_appendicitis"], result["prob_no_appendicitis"])
            if confidence > highest_probability:
                highest_probability = confidence
                highest_model = model_name
            missing_fields.update(result["missing_laboratory_fields"])

        if positive > negative:
            final_prediction = 1
            diagnosis = "Appendicitis"
        elif negative > positive:
            final_prediction = 0
            diagnosis = "No Appendicitis"
        else:
            final_prediction = None
            diagnosis = "Tie Between Models"
        agreement = max(positive, negative)
        if agreement == 4:
            confidence = "Very High"
        elif agreement == 3:
            confidence = "High"
        elif agreement == 2:
            confidence = "Low"
        else:
            confidence = "Uncertain"

        return {
            "models": model_results,
            "final_prediction": final_prediction,
            "diagnosis": diagnosis,
            "agreement": agreement,
            "total_models": len(model_results),
            "confidence": confidence,
            "highest_probability": highest_probability,
            "highest_model": highest_model,
            "missing_laboratory_fields": sorted(missing_fields),
            "laboratory_available": len(missing_fields) == 0
        }


if __name__ == "__main__":
    predictor = AppendicitisPredictor()
    print("\nAvailable Models:")
    print(predictor.get_available_models())