import pandas as pd
import numpy as np
import pickle
import re
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


TARGET_COLUMN = "Diagnosis"

LEAKAGE_COLUMNS = [
    "Diagnosis_Presumptive",
    "Gynecological_Findings",
    "Management",
    "Length_of_Stay",
    "Alvarado_Score",
    "Paedriatic_Appendicitis_Score",
    "Patient_Record_ID",
    "Source_Dataset",
    "Severity",
    "US_Number"
]

RANDOM_STATE = 42

LOCATION_MAPPING = {
    "reub": "right_lower_quadrant",
    "re ub": "right_lower_quadrant",
    "ub": "right_lower_quadrant",
    "rechter unterbauch": "right_lower_quadrant",
    "re mb": "right_mid_abdomen",
    "mb": "right_mid_abdomen",
    "rechter mittelbauch": "right_mid_abdomen",
    "re mittelbauch": "right_mid_abdomen",
    "ileocoecal": "ileocecal",
    "ileocoekal": "ileocecal",
    "ileocöcal": "ileocecal",
    "ileozökal": "ileocecal",
    "mesenterial": "mesenteric",
    "periumbilikal": "periumbilical",
    "periappendikulär": "periappendiceal",
    "lokal um die appendix": "periappendiceal",
    "inguinal": "inguinal",
    "links inguinal": "inguinal",
    "multiple lokalisationen": "multiple",
    "lymphadenopathie": "other",
    "ovarialzysten": "other",
    "douglas": "douglas_pouch",
    "retrovesikal": "retrovesical",
    "perityphlitisch": "perityphlitic",
    "an den m. psoas rechts": "right_psoas"
}

ORDINAL_MAPPINGS = {
    "Ketones_in_Urine": {"No": 0, "+": 1, "++": 2, "+++": 3},
    "RBC_in_Urine": {"No": 0, "+": 1, "++": 2, "+++": 3},
    "WBC_in_Urine": {"No": 0, "+": 1, "++": 2, "+++": 3}
}

# ============================================================
# OUTPUT SETUP
# ============================================================

def setup_output_files():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"transformer_preprocessing_{timestamp}.txt"
    pickle_name = f"transformer_pipeline_{timestamp}.pkl"
    script_dir = Path(__file__).parent
    report_path = script_dir / report_name
    pickle_path = script_dir / pickle_name
    report_file = open(report_path, "w", encoding="utf-8")
    return report_file, report_path, pickle_path


def print_save(text, output_file):
    print(text)
    output_file.write(text + "\n")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(output_file):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_path = (project_root / "00_Dataset Integration" / "merged_dataset.csv")

    if not dataset_path.exists():
        print_save(f"[ERROR] Dataset not found\n{dataset_path}", output_file)
        return None

    df = pd.read_csv(dataset_path)
    print_save(f"[OK] Dataset Loaded\nShape : {df.shape}", output_file)
    return df


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_location_string(value):
    if pd.isna(value):
        return value
    value = str(value).strip().lower()
    parts = re.split(r'[,;/+]| and ', value)
    parts = [p.strip() for p in parts if p.strip()]
    normalized = []
    for part in parts:
        if part == "rechter unter- und mittelbauch":
            normalized.extend(["right_lower_quadrant", "right_mid_abdomen"])
            continue
        normalized.append(LOCATION_MAPPING.get(part, part))
    return ", ".join(sorted(set(normalized)))


def normalize_dataframe(df):
    df = df.copy()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    replacements = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "ja": "Yes",
        "no": "No", "n": "No", "false": "No", "0": "No", "nein": "No",
        "male": "Male", "m": "Male", "female": "Female", "f": "Female"
    }

    for col in categorical_columns:
        df[col] = df[col].where(df[col].isna(), df[col].astype(str)).str.strip().str.lower().replace(replacements)

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping).astype(float)

    for col in ["Lymph_Nodes_Location", "Abscess_Location"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_location_string)

    return df


# ============================================================
# PREPROCESS DATASET
# ============================================================

def preprocess_dataset(df, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("TRANSFORMER PREPROCESSING (HARDENED)", output_file)
    print_save("=" * 60, output_file)

    if df[TARGET_COLUMN].isnull().sum() > 0:
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN])

    for col in LEAKAGE_COLUMNS:
        if col in X.columns:
            X.drop(columns=col, inplace=True)

    X = normalize_dataframe(X)

    # Three-way stratified split: Train (64%), Validation (16%), Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=RANDOM_STATE)

    # Consistent missing indicators derived strictly from training data
    missing_indicator_columns = [col for col in X_train.columns if X_train[col].isnull().mean() >= 0.05]
    for col in missing_indicator_columns:
        missing_col = f"{col}_Missing"
        X_train[missing_col] = X_train[col].isnull().astype(int)
        X_val[missing_col] = X_val[col].isnull().astype(int)
        X_test[missing_col] = X_test[col].isnull().astype(int)

    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Imputation
    numerical_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train[numerical_features] = numerical_imputer.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = numerical_imputer.transform(X_val[numerical_features])
    X_test[numerical_features] = numerical_imputer.transform(X_test[numerical_features])

    categorical_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_val[categorical_features] = categorical_imputer.transform(X_val[categorical_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

    # Scaling
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = scaler.transform(X_val[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Label Encoding for Transformer Embeddings
    label_encoders = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        X_train[feature] = encoder.fit_transform(X_train[feature].astype(str))
        mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
        unknown_value = len(encoder.classes_)

        X_val[feature] = X_val[feature].astype(str).map(mapping).fillna(unknown_value).astype(int)
        X_test[feature] = X_test[feature].astype(str).map(mapping).fillna(unknown_value).astype(int)
        label_encoders[feature] = encoder

    # Target Encoding
    target_encoder = {"no appendicitis": 0, "appendicitis": 1}
    y_train = pd.Series(y_train.astype(str).str.strip().str.lower().map(target_encoder), name=TARGET_COLUMN, dtype=int)
    y_val = pd.Series(y_val.astype(str).str.strip().str.lower().map(target_encoder), name=TARGET_COLUMN, dtype=int)
    y_test = pd.Series(y_test.astype(str).str.strip().str.lower().map(target_encoder), name=TARGET_COLUMN, dtype=int)

    # Feature Metadata
    feature_info = {}
    for feature in X_train.columns:
        if feature in numerical_features:
            feature_info[feature] = {"type": "numerical"}
        else:
            encoder = label_encoders[feature]
            feature_info[feature] = {"type": "categorical", "unique_values": len(encoder.classes_) + 1}

    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_info, numerical_features, categorical_features, 
            missing_indicator_columns, label_encoders, scaler, numerical_imputer, categorical_imputer, target_encoder)


# ============================================================
# SAVE OUTPUTS & MAIN
# ============================================================

def save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, feature_info, numerical_features, categorical_features, 
                 missing_indicator_columns, label_encoders, scaler, numerical_imputer, categorical_imputer, target_encoder, pickle_path, output_file):
    script_dir = Path(__file__).parent
    X_train.to_csv(script_dir / "transformer_X_train.csv", index=False)
    X_val.to_csv(script_dir / "transformer_X_val.csv", index=False)
    X_test.to_csv(script_dir / "transformer_X_test.csv", index=False)
    y_train.to_csv(script_dir / "transformer_y_train.csv", index=False)
    y_val.to_csv(script_dir / "transformer_y_val.csv", index=False)
    y_test.to_csv(script_dir / "transformer_y_test.csv", index=False)

    pipeline = {
        "feature_info": feature_info,
        "feature_order": list(feature_info.keys()),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "missing_indicator_columns": missing_indicator_columns,
        "numerical_imputer": numerical_imputer,
        "categorical_imputer": categorical_imputer,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "target_encoder": target_encoder
    }

    with open(pickle_path, "wb") as f:
        pickle.dump(pipeline, f)

    print_save(f"Transformer Pipeline Saved Successfully:\n{pickle_path}", output_file)


def main():
    output_file, report_path, pickle_path = setup_output_files()
    print_save("=" * 70, output_file)
    print_save("TRANSFORMER PREPROCESSING PIPELINE", output_file)
    print_save("=" * 70, output_file)
    dataset = load_dataset(output_file)

    if dataset is None:
        output_file.close()
        return

    try:
        (X_train, X_val, X_test, y_train, y_val, y_test, feature_info, numerical_features, categorical_features,
         missing_indicator_columns, label_encoders, scaler, numerical_imputer, categorical_imputer, target_encoder) = preprocess_dataset(dataset, output_file)

        save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, feature_info, numerical_features, categorical_features,
                     missing_indicator_columns, label_encoders, scaler, numerical_imputer, categorical_imputer, target_encoder, 
                     pickle_path, output_file)

        print_save("\nPreprocessing Finished Successfully.", output_file)
    except Exception as e:
        print_save(f"\nERROR : {e}", output_file)
        print_save(traceback.format_exc(), output_file)
    finally:
        output_file.close()
        print(f"\nReport saved to : {report_path}")


if __name__ == "__main__":
    main()