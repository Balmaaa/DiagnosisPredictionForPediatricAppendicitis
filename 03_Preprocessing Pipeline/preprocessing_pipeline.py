import pandas as pd
import numpy as np
import pickle
import re
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

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
    "an den m. psoas rechts": "right_psoas",
    "rechter unterbauch": "right_lower_quadrant",
    "rechter mittelbauch": "right_mid_abdomen",
    "re mittelbauch": "right_mid_abdomen"
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
    report_name = f"preprocessing_pipeline_{timestamp}.txt"
    pickle_name = f"preprocessing_pipeline_{timestamp}.pkl"
    script_dir = Path(__file__).parent
    report_path = script_dir / report_name
    pickle_path = script_dir / pickle_name
    report_file = open(report_path, "w", encoding="utf-8")
    return report_file, str(report_path), str(pickle_path)


def print_save(text, output_file):
    print(text)
    output_file.write(text + "\n")


# ============================================================
# LOAD INTEGRATED DATASET
# ============================================================

def load_dataset(output_file):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_path = (project_root / "00_Dataset Integration" / "merged_dataset.csv")

    if not dataset_path.exists():
        print_save(f"[ERROR] Integrated dataset not found:\n{dataset_path}", output_file)
        return None
    try:
        df = pd.read_csv(dataset_path)
        print_save(f"[OK] Successfully loaded integrated dataset:\n{dataset_path}", output_file)
        print_save(f"Dataset Shape : {df.shape}", output_file)
        return df
    except Exception as e:
        print_save(f"[ERROR] Failed loading dataset\n{e}", output_file)
        return None


# ============================================================
# TARGET SEPARATION
# ============================================================

def separate_target(df, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("TARGET SEPARATION", output_file)
    print_save("=" * 60, output_file)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    print_save(f"Target Column : {TARGET_COLUMN}", output_file)
    print_save(f"Feature Shape : {X.shape}", output_file)
    return X, y


# ============================================================
# REMOVE LEAKAGE FEATURES
# ============================================================

def remove_leakage_columns(X, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("REMOVING LEAKAGE FEATURES", output_file)
    print_save("=" * 60, output_file)

    removed = []
    for column in LEAKAGE_COLUMNS:
        if column in X.columns:
            X = X.drop(columns=column)
            removed.append(column)
            print_save(f"Removed : {column}", output_file)

    print_save(f"\nLeakage Columns Removed : {len(removed)}", output_file)
    return X


# ============================================================
# NORMALIZE CATEGORICAL VALUES
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


def normalize_categorical_values(df, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("NORMALIZING CATEGORICAL VARIABLES", output_file)
    print_save("=" * 60, output_file)

    categorical_columns = df.select_dtypes(exclude=np.number).columns
    replacements = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "ja": "Yes",
        "no": "No", "n": "No", "false": "No", "0": "No", "nein": "No",
        "male": "Male", "m": "Male", "female": "Female", "f": "Female"
    }

    for column in categorical_columns:
        df[column] = df[column].where(df[column].isna(), df[column].astype(str)).str.strip().str.lower().replace(replacements)

    for column, mapping in ORDINAL_MAPPINGS.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping).astype(float)

    for loc_col in ["Lymph_Nodes_Location", "Abscess_Location"]:
        if loc_col in df.columns:
            df[loc_col] = df[loc_col].apply(normalize_location_string)

    print_save("Categorical values normalized.", output_file)
    return df


# ============================================================
# DATA PREPROCESSING PIPELINE
# ============================================================

def preprocess_dataset(df, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("DATA PREPROCESSING", output_file)
    print_save("=" * 60, output_file)

    if df[TARGET_COLUMN].isna().sum() > 0:
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    X, y = separate_target(df, output_file)
    X = remove_leakage_columns(X, output_file)
    
    for extra_drop in ["Source_Dataset", "US_Number"]:
        if extra_drop in X.columns:
            X = X.drop(columns=[extra_drop])

    X = normalize_categorical_values(X, output_file)

    # Stratified Split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for train_idx, test_idx in split.split(X, y):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

    # Group Rare Locations Using Training Data Only
    for col in ["Lymph_Nodes_Location", "Abscess_Location"]:
        if col in X_train.columns:
            counts = X_train[col].value_counts(dropna=True)
            common_categories = counts[counts >= 10].index
            X_train[col] = X_train[col].where(X_train[col].isna() | X_train[col].isin(common_categories), "Other")
            X_test[col] = X_test[col].where(X_test[col].isna() | X_test[col].isin(common_categories), "Other")

    # Fixed Missing Indicators based strictly on Training set criteria to guarantee schema alignment
    missing_indicator_columns = [col for col in X_train.columns if X_train[col].isnull().mean() >= 0.05]
    for col in missing_indicator_columns:
        indicator = f"{col}_Missing"
        X_train[indicator] = X_train[col].isnull().astype(int)
        X_test[indicator] = X_test[col].isnull().astype(int)

    print_save(f"Missing Indicators Created: {len(missing_indicator_columns)}", output_file)

    numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Preprocessing Column Transformer
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[("numeric", numeric_pipeline, numeric_columns), ("categorical", categorical_pipeline, categorical_columns)], remainder="drop")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Recover feature names cleanly
    feature_names = numeric_columns.copy()
    if len(categorical_columns) > 0:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        feature_names.extend(encoder.get_feature_names_out(categorical_columns).tolist())

    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    # Encode target
    label_encoder = {"no appendicitis": 0, "appendicitis": 1}
    y_train_encoded = pd.Series(y_train.astype(str).str.strip().str.lower().map(label_encoder), name=TARGET_COLUMN, dtype=int)
    y_test_encoded = pd.Series(y_test.astype(str).str.strip().str.lower().map(label_encoder), name=TARGET_COLUMN, dtype=int)

    return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, preprocessor, label_encoder, feature_names


# ============================================================
# SAVE OUTPUTS & MAIN
# ============================================================

def save_outputs(X_train, X_test, y_train, y_test, preprocessor, label_encoder, feature_names, pickle_path, output_file):
    script_dir = Path(__file__).parent
    X_train.to_csv(script_dir / "X_train.csv", index=False)
    X_test.to_csv(script_dir / "X_test.csv", index=False)
    y_train.to_csv(script_dir / "y_train.csv", index=False)
    y_test.to_csv(script_dir / "y_test.csv", index=False)

    pipeline_data = {"preprocessor": preprocessor, "label_encoder": label_encoder, "feature_names": feature_names}
    with open(pickle_path, "wb") as f:
        pickle.dump(pipeline_data, f)
    print_save(f"Saved preprocessing pipeline:\n{pickle_path}", output_file)


def main():
    output_file, report_path, pickle_path = setup_output_files()
    print_save("=" * 60, output_file)
    print_save("DATA PREPROCESSING PIPELINE (HARDENED)", output_file)
    print_save("=" * 60, output_file)

    dataset = load_dataset(output_file)
    if dataset is None:
        output_file.close()
        return

    try:
        X_train, X_test, y_train, y_test, preprocessor, label_encoder, feature_names = preprocess_dataset(dataset, output_file)
        save_outputs(X_train, X_test, y_train, y_test, preprocessor, label_encoder, feature_names, pickle_path, output_file)
        print_save("\nPreprocessing Completed Successfully.", output_file)
    except Exception as e:
        print_save(f"\nPREPROCESSING FAILED : {e}", output_file)
        print_save(traceback.format_exc(), output_file)
    finally:
        output_file.close()
        print(f"\nReport saved to : {report_path}")


if __name__ == "__main__":
    main()