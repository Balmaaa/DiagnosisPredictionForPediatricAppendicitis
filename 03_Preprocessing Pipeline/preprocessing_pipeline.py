import pandas as pd
import numpy as np
import pickle
import re
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
    "Ketones_in_Urine": {
        "No": 0,
        "+": 1,
        "++": 2,
        "+++": 3
    },

    "RBC_in_Urine": {
        "No": 0,
        "+": 1,
        "++": 2,
        "+++": 3
    },

    "WBC_in_Urine": {
        "No": 0,
        "+": 1,
        "++": 2,
        "+++": 3
    }
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
    print_save("\nTarget Distribution", output_file)

    counts = y.value_counts(dropna=False)

    for value, count in counts.items():
        percentage = count / len(y) * 100
        print_save(f"{value:<20}{count:>6} ({percentage:.2f}%)", output_file)

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

    if len(removed) == 0:
        print_save("No leakage columns detected.", output_file)

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
            normalized.extend(["Right_Lower_Quadrant", "Right_Mid_Abdomen"])
            continue

        normalized.append(LOCATION_MAPPING.get(part, part))

    normalized = sorted(set(normalized))
    return ", ".join(normalized)

def normalize_categorical_values(df, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("NORMALIZING CATEGORICAL VARIABLES", output_file)
    print_save("=" * 60, output_file)

    categorical_columns = df.select_dtypes(exclude=np.number).columns

    replacements = {
        "yes": "Yes",
        "y": "Yes",
        "true": "Yes",
        "1": "Yes",
        "ja": "Yes",

        "no": "No",
        "n": "No",
        "false": "No",
        "0": "No",
        "nein": "No",

        "male": "Male",
        "m": "Male",

        "female": "Female",
        "f": "Female"
    }

    for column in categorical_columns:
        df[column] = (df[column].where(df[column].isna(), df[column].astype(str)).str.strip().str.lower().replace(replacements))

    # ==========================================
    # ORDINAL ENCODING
    # ==========================================

    for column, mapping in ORDINAL_MAPPINGS.items():
        if column in df.columns:
            df[column] = (df[column].replace(mapping).astype(float))

    print_save("\nORDINAL COLUMN TYPES", output_file)

    for col in ORDINAL_MAPPINGS.keys():
        if col in df.columns:
            print_save(f"{col}: {df[col].dtype}", output_file)

    if "Lymph_Nodes_Location" in df.columns:
        print_save(f"Unique locations BEFORE normalization: {df['Lymph_Nodes_Location'].nunique(dropna=True)}", output_file)
        df["Lymph_Nodes_Location"] = (df["Lymph_Nodes_Location"].apply(normalize_location_string))
        counts = df["Lymph_Nodes_Location"].value_counts()
        rare_categories = counts[counts < 10].index
        df["Lymph_Nodes_Location"] = (df["Lymph_Nodes_Location"].replace(rare_categories, "Other"))
        print_save(f"Unique locations AFTER normalization: {df['Lymph_Nodes_Location'].nunique(dropna=True)}", output_file)

    if "Abscess_Location" in df.columns:
        print_save(f"Unique abscess locations BEFORE normalization: {df['Abscess_Location'].nunique(dropna=True)}", output_file)
        df["Abscess_Location"] = (df["Abscess_Location"].apply(normalize_location_string))
        counts = df["Abscess_Location"].value_counts()
        rare_categories = counts[counts < 10].index
        df["Abscess_Location"] = (df["Abscess_Location"].replace(rare_categories, "Other"))
        print_save(f"Unique abscess locations AFTER normalization: {df['Abscess_Location'].nunique(dropna=True)}", output_file)

    print_save("Categorical values normalized.", output_file)

    return df

# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_column_types(X, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("FEATURE TYPE IDENTIFICATION", output_file)
    print_save("=" * 60, output_file)

    numeric_columns = (X.select_dtypes(include=np.number).columns.tolist())
    categorical_columns = (X.select_dtypes(exclude=np.number).columns.tolist())

    print_save(f"Numeric Features : {len(numeric_columns)}", output_file)

    for column in numeric_columns:
        print_save(f"   {column}", output_file)

    print_save(f"\nCategorical Features : {len(categorical_columns)}", output_file)

    for column in categorical_columns:
        print_save(f"   {column}", output_file)

    return numeric_columns, categorical_columns


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(numeric_columns, categorical_columns, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("BUILDING PREPROCESSING PIPELINE", output_file)
    print_save("=" * 60, output_file)

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[("numeric", numeric_pipeline, numeric_columns), ("categorical", categorical_pipeline, categorical_columns)], remainder="drop")

    print_save("Preprocessing pipeline successfully created.", output_file)

    return preprocessor


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_dataset(df, output_file):

    print_save("\n" + "=" * 60, output_file)
    print_save("DATA PREPROCESSING", output_file)
    print_save("=" * 60, output_file)

    print_save(f"Original Dataset Shape : {df.shape}", output_file)

    # ----------------------------------------------------
    # Remove samples with missing target
    # ----------------------------------------------------

    missing_target = df[TARGET_COLUMN].isna().sum()

    if missing_target > 0:
        print_save(f"Removing {missing_target} samples with missing target values.", output_file)
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    print_save(f"Dataset Shape After Target Cleaning : {df.shape}", output_file)

    # ---------------------------------------
    # Separate target
    # ---------------------------------------

    X, y = separate_target(df, output_file)

    # ---------------------------------------
    # Remove leakage variables
    # ---------------------------------------

    X = remove_leakage_columns(X, output_file)
    
    if "Source_Dataset" in X.columns:
        X = X.drop(columns=["Source_Dataset"])

    if "US_Number" in X.columns:
        X = X.drop(columns=["US_Number"])

    X = normalize_categorical_values(X, output_file)

    if "Lymph_Nodes_Location" in X.columns:
        print_save("\nNORMALIZED LYMPH NODE LOCATIONS", output_file)
        print_save(str(X["Lymph_Nodes_Location"].value_counts(dropna=False)), output_file)

    # ---------------------------------------
    # Report
    # ---------------------------------------

    print_save("\n" + "=" * 60, output_file)
    print_save("CATEGORICAL CARDINALITY REPORT", output_file)
    print_save("=" * 60, output_file)

    categorical_columns = X.select_dtypes(exclude=np.number).columns

    for col in categorical_columns:

        print_save("\n" + "=" * 70, output_file)
        print_save(f"Feature : {col}", output_file)
        print_save("=" * 70, output_file)

        unique_values = X[col].value_counts(dropna=False)

        print_save(f"Number of Unique Categories : {len(unique_values)}", output_file)
        print_save("", output_file)
        print_save("Categories:", output_file)
        print_save("-" * 70, output_file)
        print_save(f"{'Category':<45}{'Count':>10}", output_file)
        print_save("-" * 70, output_file)

        for value, count in unique_values.items():

            if pd.isna(value):
                value = "<Missing>"

            print_save(f"{str(value):<45}{count:>10}", output_file)

        print_save("-" * 70, output_file)

    # ---------------------------------------
    # Identify feature types
    # ---------------------------------------

    for col in X.columns:
        missing_rate = X[col].isnull().mean()
        if missing_rate >= 0.05:
            X[f"{col}_Missing"] = X[col].isnull().astype(int)

    numeric_columns, categorical_columns = identify_column_types(X, output_file)

    print_save("\nCreating Train/Test Split...", output_file)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for train_idx, test_idx in split.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

    print_save(f"Train Shape : {X_train.shape}", output_file)
    print_save(f"Test Shape : {X_test.shape}", output_file)

    # ---------------------------------------
    # Create preprocessing object
    # ---------------------------------------

    preprocessor = create_preprocessor(numeric_columns, categorical_columns, output_file)

    print_save("\nFitting preprocessing pipeline...", output_file)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print_save(f"Processed Shape : {X_train_processed.shape}", output_file)

    # ---------------------------------------
    # Recover feature names
    # ---------------------------------------

    feature_names = []

    feature_names.extend(numeric_columns)

    if len(categorical_columns) > 0:
        encoder = (preprocessor.named_transformers_["categorical"].named_steps["encoder"])
        encoded_names = encoder.get_feature_names_out(categorical_columns)
        feature_names.extend(encoded_names.tolist())
        print_save("\nORDINAL FEATURE CHECK", output_file)
        for feature in feature_names:
            if ("Ketones" in feature or "RBC_in_Urine" in feature or "WBC_in_Urine" in feature):
                print_save(feature, output_file)

    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    print_save(f"\nFeature Reduction Summary", output_file)
    print_save(f"Original Features : {df.shape[1]-1}", output_file)
    print_save(f"Features After Leakage Removal : {X.shape[1]}", output_file)
    print_save(f"Encoded Features : {X_train_processed.shape[1]}", output_file)
    print_save(f"Final Feature Count : {len(feature_names)}", output_file)

    # ---------------------------------------
    # Encode target
    # ---------------------------------------

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_encoded = pd.Series(y_train_encoded, name=TARGET_COLUMN)
    y_test_encoded = pd.Series(y_test_encoded, name=TARGET_COLUMN)
    print_save("\nTarget Encoding", output_file)

    for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print_save(f"{original} -> {encoded}", output_file)

    missing = (X_train_processed.isnull().sum().sum() + X_test_processed.isnull().sum().sum())
    print_save(f"\nRemaining Missing Values : {missing}", output_file)

    return (X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, preprocessor, label_encoder, feature_names)


# ============================================================
# SAVE OUTPUTS
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

# ============================================================
# MAIN
# ============================================================

def main():
    global output_file

    output_file, report_path, pickle_path = setup_output_files()

    print_save("=" * 60, output_file)
    print_save("DATA PREPROCESSING PIPELINE", output_file)
    print_save("=" * 60, output_file)
    print_save("Preparing integrated dataset for machine learning", output_file)
    print_save("=" * 60, output_file)

    dataset = load_dataset(output_file)

    if dataset is None:
        print_save("\n[ERROR] Unable to continue preprocessing.", output_file)
        output_file.close()
        print(f"\nReport saved to : {report_path}")
        return

    try:
        (X_train, X_test, y_train, y_test, preprocessor, label_encoder, feature_names) = preprocess_dataset(dataset, output_file)
        save_outputs(X_train, X_test, y_train, y_test, preprocessor, label_encoder, feature_names, pickle_path, output_file)

        print_save("\n" + "=" * 60, output_file)
        print_save("PREPROCESSING SUMMARY", output_file)
        print_save("=" * 60, output_file)

        print_save(f"Original Samples : {len(dataset)}", output_file)
        print_save(f"Processed Samples : {len(X_train) + len(X_test)}", output_file)
        print_save(f"Final Features : {X_train.shape[1]}", output_file)
        print_save(f"Target Column : {TARGET_COLUMN}", output_file)
        print_save(f"Leakage Columns Removed : {len(LEAKAGE_COLUMNS)}", output_file)

        for column in LEAKAGE_COLUMNS:
            print_save(f"   • {column}", output_file)

        print_save("\n" + "=" * 60, output_file)
        print_save("DATA PREPROCESSING COMPLETED SUCCESSFULLY", output_file)
        print_save("=" * 60, output_file)

    except Exception as e:
        print_save(f"\nPREPROCESSING FAILED : {e}", output_file)
        print_save(traceback.format_exc(), output_file)

    output_file.close()
    print(f"\nReport saved to : {report_path}")


if __name__ == "__main__":

    main()