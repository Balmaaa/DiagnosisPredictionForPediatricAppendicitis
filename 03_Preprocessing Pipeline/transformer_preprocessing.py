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
    "Severity"
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
        "No":0,
        "+":1,
        "++":2,
        "+++":3
    },

    "RBC_in_Urine": {
        "No":0,
        "+":1,
        "++":2,
        "+++":3
    },

    "WBC_in_Urine": {
        "No":0,
        "+":1,
        "++":2,
        "+++":3
    }

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
            normalized.extend(["Right_Lower_Quadrant", "Right_Mid_Abdomen"])
            continue

        normalized.append(LOCATION_MAPPING.get(part, part))

    normalized = sorted(set(normalized))
    return ", ".join(normalized)


def normalize_dataframe(df):
    categorical_columns = df.select_dtypes(exclude=np.number).columns

    replacements = {
        "yes":"Yes",
        "y":"Yes",
        "true":"Yes",
        "1":"Yes",
        "ja":"Yes",

        "no":"No",
        "n":"No",
        "false":"No",
        "0":"No",
        "nein":"No",

        "male":"Male",
        "m":"Male",

        "female":"Female",
        "f":"Female"
    }

    for col in categorical_columns:
        df[col] = (df[col].where(df[col].isna(), df[col].astype(str)).str.strip().str.lower().replace(replacements))

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = (df[col].replace(mapping).astype(float))

    if "Lymph_Nodes_Location" in df.columns:
        df["Lymph_Nodes_Location"] = (df["Lymph_Nodes_Location"].apply(normalize_location_string))

    if "Abscess_Location" in df.columns:
        df["Abscess_Location"] = (df["Abscess_Location"].apply(normalize_location_string))

    return df

# ============================================================
# PREPROCESS DATASET
# ============================================================

def preprocess_dataset(df, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("TRANSFORMER PREPROCESSING", output_file)
    print_save("=" * 60, output_file)

    if df[TARGET_COLUMN].isnull().sum() > 0:
        removed = df[TARGET_COLUMN].isnull().sum()
        print_save(f"Removing {removed} samples with missing target.", output_file)
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN])
    print_save(f"Original Features : {X.shape[1]}", output_file)

    removed_columns = []

    for col in LEAKAGE_COLUMNS:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
            removed_columns.append(col)

    print_save(f"Leakage Columns Removed : {len(removed_columns)}", output_file)
    X = normalize_dataframe(X)

    for col in X.columns:
        if X[col].isnull().mean() >= 0.05:
            X[f"{col}_Missing"] = X[col].isnull().astype(int)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    print_save(f"Numerical Features : {len(numerical_features)}", output_file)
    print_save(f"Categorical Features : {len(categorical_features)}", output_file)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=RANDOM_STATE)

    print_save(f"Train Shape : {X_train.shape}", output_file)
    print_save(f"Test Shape : {X_test.shape}", output_file)

    numerical_imputer = SimpleImputer(strategy="median")
    X_train[numerical_features] = numerical_imputer.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = numerical_imputer.transform(X_val[numerical_features])
    X_test[numerical_features] = numerical_imputer.transform(X_test[numerical_features])

    categorical_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_val[categorical_features] = categorical_imputer.transform(X_val[categorical_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = scaler.transform(X_val[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    print_save(f"Train Shape : {X_train.shape}", output_file)
    print_save(f"Validation Shape : {X_val.shape}", output_file)
    print_save(f"Test Shape : {X_test.shape}", output_file)

    label_encoders = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        X_train[feature] = encoder.fit_transform(X_train[feature].astype(str))

        mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}

        unknown_value = len(encoder.classes_)

        X_val[feature] = (X_val[feature].astype(str).map(mapping).fillna(unknown_value).astype(int))
        X_test[feature] = (X_test[feature].astype(str).map(mapping).fillna(unknown_value).astype(int))
        label_encoders[feature] = encoder

    target_encoder = LabelEncoder()
    y_train = pd.Series(target_encoder.fit_transform(y_train), name=TARGET_COLUMN)
    y_val = pd.Series(target_encoder.transform(y_val), name=TARGET_COLUMN)
    y_test = pd.Series(target_encoder.transform(y_test), name=TARGET_COLUMN)
    print_save("Target successfully encoded.", output_file)

    feature_info = {}
    for feature in numerical_features:
        feature_info[feature] = {"type": "numerical"}

    for feature in categorical_features:
        encoder = label_encoders[feature]
        feature_info[feature] = {"type": "categorical", "unique_values": len(encoder.classes_) + 1}

    print_save(f"Feature Info Built : {len(feature_info)} features", output_file)
    feature_order = list(feature_info.keys())
    return(X_train, X_val, X_test, y_train, y_val, y_test, feature_info, numerical_features, categorical_features, label_encoders, scaler, numerical_imputer, categorical_imputer, target_encoder)

# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, feature_info, numerical_features, categorical_features, label_encoders, scaler, numerical_imputer, categorical_imputer, target_encoder, pickle_path, output_file):
    script_dir = Path(__file__).parent
    X_train.to_csv(script_dir / "transformer_X_train.csv", index=False)
    X_test.to_csv(script_dir / "transformer_X_test.csv", index=False)
    X_val.to_csv(script_dir/"transformer_X_val.csv", index=False)
    y_train.to_csv(script_dir / "transformer_y_train.csv", index=False)
    y_test.to_csv(script_dir / "transformer_y_test.csv", index=False)
    y_val.to_csv(script_dir/"transformer_y_val.csv", index=False)

    pipeline = {
        "feature_info": feature_info,
        "feature_order": list(feature_info.keys()),
        "feature_names": list(feature_info.keys()),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "numerical_imputer": numerical_imputer,
        "categorical_imputer": categorical_imputer,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "target_encoder": target_encoder
    }

    with open(pickle_path, "wb") as f:
        pickle.dump(pipeline, f)

    print_save(f"Pipeline Saved:\n{pickle_path}", output_file)
    print_save(f"Total Numerical Features : {len(numerical_features)}", output_file)
    print_save(f"Total Categorical Features : {len(categorical_features)}", output_file)
    print_save(f"Total Features : {len(feature_info)}", output_file)

# ============================================================
# MAIN
# ============================================================

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
        label_encoders, scaler, numerical_imputer, categorical_imputer, target_encoder) = preprocess_dataset(dataset, output_file)

        save_outputs(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_info,
            numerical_features,
            categorical_features,
            label_encoders,
            scaler,
            numerical_imputer,
            categorical_imputer,
            target_encoder,
            pickle_path,
            output_file
        )

        print_save("\n" + "=" * 70, output_file)
        print_save("PREPROCESSING FINISHED SUCCESSFULLY", output_file)
        print_save("=" * 70, output_file)

    except Exception as e:
        print_save(f"\nERROR : {e}", output_file)
        print_save(traceback.format_exc(), output_file)

    output_file.close()

    print(f"\nReport saved to : {report_path}")

if __name__ == "__main__":
    main()