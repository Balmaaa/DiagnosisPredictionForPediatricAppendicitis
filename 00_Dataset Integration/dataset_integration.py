import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# OUTPUT SETUP
# ============================================================

def setup_output_files():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"integration_report_{timestamp}.txt"
    pickle_filename = f"integration_statistics_{timestamp}.pkl"
    script_dir = Path(__file__).parent
    report_path = script_dir / report_filename
    pickle_path = script_dir / pickle_filename
    report_file = open(report_path, "w", encoding="utf-8")

    return (report_file, str(report_path), str(pickle_path))

def print_and_save(text, output_file):
    print(text)
    output_file.write(text + "\n")


# ============================================================
# PATH DETECTION
# ============================================================

def get_dataset_paths():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    dataset1 = project_root / "Dataset 1" / "app_data.csv"
    dataset2 = project_root / "Dataset 2" / "Regensburg Pediatric Appendicitis.csv"

    if not dataset1.exists():
        raise FileNotFoundError(f"Dataset 1 not found:\n{dataset1}")
    if not dataset2.exists():
        raise FileNotFoundError(f"Dataset 2 not found:\n{dataset2}")

    return str(dataset1), str(dataset2)


# ============================================================
# DATA LOADING
# ============================================================

def load_datasets(output_file):

    dataset1_path, dataset2_path = get_dataset_paths()
    print_and_save("\nLoading datasets...", output_file)

    df1 = pd.read_csv(dataset1_path)
    df2 = pd.read_csv(dataset2_path)

    print_and_save(f"[OK] Dataset 1 loaded : {dataset1_path}", output_file)
    print_and_save(f"[OK] Dataset 2 loaded : {dataset2_path}", output_file)
    print_and_save(f"Dataset 1 Shape : {df1.shape}", output_file)
    print_and_save(f"Dataset 2 Shape : {df2.shape}", output_file)

    return df1, df2


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def remove_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def normalize_column_names(df):
    df.columns = (df.columns.str.strip().str.replace(r"\s+", "_", regex=True))
    return df


def normalize_missing_values(df):
    missing_tokens = ["", " ", "NA", "N/A", "na", "n/a", "NULL", "null", "-", "--"]
    df.replace(missing_tokens, np.nan, inplace=True)

    return df


def normalize_text_columns(df):
    object_columns = df.select_dtypes(include="object").columns

    for column in object_columns:
        df[column] = df[column].where(df[column].isna(), df[column].astype(str).str.strip())

        df[column] = df[column].replace(
            {
                "YES": "yes",
                "Yes": "yes",
                "NO": "no",
                "No": "no",
                "TRUE": "yes",
                "FALSE": "no",
                "re UB": "reUB",
                "Re UB": "reUB"
            }
        )

    return df


def standardize_numeric_columns(df):
    numeric_columns = [
        "Age",
        "BMI",
        "Height",
        "Weight",
        "Length_of_Stay",
        "Alvarado_Score",
        "Paedriatic_Appendicitis_Score",
        "Appendix_Diameter",
        "Body_Temperature",
        "WBC_Count",
        "Neutrophil_Percentage",
        "Segmented_Neutrophils",
        "RBC_Count",
        "Hemoglobin",
        "RDW",
        "Thrombocyte_Count",
        "CRP",
        "US_Number"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(df1, df2, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("SCHEMA VALIDATION", output_file)
    print_and_save("=" * 60, output_file)
    schema_valid = True
    print_and_save(f"Dataset 1 Columns : {len(df1.columns)}", output_file)
    print_and_save(f"Dataset 2 Columns : {len(df2.columns)}", output_file)

    if len(df1.columns) != len(df2.columns):
        print_and_save("[WARNING] Column count mismatch detected.", output_file)
    else:
        print_and_save("[OK] Column counts match.", output_file)

    # --------------------------------------------------------
    # Column Names
    # --------------------------------------------------------

    missing_dataset1 = sorted(set(df2.columns) - set(df1.columns))
    missing_dataset2 = sorted(set(df1.columns) - set(df2.columns))

    if missing_dataset1:
        print_and_save("\nColumns missing from Dataset 1:", output_file)
        for column in missing_dataset1:
            df2[column] = pd.Series([pd.NA] * len(df2), dtype=df1[column].dtype)
        schema_valid = False
    if missing_dataset2:
        print_and_save("\nColumns missing from Dataset 2:", output_file)
        for column in missing_dataset2:
            print_and_save(f"   {column}", output_file)
        schema_valid = False
    if not missing_dataset1 and not missing_dataset2:
        print_and_save("[OK] Column names match.", output_file)

    # --------------------------------------------------------
    # Column Order
    # --------------------------------------------------------

    if list(df1.columns) == list(df2.columns):
        print_and_save("[OK] Column order is identical.", output_file)
    else:
        print_and_save("[WARNING] Column order differs.", output_file)
        print_and_save("Aligning Dataset 2 to Dataset 1 schema.", output_file)
        df2 = df2.reindex(columns=df1.columns)
        if list(df1.columns) == list(df2.columns):
            print_and_save("[OK] Schemas successfully aligned.", output_file)
            schema_valid = True
            new_columns = [column for column in df2.columns if df2[column].isna().all()]
        if new_columns:
            print_and_save("\nAdded missing columns to Dataset 2:", output_file)
            for column in new_columns:
                print_and_save(f"   {column}", output_file)

    # --------------------------------------------------------
    # Data Types
    # --------------------------------------------------------

    datatype_mismatches = []

    for column in df1.columns:
        dtype1 = str(df1[column].dtype)
        dtype2 = str(df2[column].dtype)
        if dtype1 != dtype2:
            if df2[column].isna().all():
                continue
            datatype_mismatches.append((column, dtype1, dtype2))

    if datatype_mismatches:
        print_and_save("\nDatatype mismatches:", output_file)
        for column, d1, d2 in datatype_mismatches:
            print_and_save(f"{column}: {d1} vs {d2}", output_file)
    else:
        print_and_save("[OK] Datatypes match.", output_file)

    print_and_save("\nSchema validation completed.", output_file)

    return schema_valid, df2


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def find_exact_duplicates(df1, df2, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("EXACT DUPLICATE ANALYSIS", output_file)
    print_and_save("=" * 60, output_file)

    duplicates = pd.concat([df1, df2])
    duplicates = duplicates[duplicates.duplicated(keep=False)]
    duplicates = duplicates.drop_duplicates()
    duplicate_count = len(duplicates)

    print_and_save(f"Exact duplicate patients : {duplicate_count}", output_file)

    if duplicate_count == 0:
        print_and_save("[OK] No exact duplicate patients detected.", output_file)
    else:
        percentage = (duplicate_count / min(len(df1), len(df2))) * 100
        print_and_save(f"Duplicate percentage : {percentage:.2f}%", output_file)

    return duplicates


# ============================================================
# UNIQUE PATIENT ANALYSIS
# ============================================================

def identify_unique_records(df1, df2):
    df1 = df1.copy()
    df2 = df2.copy()

    for column in df1.columns:
        if column in df2.columns:
            df2[column] = df2[column].astype(df1[column].dtype)

    merged = df1.merge(df2, how="outer", indicator=True)
    unique_dataset1 = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    unique_dataset2 = merged[merged["_merge"] == "right_only"].drop(columns="_merge")

    return unique_dataset1, unique_dataset2


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def integration_statistics(df1, df2, duplicates, unique1, unique2, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("INTEGRATION SUMMARY", output_file)
    print_and_save("=" * 60, output_file)
    print_and_save(f"Dataset 1 Patients : {len(df1)}", output_file)
    print_and_save(f"Dataset 2 Patients : {len(df2)}", output_file)
    print_and_save(f"Exact Duplicates : {len(duplicates)}", output_file)
    print_and_save(f"Unique Dataset 1 : {len(unique1)}", output_file)
    print_and_save(f"Unique Dataset 2 : {len(unique2)}", output_file)
    projected = (len(df1) + len(df2) - len(duplicates))
    print_and_save(f"Projected Merged Size : {projected}", output_file)

    statistics = {
        "dataset1_rows": len(df1),
        "dataset2_rows": len(df2),
        "duplicates": len(duplicates),
        "unique_dataset1": len(unique1),
        "unique_dataset2": len(unique2),
        "projected_rows": projected
    }

    return statistics

# ============================================================
# NEAR DUPLICATE ANALYSIS
# ============================================================

def find_near_duplicates(df1, df2, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("NEAR DUPLICATE ANALYSIS", output_file)
    print_and_save("=" * 60, output_file)

    candidate_columns = [
        "Age",
        "Sex",
        "Height",
        "Weight",
        "BMI",
        "Diagnosis",
        "Appendix_Diameter",
        "CRP",
        "WBC_Count"
    ]

    available_columns = [column for column in candidate_columns if column in df1.columns and column in df2.columns]

    if len(available_columns) < 3:
        print_and_save("Insufficient common columns for near duplicate detection.", output_file)
        return pd.DataFrame()

    possible_duplicates = pd.merge(df1, df2, how="inner", on=available_columns, suffixes=("_Dataset1", "_Dataset2"))
    print_and_save(f"Potential duplicate patients : {len(possible_duplicates)}", output_file)

    return possible_duplicates


# ============================================================
# SOURCE LABELING
# ============================================================

def add_dataset_source(df, source_name):
    df = df.copy()
    df.insert(0, "Source_Dataset", source_name)
    return df


# ============================================================
# MERGING
# ============================================================

def merge_datasets(df1, df2, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("DATASET MERGING", output_file)
    print_and_save("=" * 60, output_file)

    df1 = add_dataset_source(df1.copy(), "Older_Cohort")
    df2 = add_dataset_source(df2.copy(), "Newer_Cohort")

    merged = pd.concat([df1, df2], ignore_index=True)
    before = len(merged)
    duplicate_subset = [column for column in merged.columns if column != "Source_Dataset"]
    merged = merged.drop_duplicates(subset=duplicate_subset, keep="last")
    removed = before - len(merged)
    
    print_and_save(f"Rows before duplicate removal : {before}", output_file)
    print_and_save(f"Duplicate rows removed : {removed}", output_file)
    print_and_save(f"Final merged dataset : {len(merged)} patients", output_file)

    return merged


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

def save_outputs(merged, duplicates, possible_duplicates, unique_dataset1, unique_dataset2, statistics, pickle_path, output_file):

    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("SAVING OUTPUT FILES", output_file)
    print_and_save("=" * 60, output_file)

    script_dir = Path(__file__).parent
    merged_path = script_dir / "merged_dataset.csv"
    merged.to_csv(merged_path, index=False)
    print_and_save(f"Saved : {merged_path.name}", output_file)
    duplicates.to_csv(script_dir / "duplicate_patients.csv", index=False)
    possible_duplicates.to_csv(script_dir / "possible_duplicate_patients.csv", index=False)
    unique_dataset1.to_csv(script_dir / "unique_dataset1.csv", index=False)
    unique_dataset2.to_csv(script_dir / "unique_dataset2.csv", index=False)
    pd.DataFrame([statistics]).to_csv(script_dir / "duplicate_summary.csv", index=False)

    with open(pickle_path, "wb") as file:
        pickle.dump(statistics, file)

    print_and_save("All integration files saved successfully.", output_file)

    # ============================================================
# FINAL DATASET PREPARATION
# ============================================================

def prepare_final_dataset(merged, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("FINAL DATASET PREPARATION", output_file)
    print_and_save("=" * 60, output_file)

    merged = merged.reset_index(drop=True)
    merged.insert(0, "Patient_Record_ID", [f"PID_{i+1:06d}" for i in range(len(merged))])
    priority = ["Patient_Record_ID", "Source_Dataset"]
    remaining = [column for column in merged.columns if column not in priority]
    merged = merged[priority + remaining]

    print_and_save(f"Patient IDs generated : {len(merged)}", output_file)
    print_and_save("Dataset organization completed.", output_file)

    return merged


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def dataset_quality_summary(merged, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("FINAL DATA QUALITY", output_file)
    print_and_save("=" * 60, output_file)

    missing = merged.isnull().sum()
    missing = missing[missing > 0]

    print_and_save(f"Rows : {merged.shape[0]}", output_file)
    print_and_save(f"Columns : {merged.shape[1]}", output_file)
    print_and_save(f"Duplicate Rows : {merged.duplicated().sum()}", output_file)
    print_and_save(f"Columns with Missing Values : {len(missing)}", output_file)

    if len(missing):
        print_and_save("\nMissing Value Summary", output_file)
        for column, count in missing.items():
            percent = (count /len(merged)) * 100
            print_and_save(f"{column:<35} {count:>5} ({percent:.2f}%)", output_file)


# ============================================================
# PIPELINE EXECUTION
# ============================================================

def integrate_datasets(output_file, pickle_path):

    df1, df2 = load_datasets(output_file)

    df1 = remove_unnamed_columns(df1)
    df2 = remove_unnamed_columns(df2)

    df1 = normalize_column_names(df1)
    df2 = normalize_column_names(df2)

    df1 = normalize_missing_values(df1)
    df2 = normalize_missing_values(df2)

    df1 = normalize_text_columns(df1)
    df2 = normalize_text_columns(df2)

    df1 = standardize_numeric_columns(df1)
    df2 = standardize_numeric_columns(df2)

    valid, df2 = validate_schema(df1, df2, output_file)

    if not valid:
        raise ValueError("Schema validation failed.")

    duplicates = find_exact_duplicates(df1, df2, output_file)
    possible_duplicates = find_near_duplicates(df1, df2, output_file)
    unique1, unique2 = identify_unique_records(df1, df2)
    statistics = integration_statistics(df1, df2, duplicates, unique1, unique2, output_file)
    merged = merge_datasets(df1, df2, output_file)
    merged = prepare_final_dataset(merged, output_file)
    dataset_quality_summary(merged, output_file)
    save_outputs(merged, duplicates, possible_duplicates, unique1, unique2, statistics, pickle_path, output_file)

    return merged

# ============================================================
# MAIN
# ============================================================

def main():
    output_file, report_path, pickle_path = setup_output_files()
    print_and_save("=" * 60, output_file)
    print_and_save("DATASET INTEGRATION", output_file)
    print_and_save("=" * 60, output_file)

    try:
        integrate_datasets(output_file, pickle_path)
        print_and_save("\nDataset integration completed successfully.", output_file)
    except Exception as e:
        print_and_save(f"\nIntegration failed: {e}", output_file)
        import traceback
        print_and_save(traceback.format_exc(), output_file)
    finally:
        output_file.close()
        print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()