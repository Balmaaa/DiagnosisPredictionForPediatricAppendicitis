import pandas as pd
import numpy as np
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
    script_dir = Path(__file__).resolve().parent
    report_path = script_dir / report_filename
    pickle_path = script_dir / pickle_filename
    report_file = open(report_path, "w", encoding="utf-8")
    return (report_file, str(report_path), str(pickle_path))


def print_and_save(text, output_file):
    print(text)
    output_file.write(text + "\n")


def get_dataset_path():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset1 = project_root / "Dataset 1" / "app_data.csv"

    if not dataset1.exists():
        raise FileNotFoundError(f"Dataset 1 not found:\n{dataset1}")

    return str(dataset1)


def load_dataset(output_file):
    dataset1_path = get_dataset_path()
    print_and_save("\nLoading primary dataset...", output_file)
    df = pd.read_csv(dataset1_path)
    print_and_save(f"[OK] Dataset 1 loaded : {dataset1_path}", output_file)
    print_and_save(f"Dataset 1 Shape : {df.shape}", output_file)
    return df


def remove_unnamed_columns(df):
    unnamed_columns = [column for column in df.columns if str(column).strip().startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)
    return df


def normalize_column_names(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.replace(r"\s+", "_", regex=True))
    return df


def normalize_missing_values(df):
    df = df.copy()
    missing_tokens = ["", " ", "NA", "N/A", "na", "n/a", "NULL", "null", "-", "--"]
    df.replace(missing_tokens, np.nan, inplace=True)
    return df


def normalize_text_columns(df):
    df = df.copy()
    object_columns = df.select_dtypes(include="object").columns
    for column in object_columns:
        df[column] = (df[column].where(df[column].isna(), df[column].astype(str).str.strip()))

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
    df = df.copy()
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


def validate_dataset(df, output_file):

    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("DATASET VALIDATION", output_file)
    print_and_save("=" * 60, output_file)
    validation_passed = True
    
    print_and_save(f"Dataset Columns : {len(df.columns)}", output_file)

    if df.columns.duplicated().any():
        duplicated_columns = (df.columns[ df.columns.duplicated() ].tolist() )
        print_and_save("[ERROR] Duplicate column names detected:", output_file)
        for column in duplicated_columns:
            print_and_save(f"   {column}", output_file)
        validation_passed = False

    else:
        print_and_save("[OK] Column names are unique.", output_file)

    target_column = "Diagnosis"

    if target_column not in df.columns:
        print_and_save(f"[ERROR] Required target column " f"'{target_column}' is missing.", output_file)
        validation_passed = False

    else:
        missing_targets = int(df[target_column].isna().sum())
        print_and_save(f"Target Column : {target_column}", output_file)
        print_and_save(f"Missing Target Values : {missing_targets}", output_file)

        if missing_targets > 0:
            print_and_save("[WARNING] Records with missing target " "values will require handling downstream.", output_file)

        target_values = (df[target_column].dropna().astype(str).str.strip().str.lower().unique())
        print_and_save(f"Observed Target Categories : {len(target_values)}", output_file)

        for value in sorted(target_values):
            print_and_save(f"   {value}", output_file)

    print_and_save(f"Total Records : {len(df)}", output_file)

    if df.empty:
        print_and_save("[ERROR] Dataset contains no records.", output_file)
        validation_passed = False

    duplicate_count = int(df.duplicated().sum())
    print_and_save(f"Exact Duplicate Rows : {duplicate_count}", output_file)

    if duplicate_count > 0:
        print_and_save("[WARNING] Exact duplicate rows exist " "within Dataset 1.", output_file)
    else:
        print_and_save("[OK] No exact duplicate rows detected.", output_file)

    if validation_passed:
        print_and_save("\n[OK] Dataset validation passed.", output_file)
    else:
        print_and_save("\n[ERROR] Dataset validation failed.", output_file)

    return validation_passed


def calculate_dataset_statistics(df, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("DATASET STATISTICS", output_file)
    print_and_save("=" * 60, output_file)

    row_count = len(df)
    column_count = len(df.columns)

    duplicate_count = int(df.duplicated().sum())

    target_column = "Diagnosis"

    missing_target_count = 0
    target_distribution = {}

    if target_column in df.columns:
        missing_target_count = int(df[target_column].isna().sum())
        target_distribution = (df[target_column].dropna().astype(str).str.strip().str.lower().value_counts().to_dict())

    print_and_save(f"Dataset Records : {row_count}", output_file)
    print_and_save(f"Dataset Features : {column_count}", output_file)
    print_and_save(f"Exact Duplicate Rows : {duplicate_count}", output_file)
    print_and_save(f"Missing Diagnosis Values : {missing_target_count}", output_file)

    if target_distribution:
        print_and_save("\nDiagnosis Distribution", output_file)

        for diagnosis, count in sorted(target_distribution.items()):
            percentage = (count / (row_count - missing_target_count)) * 100
            print_and_save(f"{diagnosis:<25} " f"{count:>5} " f"({percentage:.2f}%)", output_file)

    statistics = {
        "dataset_source": "Dataset 1",
        "dataset_path": get_dataset_path(),
        "dataset_rows": row_count,
        "dataset_columns": column_count,
        "duplicate_rows": duplicate_count,
        "missing_target_values": missing_target_count,
        "target_distribution": target_distribution,
        "generated_at": datetime.now().isoformat()
    }

    return statistics


def save_outputs(df, statistics, pickle_path, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("SAVING OUTPUT FILES", output_file)
    print_and_save("=" * 60, output_file)
    script_dir = Path(__file__).resolve().parent

    merged_path = (script_dir / "merged_dataset.csv")
    df.to_csv(merged_path, index=False)
    print_and_save(f"Saved : {merged_path.name}", output_file)

    empty_duplicates = pd.DataFrame(columns=df.columns)
    empty_possible_duplicates = pd.DataFrame(columns=df.columns)
    empty_unique_dataset2 = pd.DataFrame(columns=df.columns)
    empty_duplicates.to_csv(script_dir / "duplicate_patients.csv", index=False)
    empty_possible_duplicates.to_csv(script_dir / "possible_duplicate_patients.csv", index=False)
    df.to_csv(script_dir / "unique_dataset1.csv", index=False)
    empty_unique_dataset2.to_csv(script_dir / "unique_dataset2.csv", index=False)
    pd.DataFrame([statistics]).to_csv(script_dir / "duplicate_summary.csv", index=False)

    with open(pickle_path, "wb") as file:
        pickle.dump(statistics, file)

    print_and_save("All integration files saved successfully.", output_file)


def prepare_final_dataset(df, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("FINAL DATASET PREPARATION", output_file)
    print_and_save("=" * 60, output_file)

    df = df.copy().reset_index(drop=True)

    if "Patient_Record_ID" in df.columns:
        df = df.drop(columns=["Patient_Record_ID"])

    if "Source_Dataset" in df.columns:
        df = df.drop(columns=["Source_Dataset"])

    df.insert(0, "Patient_Record_ID", [ f"PID_{i + 1:06d}" for i in range(len(df))])
    df.insert(1, "Source_Dataset", "Dataset_1")

    print_and_save(f"Patient IDs generated : {len(df)}", output_file)
    print_and_save("Dataset source : Dataset 1 only", output_file)
    print_and_save("Dataset organization completed.", output_file)

    return df


def dataset_quality_summary(df, output_file):
    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("FINAL DATA QUALITY", output_file)
    print_and_save("=" * 60, output_file)
    print_and_save("", output_file)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print_and_save(f"Rows : {df.shape[0]}", output_file)
    print_and_save(f"Columns : {df.shape[1]}", output_file)
    print_and_save(f"Duplicate Rows : {df.duplicated().sum()}", output_file)
    print_and_save(f"Columns with Missing Values : {len(missing)}", output_file)

    if len(missing):
        print_and_save("\nMissing Value Summary", output_file)
        for column, count in missing.items():
            percent = (count / len(df)) * 100
            print_and_save(f"{column:<35} {count:>5} ({percent:.2f}%)", output_file)


def integrate_dataset(output_file, pickle_path):
    df = load_dataset(output_file)
    df = remove_unnamed_columns(df)
    df = normalize_column_names(df)
    df = normalize_missing_values(df)
    df = normalize_text_columns(df)
    df = standardize_numeric_columns(df)

    print_and_save("\n" + "=" * 60, output_file)
    print_and_save("DATA CLEANING SUMMARY", output_file)
    print_and_save("=" * 60, output_file)
    print_and_save("=" * 60, output_file)
    print_and_save(f"Records after cleaning : {len(df)}", output_file)
    print_and_save(f"Columns after cleaning : {len(df.columns)}", output_file)

    valid = validate_dataset(df, output_file)
    if not valid:
        raise ValueError("Dataset validation failed. " "Review the integration report before proceeding.")

    statistics = calculate_dataset_statistics(df, output_file)

    df = prepare_final_dataset(df, output_file)
    dataset_quality_summary(df, output_file)
    save_outputs(df, statistics, pickle_path, output_file)

    return df


def main():
    output_file, report_path, pickle_path = setup_output_files()
    print_and_save("=" * 60, output_file)
    print_and_save("DATASET INTEGRATION - DATASET 1 ONLY", output_file)
    print_and_save("=" * 60, output_file)

    try:
        integrate_dataset(output_file, pickle_path)
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