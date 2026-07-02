import pandas as pd
import re
from pathlib import Path
from datetime import datetime


def normalize_col(name):
    norm = str(name).lower()
    norm = re.sub(r"[^a-z0-9_]", "_", norm)
    norm = re.sub(r"_+", "_", norm)
    norm = norm.strip("_")
    return norm


def setup_output_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"data_audit_report_{timestamp}.txt"
    script_dir = Path(__file__).parent
    report_path = script_dir / report_filename
    report_file = open(report_path, "w", encoding="utf-8")

    return report_file, str(report_path)


def print_save(text, output_file):
    print(text)
    output_file.write(text + "\n")


# ============================================================
# DATA LOADING
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
        return df

    except Exception as e:
        print_save(f"[ERROR] Failed to load dataset:\n{e}", output_file)
        return None


# ============================================================
# DATASET INFORMATION
# ============================================================

def display_info(df, dataset_name, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save(f"DATASET : {dataset_name}", output_file)
    print_save("=" * 60, output_file)

    print_save(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns", output_file)

    print_save(f"\nColumn Names ({len(df.columns)} total)", output_file)

    for index, column in enumerate(df.columns, start=1):
        print_save(f"{index:>2}. {column}", output_file)

    print_save("\nData Types", output_file)

    datatype_summary = df.dtypes.value_counts()

    for dtype, count in datatype_summary.items():
        print_save(f"{dtype} : {count} columns", output_file)

    print_save("\nDetailed Data Types", output_file)

    for column, dtype in df.dtypes.items():
        print_save(f"{column:<40} {dtype}", output_file)


# ============================================================
# DATASET INTEGRITY
# ============================================================

def dataset_integrity(df, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save("DATASET INTEGRITY", output_file)
    print_save("=" * 60, output_file)

    if "Patient_Record_ID" in df.columns:
        duplicate_ids = df["Patient_Record_ID"].duplicated().sum()
        unique_ids = df["Patient_Record_ID"].nunique()

        print_save(f"Total Patient IDs : {unique_ids}", output_file)
        print_save(f"Duplicate Patient IDs : {duplicate_ids}", output_file)
        if duplicate_ids == 0:
            print_save("[OK] Patient_Record_ID values are unique.", output_file)
        else:
            print_save("[WARNING] Duplicate Patient_Record_ID values detected.", output_file)
    else:
        print_save("[WARNING] Patient_Record_ID column not found.", output_file)

    if "Source_Dataset" in df.columns:
        print_save("\nSource Dataset Distribution", output_file)
        source_counts = (df["Source_Dataset"].value_counts(dropna=False).sort_index())
        for source, count in source_counts.items():
            print_save(f"{str(source):<20} {count}", output_file)
    else:
        print_save("[WARNING] Source_Dataset column not found.", output_file)

    duplicate_rows = df.duplicated().sum()
    print_save(f"\nDuplicate Rows : {duplicate_rows}", output_file)

    if "Patient_Record_ID" in df.columns:
        subset_columns = [column for column in df.columns if column != "Patient_Record_ID"]
        duplicate_records = (df.duplicated(subset=subset_columns).sum())
        print_save(f"Duplicate Records (excluding Patient_Record_ID) : {duplicate_records}", output_file)


# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def analysis_normalize_col(df, dataset_name, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save(f"COLUMN NAME NORMALIZATION - {dataset_name}", output_file)
    print_save("=" * 60, output_file)
    print_save(f'{"Original Name":<40} {"Normalized Name":<40}', output_file)
    print_save(f'{"-" * 40} {"-" * 40}', output_file)

    for column in df.columns:
        normalized = normalize_col(column)
        print_save(f"{column:<40} {normalized:<40}", output_file)


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def analysis_missing_value(df, dataset_name, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save(f"MISSING VALUE ANALYSIS - {dataset_name}", output_file)
    print_save("=" * 60, output_file)

    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    total_missing = missing_counts.sum()
    total_cells = df.shape[0] * df.shape[1]
    overall_missing_percentage = (total_missing / total_cells) * 100

    print_save(f"Overall Missing Data : {total_missing:,} cells ({overall_missing_percentage:.2f}% of all data)", output_file)

    print_save(f"Columns with Missing Values : {(missing_percentages > 0).sum()} out of {len(df.columns)}", output_file)

    expected_missing_columns = {"Diagnosis_Presumptive"}

    print_save("\nEXPECTED MISSING VALUES", output_file)
    print_save("-" * 60, output_file)

    expected_found = False

    for column in expected_missing_columns:
        if column in df.columns:
            expected_found = True

            count = int(missing_counts[column])
            percent = missing_percentages[column]

            print_save(f"{column:<35} {count:>5} ({percent:.2f}%)", output_file)
            print_save("Reason : Column unavailable in Dataset 2 during integration.", output_file)
            print_save("Action : Retain column. Handle during preprocessing if needed.\n", output_file)

    if not expected_found:
        print_save("No expected missing-value columns detected.", output_file)

    print_save("\nCOLUMNS REQUIRING PREPROCESSING", output_file)
    print_save("-" * 60, output_file)

    missing_data = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": missing_counts.values,
            "Missing %": missing_percentages.values
        }
    )

    missing_data = missing_data[~missing_data["Column"].isin(expected_missing_columns)]

    missing_data = missing_data.sort_values("Missing %", ascending=False)

    print_save(f'{"Column Name":<40} {"Missing":<12} {"Percent":<12} {"Status":<18}', output_file)
    print_save("-" * 90, output_file)

    for _, row in missing_data.iterrows():

        column = row["Column"]
        count = int(row["Missing Count"])
        percent = row["Missing %"]

        if count == 0:
            status = "Complete"
        elif percent < 5:
            status = "Good"
        elif percent < 20:
            status = "Fair"
        elif percent < 50:
            status = "Poor"
        else:
            status = "Critical"

        print_save(f"{column:<40} {count:<12} {percent:<11.2f}% {status:<18}", output_file)


# ============================================================
# MAIN
# ============================================================

def main():
    output_file, report_path = setup_output_file()
    print_save("=" * 60, output_file)
    print_save("DATA AUDIT REPORT", output_file)
    print_save("=" * 60, output_file)
    print_save("Analyzing Integrated Dataset", output_file)
    print_save("=" * 60, output_file)
    
    dataset = load_dataset(output_file)

    if dataset is None:
        print_save("\n[ERROR] Unable to continue.", output_file)
        output_file.close()
        print(f"\nReport saved to: {report_path}")

        return

    display_info(dataset, "Integrated Dataset", output_file)
    dataset_integrity(dataset, output_file)
    analysis_normalize_col(dataset, "Integrated Dataset", output_file)
    analysis_missing_value(dataset, "Integrated Dataset", output_file)

    print_save("\n" + "=" * 60, output_file)
    print_save("DATA AUDIT COMPLETED", output_file)
    print_save("=" * 60, output_file)

    output_file.close()

    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()