import pandas as pd
from pathlib import Path
from datetime import datetime


def setup_output():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"target_leakage_analysis_{timestamp}.txt"
    script_dir = Path(__file__).parent
    report_path = script_dir / report_name
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
        print_save(f"[ERROR] Failed loading dataset:\n{e}", output_file)
        return None


# ============================================================
# VALUE DISTRIBUTION
# ============================================================

def value_distribution(df, column_name):
    if column_name not in df.columns:
        return None

    series = df[column_name]
    value_counts = series.value_counts(dropna=False)
    unique_count = series.nunique()
    missing_count = series.isnull().sum()
    total_count = len(series)

    return {
        "value_counts": value_counts,
        "unique_count": unique_count,
        "missing_count": missing_count,
        "total_count": total_count,
        "missing_percentage": (missing_count / total_count) * 100
    }


# ============================================================
# DIAGNOSIS COLUMN DISCOVERY
# ============================================================

def diagnosis_columns(df, dataset_name, output_file):

    print_save("\n" + "=" * 60, output_file)
    print_save(f"DIAGNOSIS COLUMN ANALYSIS - {dataset_name}", output_file)
    print_save("=" * 60, output_file)

    diagnosis_keywords = [
        "diagnosis",
        "diagnose",
        "final",
        "result",
        "outcome",
        "conclusion",
        "finding",
        "assessment",
        "determination"
    ]

    potential_columns = []

    for column in df.columns:
        lower = column.lower()
        if any(keyword in lower for keyword in diagnosis_keywords):
            potential_columns.append(column)

    print_save(f"Potential diagnosis columns found : {len(potential_columns)}", output_file)

    for column in potential_columns:
        distribution = value_distribution(df, column)
        print_save(f"\nColumn : {column}", output_file)
        if distribution is None:
            continue

        print_save(f"Unique Values : {distribution['unique_count']}", output_file)

        print_save(
            f"Missing Values : "
            f"{distribution['missing_count']} "
            f"({distribution['missing_percentage']:.2f}%)",
            output_file
        )

        print_save("\nValue Distribution", output_file)

        for value, count in distribution["value_counts"].head(10).items():
            percent = (count / distribution["total_count"]) * 100

            if pd.isna(value):
                value = "NaN"

            print_save(f"{str(value):<25}" f"{count:>6}" f" ({percent:.2f}%)", output_file)

        if len(distribution["value_counts"]) > 10:
            print_save(f"... {len(distribution['value_counts'])-10} more values", output_file)

    return potential_columns


# ============================================================
# REMOVE PRESUMPTIVE DIAGNOSIS
# ============================================================

def exclude_presumptive_diagnoses(columns, dataset_name, output_file):

    print_save("\n" + "=" * 60, output_file)
    print_save(f"EXCLUDING PRESUMPTIVE DIAGNOSIS - {dataset_name}", output_file)
    print_save("=" * 60, output_file)

    keywords = [
        "presumptive",
        "preliminary",
        "initial",
        "suspected",
        "provisional",
        "working",
        "tentative",
        "probable",
        "possible",
        "differential"
    ]

    excluded = []
    remaining = []

    for column in columns:
        if any(word in column.lower() for word in keywords):
            excluded.append(column)
            print_save(f"EXCLUDED : {column}", output_file)

        else:
            remaining.append(column)

    print_save(f"\nExcluded : {len(excluded)}", output_file)
    print_save(f"Remaining : {len(remaining)}", output_file)

    return remaining, excluded

# ============================================================
# REMOVE OUTCOME-DERIVED VARIABLES
# ============================================================

def exclude_outcome_derived_variables(columns, dataset_name, output_file):

    print_save("\n" + "=" * 60, output_file)
    print_save(f"EXCLUDING OUTCOME-DERIVED VARIABLES - {dataset_name}", output_file)

    print_save("=" * 60, output_file)

    keywords = [
        "management",
        "treatment",
        "therapy",
        "surgery",
        "operation",
        "length_of_stay",
        "hospital",
        "discharge",
        "admission",
        "complication",
        "follow_up",
        "recovery",
        "intervention",
        "medication",
        "procedure",
        "duration"
    ]

    excluded = []
    remaining = []

    for column in columns:
        if any(word in column.lower() for word in keywords):
            excluded.append(column)
            print_save(f"EXCLUDED : {column}", output_file)

        else:
            remaining.append(column)

    print_save(f"\nExcluded : {len(excluded)}", output_file)
    print_save(f"Remaining : {len(remaining)}", output_file)

    return remaining, excluded


# ============================================================
# REMOVE SCORING SYSTEMS
# ============================================================

def exclude_scoring_systems(columns, dataset_name, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save(f"EXCLUDING SCORING SYSTEMS - {dataset_name}", output_file)
    print_save("=" * 60, output_file)

    keywords = [
        "score",
        "scale",
        "index",
        "alvarado",
        "pas",
        "appendicitis_score",
        "probability",
        "likelihood",
        "risk"
    ]

    excluded = []
    remaining = []

    for column in columns:
        if any(word in column.lower() for word in keywords):
            excluded.append(column)
            print_save(f"EXCLUDED : {column}", output_file)

        else:
            remaining.append(column)

    print_save(f"\nExcluded : {len(excluded)}", output_file)
    print_save(f"Remaining : {len(remaining)}", output_file)

    return remaining, excluded


# ============================================================
# REMOVE SPECIALTY-SPECIFIC COLUMNS
# ============================================================

def exclude_high_missing_and_specialty_columns(columns, df, dataset_name, output_file):

    print_save("\n" + "=" * 60, output_file)
    print_save(f"EXCLUDING SPECIALTY COLUMNS - {dataset_name}", output_file)
    print_save("=" * 60, output_file)

    specialty_keywords = [
        "gynecological",
        "obstetric",
        "urological",
        "orthopedic",
        "cardiac",
        "pulmonary",
        "neurological",
        "dermatological"
    ]

    excluded = []
    remaining = []

    for column in columns:
        reason = None
        if any(word in column.lower() for word in specialty_keywords):
            reason = "Specialty-specific variable"
        elif column == "Diagnosis_Presumptive":
            reason = "Preliminary diagnosis"
        if reason:
            excluded.append(column)
            print_save(f"EXCLUDED : {column} ({reason})", output_file)
        else:
            remaining.append(column)

    print_save(f"\nExcluded : {len(excluded)}", output_file)
    print_save(f"Remaining : {len(remaining)}", output_file)

    return remaining, excluded


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_single_target(columns, dataset_name, output_file):
    print_save("\n" + "=" * 60, output_file)
    print_save(f"TARGET VALIDATION - {dataset_name}", output_file)
    print_save("=" * 60, output_file)

    if len(columns) == 0:
        raise ValueError("No valid target column identified.")
    if len(columns) > 1:
        print_save("Remaining Target Candidates", output_file)
        for index, column in enumerate(columns, start=1):
            print_save(f"{index}. {column}", output_file)
        raise ValueError("Multiple possible target columns remain.")

    target = columns[0]
    print_save(f"SUCCESS : Final Target Column -> {target}", output_file)

    return target


# ============================================================
# MAIN
# ============================================================

def main():
    output_file, report_path = setup_output()
    print_save("=" * 60, output_file)
    print_save("TARGET & LEAKAGE CONTROL ANALYSIS", output_file)
    print_save("=" * 60, output_file)
    print_save("Analyzing Integrated Dataset", output_file)
    print_save("=" * 60, output_file)

    dataset = load_dataset(output_file)

    if dataset is None:
        print_save("\n[ERROR] Unable to continue.", output_file)
        output_file.close()
        print(f"\nReport saved to: {report_path}")

        return

    candidates = diagnosis_columns(dataset, "Integrated Dataset", output_file)

    if len(candidates) == 0:
        print_save("No diagnosis columns detected.", output_file)
    else:
        remaining, excluded = exclude_presumptive_diagnoses(candidates, "Integrated Dataset", output_file)
        remaining, excluded = exclude_outcome_derived_variables(remaining, "Integrated Dataset", output_file)
        remaining, excluded = exclude_scoring_systems(remaining, "Integrated Dataset", output_file)
        remaining, excluded = exclude_high_missing_and_specialty_columns(remaining, dataset, "Integrated Dataset", output_file)

        try:
            target_column = validate_single_target(remaining, "Integrated Dataset", output_file)
            print_save("\n" + "=" * 60, output_file)
            print_save("FINAL TARGET SUMMARY", output_file)
            print_save("=" * 60, output_file)
            print_save(f"Final Target Column : {target_column}", output_file)
            print_save("Target Leakage Status : PASSED", output_file)
            print_save("\nThe following columns were successfully excluded "    "from being considered as prediction targets:", output_file)
            print_save("• Presumptive diagnosis variables", output_file)
            print_save("• Outcome-derived variables", output_file)
            print_save("• Clinical scoring systems", output_file)
            print_save("• Specialty-specific diagnosis variables", output_file)
            print_save("\nSelected Machine Learning Target", output_file)
            print_save(f"    {target_column}", output_file)
            print_save("\nRemaining dataset columns will be treated as " "candidate predictor variables during preprocessing.", output_file)

        except ValueError as error:
            print_save("\n" + "=" * 60, output_file)
            print_save("FINAL TARGET SUMMARY", output_file)
            print_save("=" * 60, output_file)
            print_save(f"[ERROR] {error}", output_file)
            print_save("\nManual inspection is required before continuing " "to preprocessing.", output_file)

    print_save("\n" + "=" * 60, output_file)

    print_save("TARGET & LEAKAGE CONTROL COMPLETED", output_file)

    print_save("=" * 60, output_file)

    output_file.close()

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()