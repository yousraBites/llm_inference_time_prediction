import os
import json
import pandas as pd

BASE_FOLDER = os.path.expanduser("~/Desktop/dataset")
INPUT_FOLDER = os.path.join(BASE_FOLDER, "dataSets")
OUTPUT_XLSX = os.path.join(BASE_FOLDER, "dataset.xlsx")
OUTPUT_CSV = os.path.join(BASE_FOLDER, "dataset.csv")

COLUMNS_TO_KEEP = [
    "model",
    "nb_input_token",
    "temperature",
    "top_p",
    "top_k",
    "nb_user",
    "gpu_model",
    "gpu_fp32_tflops",
    "gpu_memory_gib",
    "gpu_num",
    "model_num_params",
    "nb_output_token",
    "inference_time",
]

ROUND_DIGITS = 3


def read_jsonl_file(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                clean_record = {}

                for col in COLUMNS_TO_KEEP:
                    clean_record[col] = record.get(col, None)

                rows.append(clean_record)

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON in {file_path}, line {line_num}: {e}")

    return rows


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Folder not found: {INPUT_FOLDER}")
        return

    jsonl_files = sorted(
        [
            os.path.join(INPUT_FOLDER, f)
            for f in os.listdir(INPUT_FOLDER)
            if f.endswith(".jsonl")
        ]
    )

    if not jsonl_files:
        print(f"No .jsonl files found in: {INPUT_FOLDER}")
        return

    all_rows = []
    for file_path in jsonl_files:
        all_rows.extend(read_jsonl_file(file_path))

    if not all_rows:
        print("No valid rows found.")
        return

    df = pd.DataFrame(all_rows)

    # Round numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].round(ROUND_DIGITS)

    # Optional sort
    sort_cols = [c for c in ["model", "temperature", "top_k", "top_p", "nb_user"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # Save Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dataset")

        ws = writer.sheets["knn_dataset"]
        for col_cells in ws.columns:
            ws.column_dimensions[col_cells[0].column_letter].width = 20

    print("Done.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Excel saved to: {OUTPUT_XLSX}")
    print(f"CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()