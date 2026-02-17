#!/usr/bin/env python3
"""
convert_master_sheet_to_csv.py (CLI)
=====================================

Purpose:
    Convert all (or selected) sheets from an Excel workbook into CSV files.

Layer: Curate / Utilities

Usage Examples:
    python tools/curate/utils/convert_master_sheet_to_csv.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    --excel         : Path to the Excel workbook (.xlsx). If omitted, the default JAM master path in this folder will be used.
    --outdir        : Output directory for CSV files (default: ./csv)
    --sheets        : Comma-separated list of sheet names to convert (default: all sheets)
    --write-empty   : Write a CSV even if the sheet has no non-empty rows/columns
    --same-name     : When converting a workbook with exactly one non-empty sheet, also write the CSV with the same base name as the workbook (e.g., JAM-SharePoint-MasterSheet.csv)

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - sanitize_sheet_name(): No description.
    - convert_excel_to_csv(): No description.
    - main(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
from pathlib import Path
import argparse
import sys
import re

try:
    import pandas as pd
except Exception as e:
    print("Missing dependency: pandas. Install with `pip install pandas openpyxl` and try again.")
    raise e


def sanitize_sheet_name(name: str) -> str:
    # make filename-safe: replace spaces and illegal chars
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = name.replace(" ", "_")
    # truncate if very long
    return name[:120]


def convert_excel_to_csv(excel_path: Path, out_dir: Path, sheets=None, write_empty=False, encoding="utf-8-sig") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read either all sheets or the selected ones
    if sheets:
        # Accept comma-separated list
        sheet_list = [s.strip() for s in sheets.split(",") if s.strip()]
        # pandas accepts list of names
        df_map = pd.read_excel(excel_path, sheet_name=sheet_list, engine="openpyxl")
    else:
        df_map = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")

    summary = {"written": [], "skipped": []}

    last_written = None
    written_count = 0
    for sheet_name, df in df_map.items():
        # If df is None (happens if sheet empty) create empty DataFrame
        if df is None:
            if write_empty:
                out_path = out_dir / f"{sanitize_sheet_name(sheet_name)}.csv"
                out_path.write_text("", encoding=encoding)
                summary["written"].append(str(out_path))
                last_written = (sheet_name, df)
                written_count += 1
            else:
                summary["skipped"].append(sheet_name)
            continue

        # Drop rows & columns that are completely empty
        df2 = df.dropna(axis=0, how="all")
        df2 = df2.dropna(axis=1, how="all")

        if df2.empty and not write_empty:
            summary["skipped"].append(sheet_name)
            continue

        out_path = out_dir / f"{sanitize_sheet_name(sheet_name)}.csv"
        df2.to_csv(out_path, index=False, encoding=encoding)
        summary["written"].append(str(out_path))
        last_written = (sheet_name, df2)
        written_count += 1

    # If --same-name requested and exactly one non-empty/written sheet, also write a CSV named after the workbook base
    if written_count == 1 and 'same_name' in locals() and locals().get('same_name'):
        # locals usage is a precaution — we set this in the caller via args
        base_name = excel_path.stem
        single_out = out_dir / f"{base_name}.csv"
        # last_written[1] is the dataframe (or None if it was empty-but-write_empty)
        if last_written and last_written[1] is not None:
            last_written[1].to_csv(single_out, index=False, encoding=encoding)
            summary["written"].append(str(single_out))
        else:
            # If the sheet was empty but write_empty==True, create an empty file
            single_out.write_text("", encoding=encoding)
            summary["written"].append(str(single_out))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert Excel workbook sheets to CSV files")
    parser.add_argument("--excel", "-e", required=True,
                        help="Path to the Excel workbook (.xlsx).")
    parser.add_argument("--outdir", "-o", default="./csv",
                        help="Output directory for CSV files (default: ./csv)")
    parser.add_argument("--sheets", "-s", default=None,
                        help="Comma-separated list of sheet names to convert (default: all sheets)")
    parser.add_argument("--write-empty", action="store_true",
                        help="Write a CSV even if the sheet has no non-empty rows/columns")
    parser.add_argument("--same-name", action="store_true",
                        help="When converting a workbook with exactly one non-empty sheet, also write the CSV with the same base name as the workbook")

    args = parser.parse_args()

    excel_path = Path(args.excel)

    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        sys.exit(2)

    # If --same-name requested, write outputs into the same folder as the Excel file
    if args.same_name:
        out_dir = excel_path.parent
    else:
        out_dir = Path(args.outdir)

    print(f"Reading: {excel_path}")
    print(f"Output directory: {out_dir}")

    try:
        result = convert_excel_to_csv(excel_path, out_dir, sheets=args.sheets, write_empty=args.write_empty)
    except Exception as exc:
        print("Error converting workbook:", exc)
        sys.exit(3)

    # If requested, and exactly one CSV written, also write a copy named after the workbook base
    if args.same_name and len(result.get("written", [])) == 1:
        import shutil
        src = Path(result["written"][0])
        dst = out_dir / f"{excel_path.stem}.csv"
        shutil.copyfile(src, dst)
        # Add to summary so it is listed below
        result["written"].append(str(dst))
        print(f"Also wrote single-sheet CSV with workbook base name: {dst}")

    print("Summary:")
    print("  Written:")
    for p in result["written"]:
        print("    ", p)
    print("  Skipped (empty):")
    for s in result["skipped"]:
        print("    ", s)


if __name__ == "__main__":
    main()
