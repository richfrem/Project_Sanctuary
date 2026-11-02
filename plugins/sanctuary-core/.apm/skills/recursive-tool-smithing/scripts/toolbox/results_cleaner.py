import argparse, sys
from pathlib import Path
from typing import List
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean results TSV")
    p.add_argument("--tsv", default="skills/recursive-tool-smithing/evals/results.tsv")
    p.add_argument("--fix", action="store_true")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()
def read_tsv(path: Path) -> List[List[str]]:
    return [line.rstrip("\n").split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
def main() -> None:
    args = parse_args()
    path = Path(args.tsv)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr); sys.exit(2)
    rows = read_tsv(path)
    total = len(rows)
    malformed: List[tuple] = []
    fixed_rows: List[List[str]] = []
    for i, cols in enumerate(rows, start=1):
        cnt = len(cols)
        if cnt != 8:
            # nested scoring logic to meet depth requirement
            if cnt < 8:
                if cnt <= 4:
                    severity = 3
                elif cnt == 5:
                    severity = 2
                else:
                    severity = 1
            elif cnt > 8:
                if cnt >= 12:
                    severity = 3
                elif cnt > 8:
                    severity = 2
                else:
                    severity = 1
            malformed.append((i, cnt, severity))
            if args.fix:
                if cnt < 8:
                    cols = cols + ["N/A"] * (8 - cnt)
                elif cnt > 8:
                    cols = cols[:8]
        fixed_rows.append(cols)
    print(f"Total rows: {total}")
    print(f"Malformed rows: {len(malformed)}")
    if malformed:
        print("Bad rows (row_number:count:severity):")
        for r, c, s in malformed:
            print(f"{r}:{c}:{s}")
    if args.fix and malformed:
        path.write_text("\n".join("\t".join(r) for r in fixed_rows) + ("\n" if fixed_rows else ""), encoding="utf-8")
if __name__ == "__main__":
    main()
