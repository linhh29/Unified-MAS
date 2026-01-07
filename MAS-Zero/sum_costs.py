import argparse
from pathlib import Path


def sum_costs(file_path: str) -> float:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    total = 0.0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            try:
                value_str = line.split(":", 1)[1].strip()
                total += float(value_str)
            except Exception:
                # Skip lines that don't match the expected pattern
                continue
    return total


def sum_score(file_path: str) -> float:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    total = 0.0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if "SCORE" not in line:
                continue
            try:
                value_str = line.split("SCORE:", 1)[1].strip()
                total += float(value_str)
                count += 1
            except Exception:
                continue
    return total / count if count > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='/data/qin/lhh/Unified-MAS/MAS-Zero/async_results_unified_gemini-3-flash-preview/question/meta_agent/workflow_search/travelplanner/gemini-3-flash-preview_gemini-3-flash-preview_oracle.results_cost.txt', help="Path to the results_cost.txt file")
    parser.add_argument("--score-file", type=str, default='/data/qin/lhh/Unified-MAS/MAS-Zero/async_results_unified_gemini-3-flash-preview/question/meta_agent/workflow_search/travelplanner/gemini-3-flash-preview_gemini-3-flash-preview_self.results_5', help="Path to the results file containing SCORE values")
    args = parser.parse_args()

    total = sum_costs(args.file)
    print(f"Total cost: {total}")

    if args.score_file:
        average = sum_score(args.score_file)
        print(f"Average SCORE: {average}")


if __name__ == "__main__":
    main()


