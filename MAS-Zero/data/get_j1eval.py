import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def main() -> None:
    raw_dataset = load_dataset("json", data_files="/data/qin/lhh/Unified-MAS/MAS-Zero/data/J1-Eval_CI.jsonl")['train']
    
    print(f"Raw dataset: {raw_dataset}")
    dataset = raw_dataset.flatten_indices()  # ensure a deterministic order

    total_examples = len(dataset)
    print(f"Total examples: {total_examples}")
    validate_count = 16
    if total_examples < validate_count:
        raise ValueError(
            f"Expected at least {validate_count} examples, got {total_examples}."
        )

    validate_dataset = dataset.select(range(validate_count))
    test_dataset = dataset.select(range(validate_count, total_examples))

    output_dir = Path(__file__).resolve().parent
    validate_path = output_dir / "src/j1eval_validate.jsonl"
    test_path = output_dir / "src/j1eval_test.jsonl"

    # validate_dataset.to_json(validate_path)
    # test_dataset.to_json(test_path)
    validate_dataset.to_json(validate_path, force_ascii=False)
    test_dataset.to_json(test_path, force_ascii=False)

    print(f"Wrote validate set to {validate_path}")
    print(f"Wrote test set to {test_path}")


if __name__ == "__main__":
    # main()
    # Here
    output_dir = Path(__file__).resolve().parent
    validate_path = output_dir / "src/j1eval_validate.jsonl"
    test_path = output_dir / "src/j1eval_test.jsonl"

    def load_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    validate_records = load_jsonl(validate_path)
    test_records = load_jsonl(test_path)

    print(f"Validate set size: {len(validate_records)}")
    print(f"Validate set size: {type(validate_records[0])}")
    print(f"Validate first record keys: {sorted(validate_records[0].keys()) if validate_records else []}")

    print(f"Test set size: {len(test_records)}")
    print(f"Validate set size: {type(test_records[0])}")
    print(f"Test first record keys: {sorted(test_records[0].keys()) if test_records else []}")

    print(validate_records[0])