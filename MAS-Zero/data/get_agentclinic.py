import json
import random
from pathlib import Path


def main() -> None:
    # Set random seed for reproducibility
    random.seed(0)
    
    # Input file path
    input_file = Path("/data/qin/lhh/Unified-MAS/AgentClinic/agentclinic_medqa_extended.jsonl")
    
    # Load all records from jsonl file
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    print(f"Total records loaded: {len(records)}")
    
    # Shuffle the list with seed 0
    random.shuffle(records)
    
    # Split into validate and test sets
    validate_count = 32
    if len(records) < validate_count:
        raise ValueError(
            f"Expected at least {validate_count} examples, got {len(records)}."
        )
    
    validate_records = records[:validate_count]
    test_records = records[validate_count:]
    
    # Output directory (src folder)
    output_dir = Path(__file__).resolve().parent / "src"
    output_dir.mkdir(exist_ok=True)
    
    validate_path = output_dir / "agentclinic_medqa_validate.jsonl"
    test_path = output_dir / "agentclinic_medqa_test.jsonl"
    
    # Write validate set
    with open(validate_path, "w", encoding="utf-8") as f:
        for record in validate_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Write test set
    with open(test_path, "w", encoding="utf-8") as f:
        for record in test_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Wrote validate set ({len(validate_records)} records) to {validate_path}")
    print(f"Wrote test set ({len(test_records)} records) to {test_path}")


if __name__ == "__main__":
    main()

