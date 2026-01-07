import json
import os
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from openai import OpenAI


def convert_plan_to_natural_language(plan_list_str: str, api_key: str = 'sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2', base_url: str = 'https://api.qingyuntop.top/v1', model: str = "gpt-4o") -> str:
    """
    Convert a travel plan from list format to natural language text using LLM.
    
    Args:
        plan_list_str: The travel plan in list format (as string)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        base_url: OpenAI API base URL (defaults to OPENAI_API_BASE env var)
        model: Model name to use
        
    Returns:
        Natural language description of the travel plan
    """
    # Initialize OpenAI client
    key = api_key or os.getenv("OPENAI_API_KEY")
    base = base_url or os.getenv("OPENAI_API_BASE")
    
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is required for plan conversion.")
    
    client = OpenAI(api_key=key, base_url=base)
    
    # Create prompt for conversion
    system_prompt = "You are a helpful assistant that converts structured travel plans into natural, readable text descriptions."
    
    user_prompt = f"""Please convert the following travel plan from list format to a natural, readable text description. 
Write it in a clear and engaging way, describing each day's activities, transportation, meals, attractions, and accommodations.

Travel Plan (list format):
{plan_list_str}

Please provide a natural language description of this travel plan:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Failed to convert plan to natural language: {e}")
        # Return original format if conversion fails
        return plan_list_str


def convert_messages_to_dataset_format(messages_record: dict, convert_plan: bool = True) -> dict:
    """
    Convert messages format to dataset format.
    
    Args:
        messages_record: A record in messages format with 'messages' key
        
    Returns:
        A record in dataset format with only query, reference_information, and label fields
    """
    messages = messages_record.get("messages", [])
    
    # Extract user and assistant messages
    user_content = None
    assistant_content = None
    
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
        elif msg.get("role") == "assistant":
            assistant_content = msg.get("content", "")
    
    if not user_content:
        raise ValueError("No user message found in record")
    if not assistant_content:
        raise ValueError("No assistant message found in record")
    
    # Extract query and reference_information from user content
    # Format: "Given Information:...\nQuery:..."
    query_match = re.search(r"Query:(.+?)(?:\n|$)", user_content, re.DOTALL)
    if not query_match:
        raise ValueError("Could not extract query from user content")
    query = query_match.group(1).strip()
    
    # Extract reference_information (everything between "Given Information:" and "Query:")
    info_match = re.search(r"Given Information:(.+?)Query:", user_content, re.DOTALL)
    if not info_match:
        raise ValueError("Could not extract reference_information from user content")
    reference_information = info_match.group(1).strip()
    
    # Convert plan from list format to natural language if requested
    answer = assistant_content
    # if convert_plan:
    #     try:
    #         answer = convert_plan_to_natural_language(assistant_content)
    #     except Exception as e:
    #         print(f"Warning: Failed to convert plan format, using original: {e}")
    #         answer = assistant_content
    
    # Build dataset format record with only the three required fields
    dataset_record = {
        "query": query,
        "reference_information": reference_information,
        "answer": answer  # The assistant's response (final travel plan in natural language)
    }
    
    return dataset_record


def main() -> None:
    # Step 1: Read validation set from local JSONL file
    validation_jsonl_path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/travel_planner.jsonl")
    
    if not validation_jsonl_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_jsonl_path}")
    
    print(f"Reading validation set from: {validation_jsonl_path}")
    validation_records = []
    with open(validation_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                validation_records.append(json.loads(line))
    
    print(f"Loaded {len(validation_records)} examples from validation file")
    
    # Step 2: Load raw_dataset from HuggingFace and save all as test set
    print("Loading raw dataset from HuggingFace...")
    raw_dataset = load_dataset("osunlp/TravelPlanner", "validation")
    if isinstance(raw_dataset, DatasetDict):
        if "validation" not in raw_dataset:
            raise ValueError("Validation split not found in dataset.")
        dataset = raw_dataset["validation"]
    elif isinstance(raw_dataset, Dataset):
        dataset = raw_dataset
    else:
        raise TypeError("Unexpected dataset type returned by load_dataset().")

    dataset = dataset.flatten_indices()  # ensure a deterministic order
    total_examples = len(dataset)
    print(f"Loaded {total_examples} examples from HuggingFace dataset")
    
    # Step 3: Save datasets
    output_dir = Path(__file__).resolve().parent / "src"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validate_path = output_dir / "travelplanner_validate.jsonl"
    test_path = output_dir / "travelplanner_test.jsonl"
    
    # Save validation set from local file (convert format first)
    print(f"\nConverting validation records to dataset format...")
    print("This will convert travel plans from list format to natural language using LLM...")
    converted_validation_records = []
    for i, record in enumerate(validation_records):
        try:
            print(f"Processing record {i+1}/{len(validation_records)}...", end="\r")
            converted_record = convert_messages_to_dataset_format(record)
            converted_validation_records.append(converted_record)

            if i == 0:
                print(f"\nFirst converted record keys: {list(converted_record.keys())}")
                print(f"First record answer preview (first 200 chars): {converted_record['answer']}...")
        except Exception as e:
            print(f"\nWarning: Failed to convert record {i}: {e}")
            continue
    
    print(f"\nSuccessfully converted {len(converted_validation_records)}/{len(validation_records)} records")
    
    print(f"\nWriting validation set to: {validate_path}")
    with open(validate_path, 'w', encoding='utf-8') as f:
        for record in converted_validation_records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"Wrote {len(converted_validation_records)} examples to validation set")
    
    # Save all raw_dataset as test set
    print(f"\nWriting test set to: {test_path}")
    dataset.to_json(test_path)
    print(f"Wrote {total_examples} examples to test set")
    
    print(f"\n✓ Complete!")
    print(f"  Validation set: {validate_path} ({len(converted_validation_records)} examples)")
    print(f"  Test set: {test_path} ({total_examples} examples)")


if __name__ == "__main__":
    main()