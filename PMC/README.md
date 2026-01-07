# PMC: Planning with Multi-Constraints via Collaborative Language Agents

Implementation of the PMC method from the paper "Planning with Multi-Constraints via Collaborative Language Agents" at COLING 2025.

## Overview

PMC is a zero-shot method for planning with multiple constraints using collaborative language agents. The system consists of:

1. **Manager Agent**: Responsible for task decomposition
2. **Executor Agents**: Handle specific constraints and execute subtasks
3. **Collaborative Planning System**: Coordinates agents to solve multi-constraint problems

## Architecture

```
Task + Constraints
    ↓
Manager Agent (Task Decomposition)
    ↓
Subtask 1 ──┐
Subtask 2 ──┤
Subtask 3 ──┼──> Executor Agents (Parallel/Sequential Execution)
Subtask 4 ──┤
Subtask 5 ──┘
    ↓
Integration Agent (Plan Integration)
    ↓
Verification Agent (Constraint Verification)
    ↓
Final Plan
```

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
import asyncio
from pmc_agent import PMCAgent

async def main():
    # Initialize PMC agent
    pmc = PMCAgent(model_name="gpt-4", api_key="your_api_key")
    
    # Define task and constraints
    task = "Create a travel plan with specific requirements..."
    constraints = {
        "budget": 2900,
        "temporal": {"start_date": "2022-03-16", "end_date": "2022-03-20"},
        "spatial": {"origin": "Baton Rouge", "destination_count": 2},
        "preference": {"transportation": "no_flights", "pet_friendly": True}
    }
    
    # Execute planning
    plan = await pmc.plan(task, constraints)
    
    # Access results
    print(plan.final_plan)
    print(f"Constraints satisfied: {plan.constraints_satisfied}")

asyncio.run(main())
```

### Integration with TravelPlanner

The PMC agent can be integrated with TravelPlanner dataset for evaluation:

```python
from pmc_agent import PMCAgent
from datasets import load_dataset

# Load TravelPlanner dataset
data = load_dataset('osunlp/TravelPlanner', 'validation')['validation']

# Initialize PMC
pmc = PMCAgent()

# Process a sample
sample = data[0]
task = sample['query']
constraints = extract_constraints(sample)  # Extract from sample
data_info = sample['ref_info']  # Reference information

plan = await pmc.plan(task, constraints, data=data_info)
```

## Constraint Types

PMC handles multiple constraint types:

- **Budget**: Financial constraints
- **Temporal**: Time-related constraints (dates, durations)
- **Spatial**: Location and transportation constraints
- **Preference**: User preferences (food, accommodation, etc.)
- **Hard**: Must-satisfy constraints
- **Commonsense**: Realistic planning constraints

## Components

### ManagerAgent
- Decomposes complex tasks into manageable subtasks
- Identifies dependencies between subtasks
- Assigns constraint types to each subtask

### ExecutorAgent
- Executes specific subtasks
- Specialized in handling particular constraint types
- Uses results from dependent subtasks

### PMCAgent
- Coordinates manager and executor agents
- Manages execution order (topological sort)
- Integrates results into final plan
- Verifies constraint satisfaction

## Evaluation

To evaluate on TravelPlanner:

```bash
python evaluate_pmc.py --dataset validation --output_dir ./results
```

## Citation

```bibtex
@inproceedings{pmc2025coling,
  title={Planning with Multi-Constraints via Collaborative Language Agents},
  author={...},
  booktitle={Proceedings of COLING 2025},
  year={2025}
}
```

## Implementation Details

### Task Decomposition
The Manager Agent uses chain-of-thought reasoning to decompose complex tasks into subtasks:
- Each subtask addresses specific constraints
- Dependencies between subtasks are identified
- Subtasks are ordered topologically for execution

### Constraint Handling
PMC handles six types of constraints:
1. **Budget**: Financial limitations
2. **Temporal**: Time-related (dates, durations)
3. **Spatial**: Location and transportation
4. **Preference**: User preferences
5. **Hard**: Must-satisfy constraints
6. **Commonsense**: Realistic planning requirements

### Execution Flow
1. Manager decomposes task → List of Subtasks
2. Subtasks executed in dependency order
3. Executor agents handle specific constraint types
4. Results integrated into final plan
5. Constraints verified

### Agent Collaboration
- Multiple executor agents can work on the same subtask (if it has multiple constraint types)
- Results are combined intelligently
- Integration agent ensures coherence
- Verification agent checks constraint satisfaction

## Example Output

```
Step 1: Manager agent decomposing task...
Task decomposed into 5 subtasks

Step 2.1: Executing subtask 1: Extract and understand all constraints...
  Using executor agent: hard_0
  Subtask 1 completed

Step 2.2: Executing subtask 2: Plan transportation...
  Using executor agent: temporal_0
  Using executor agent: spatial_0
  Subtask 2 completed

...

Step 3: Integrating results into final plan...

FINAL PLAN
[Detailed travel plan with all constraints satisfied]
```

## Notes

- This implementation is based on the paper description and common patterns in collaborative agent systems
- The original code is not publicly available due to privacy policy
- This implementation provides a functional baseline for multi-constraint planning with collaborative agents
- The implementation uses OpenAI's GPT models, but can be adapted for other LLM providers

## License

This implementation is provided for research purposes. Please refer to the original paper for citation information.
