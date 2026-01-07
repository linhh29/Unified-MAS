# PMC Implementation Notes

## Overview

This is an implementation of the PMC (Planning with Multi-Constraints via Collaborative Language Agents) method described in the COLING 2025 paper. Since the original code is not publicly available, this implementation is based on:

1. Paper description from web search results
2. Common patterns in collaborative agent systems
3. Existing codebase patterns (TravelPlanner, MAS-Zero)
4. Best practices for multi-agent planning systems

## Key Components

### 1. Manager Agent (`ManagerAgent`)
- **Role**: Task decomposition
- **Functionality**: 
  - Analyzes the task and constraints
  - Breaks down complex tasks into manageable subtasks
  - Identifies dependencies between subtasks
  - Assigns constraint types to each subtask
- **Implementation**: Uses GPT-4 with low temperature (0.3) for consistent decomposition

### 2. Executor Agents (`ExecutorAgent`)
- **Role**: Execute specific subtasks
- **Functionality**:
  - Each executor specializes in a constraint type
  - Processes subtasks based on their constraint types
  - Uses results from dependent subtasks
  - Provides detailed solutions for their assigned subtasks
- **Implementation**: Uses GPT-4 with moderate temperature (0.5) for creative solutions

### 3. PMC Agent (`PMCAgent`)
- **Role**: Coordinate the entire planning process
- **Functionality**:
  - Initializes manager and executor agents
  - Manages task decomposition
  - Coordinates subtask execution (topological sort)
  - Integrates results from all subtasks
  - Verifies constraint satisfaction
- **Implementation**: Orchestrates the collaborative agent system

## Architecture

```
Input: Task + Constraints
    |
    v
Manager Agent (Decomposition)
    |
    v
Subtask List (with dependencies)
    |
    v
Topological Sort (Execution Order)
    |
    v
Executor Agents (Parallel/Sequential)
    |
    v
Subtask Results
    |
    v
Integration Agent
    |
    v
Final Plan
    |
    v
Verification Agent
    |
    v
Output: Plan + Verification Status
```

## Constraint Types

The implementation supports six constraint types:

1. **Budget**: Financial constraints (e.g., total budget limit)
2. **Temporal**: Time-related constraints (dates, durations, schedules)
3. **Spatial**: Location and transportation constraints
4. **Preference**: User preferences (food, accommodation, activities)
5. **Hard**: Must-satisfy constraints (e.g., no flights, pet-friendly)
6. **Commonsense**: Realistic planning constraints (logical routes, timing)

## Key Design Decisions

### 1. Task Decomposition
- Uses LLM-based decomposition with structured output (JSON)
- Falls back to default decomposition if parsing fails
- Ensures dependencies are properly identified

### 2. Subtask Execution
- Uses topological sort to handle dependencies
- Multiple executors can work on the same subtask (if multiple constraint types)
- Results from dependent subtasks are passed as context

### 3. Result Integration
- Dedicated integration agent combines all subtask results
- Ensures coherence and resolves conflicts
- Produces a single, actionable final plan

### 4. Constraint Verification
- Separate verification step checks constraint satisfaction
- Can be used for iterative refinement if needed

## Differences from Original Implementation

Since the original code is not available, this implementation makes several assumptions:

1. **Agent Architecture**: Assumes manager-executor pattern (common in multi-agent systems)
2. **Constraint Handling**: Implements constraint types based on TravelPlanner dataset
3. **Decomposition Strategy**: Uses LLM-based decomposition (common in recent work)
4. **Integration Method**: Simple integration agent (could be enhanced with more sophisticated methods)

## Potential Improvements

1. **Iterative Refinement**: Add feedback loops for constraint violations
2. **Multi-round Execution**: Allow executors to refine their solutions
3. **Constraint Propagation**: Better handling of constraint interactions
4. **Specialized Executors**: More domain-specific executor agents
5. **Plan Optimization**: Post-processing to optimize plans

## Evaluation

The implementation includes:
- Integration with TravelPlanner dataset
- Constraint extraction from queries
- Result formatting for evaluation
- Evaluation script (`evaluate_pmc.py`)

## Usage

See `example.py` for a complete example of using PMC for travel planning.

## Limitations

1. **API Dependencies**: Requires OpenAI API access
2. **Cost**: Multiple LLM calls can be expensive
3. **Latency**: Sequential execution may be slow for complex tasks
4. **Error Handling**: Basic error handling (could be improved)

## Future Work

1. Support for other LLM providers (Anthropic, local models)
2. Caching and optimization to reduce API calls
3. Parallel execution where possible
4. Better constraint extraction and validation
5. Integration with more planning benchmarks

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{pmc2025coling,
  title={Planning with Multi-Constraints via Collaborative Language Agents},
  author={...},
  booktitle={Proceedings of COLING 2025},
  year={2025}
}
```

## Contact

For questions or issues with this implementation, please refer to the original paper authors or open an issue in the repository.

