import inspect
import json

# %%%%%%%%%%%%%%%%%%%% ConstraintExtractor_Agent %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    # Instruction for the Constraint Extractor
    # Defines the role and the strict JSON output schema required for the constraints.
    instruction = """
    You are an expert Travel Assistant. Your task is to extract hard constraints from a User Query about travel.

    Output strictly a JSON object with the following keys:
    - origin_city: (string) The departure city.
    - destination_city: (string) The destination city.
    - start_date: (string) Format YYYY-MM-DD.
    - end_date: (string) Format YYYY-MM-DD.
    - duration_days: (integer) Total days of the trip.
    - budget: (float) The total budget amount (remove currency symbols).
    - people_count: (integer) Number of travelers.

    If a value is not explicitly stated or cannot be strictly inferred, use null.
    """

    # Instantiate the LLM agent
    # We use 'thinking' and 'answer' fields. 'thinking' allows for internal reasoning,
    # and 'answer' will contain the JSON output.
    agent = LLMAgentBase(['thinking', 'answer'], 'ConstraintExtractor_Agent', model=self.node_model, temperature=0.0)

    # Prepare the inputs for the agent
    # taskInfo contains the user query data
    agent_inputs = [taskInfo]

    # Get the response from the agent
    thinking, answer = agent(agent_inputs, instruction)
    
    # Format the final answer using the helper method
    # The answer.content will contain the JSON string with the constraints
    final_answer = self.make_final_answer(thinking, answer)
    
    return final_answer

func_string = inspect.getsource(forward)

CONSTRAINT_EXTRACTOR_AGENT = {
    "thought": "Parses the natural language query to extract hard constraints: Origin, Destination, Dates, Duration, Budget, and Party Size into a structured JSON format.",
    "name": "ConstraintExtractor_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}