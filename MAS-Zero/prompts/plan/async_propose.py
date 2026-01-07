
import json
from async_code_archive import util_code, wrong_implementation

background_hosp_summ = """
To address the task of summarizing hospitalization records into comprehensive discharge summaries, it's essential to understand several key concepts and follow a structured approach. Here's a breakdown of the necessary background knowledge and a typical workflow for solving such a problem:

### Key Background Knowledge and Concepts

1. **Hospitalization Records:**
   - Hospitalization records contain comprehensive patient information including admission details, history of present illness, physical examination findings, laboratory results, imaging studies, medications, procedures, and discharge instructions.

2. **Discharge Summary Structure:**
   - A discharge summary is a critical document that synthesizes a patient's hospital stay into a concise, comprehensive summary. It typically includes: chief complaint, history of present illness, hospital course, procedures performed, medications, discharge diagnosis, discharge condition, and follow-up instructions.

3. **Clinical Information Extraction:**
   - The ability to identify and extract key diagnostic information, significant findings, and important clinical events from lengthy medical records is crucial for creating accurate summaries.

4. **Medical Terminology:**
   - Understanding medical terminology, abbreviations, and clinical concepts is necessary to accurately interpret and summarize medical records.

5. **Information Prioritization:**
   - Not all information in a hospitalization record is equally important. The ability to prioritize and include the most relevant diagnostic and clinical information is essential.

6. **Synthesis and Coherence:**
   - Creating a coherent narrative that flows logically from admission through discharge, connecting various pieces of information into a comprehensive summary.

### Typical Solution Process / Workflow

1. **Review Complete Record:**
   - Read through the entire hospitalization record to understand the full context of the patient's stay, including admission reason, course of treatment, and outcomes.

2. **Identify Key Information:**
   - Extract key diagnostic information, significant test results, procedures performed, complications, and important clinical decisions made during the hospitalization.

3. **Organize Information:**
   - Structure the information logically: admission details, hospital course, procedures, medications, discharge diagnosis, and follow-up instructions.

4. **Synthesize Content:**
   - Combine and synthesize information from different sections (physician notes, nursing notes, medications, discharge instructions) into a coherent summary.

5. **Ensure Completeness:**
   - Verify that all critical information is included: primary diagnosis, significant findings, procedures, discharge medications, and follow-up requirements.

6. **Maintain Accuracy:**
   - Ensure that the summary accurately reflects the information in the original record without adding or omitting critical details.

7. **Review and Refine:**
   - Review the summary for clarity, completeness, and accuracy. Ensure it provides a comprehensive overview of the hospitalization.

By following this structured approach, one can systematically analyze hospitalization records and create comprehensive discharge summaries that accurately capture the key diagnostic information and significant results from a patient's hospital stay.
"""

background_travelplanner = """
To create a travel itinerary for a solo traveler based on the provided JSONL samples, it's important to understand several key concepts and follow a structured approach. Here's a breakdown of the necessary background knowledge and a typical solution process:

### Key Background Knowledge and Concepts

1. **Travel Planning Basics**:
   - **Transportation**: Understanding different modes of travel (flights, trains, buses, taxis, self-driving) and how to choose the most cost-effective and time-efficient options.
   - **Accommodation**: Familiarity with various types of accommodations (hotels, hostels, Airbnb, home swaps) and how to select them based on budget and preferences.
   - **Attractions and Dining**: Knowledge of how to research and select attractions and dining options that align with the traveler's interests and budget.

2. **Budget Management**:
   - Ability to calculate and manage costs for transportation, accommodation, dining, and attractions to ensure the trip stays within the specified budget.

3. **Geographical and Cultural Knowledge**:
   - Understanding the logistics of traveling between cities and the cultural and culinary preferences that might influence itinerary choices.

4. **Constraints Handling**:
   - Ability to incorporate and respect specified local constraints, such as house rules, cuisine preferences, room types, and transportation preferences.

5. **Tools and Resources**:
   - Familiarity with travel planning tools like Google Maps, Rome2Rio, Hopper, and travel communities for logistical planning and cost-saving tips.

### Typical Solution Process / Workflow

1. **Define Trip Parameters**:
   - Extract and understand the key parameters from the JSONL samples, including origin, destination, number of days, cities to visit, travel dates, budget, and local constraints.

2. **Research and Planning**:
   - Use travel planning tools to research transportation options, accommodation, attractions, and dining that fit the traveler's preferences and budget.
   - Consider using tools like Rome2Rio for transportation logistics and Hopper for flight deals.

3. **Create a Draft Itinerary**:
   - Outline a day-by-day plan that includes transportation, accommodation, attractions, and dining options.
   - Ensure the itinerary aligns with the traveler's interests and constraints, allowing for flexibility and downtime.

4. **Budget Allocation**:
   - Allocate the budget across different categories (transportation, accommodation, dining, attractions) and ensure the total cost does not exceed the specified budget.
   - Consider cost-saving strategies like traveling during shoulder seasons or using home swaps for accommodation.

5. **Iterate and Refine**:
   - Review the draft itinerary to ensure it is realistic and feasible, making adjustments as needed to accommodate travel logistics and budget constraints.
   - Incorporate feedback from travel communities or use AI tools to refine the itinerary further.

6. **Finalize the Itinerary**:
   - Ensure all bookings and reservations are made, and the itinerary is shared with the traveler for review and final adjustments.
   - Provide additional tips for safety, packing, and cultural considerations to enhance the travel experience.

By following this structured approach, one can create a coherent and feasible travel plan that meets the solo traveler's requirements and preferences.
"""

background_j1eval = """
To address the task of analyzing legal case data related to personality rights disputes, contract disputes, and marriage and family issues, it's essential to understand the relevant legal concepts and the typical process for analyzing such cases.

### Key Background Knowledge and Concepts

1. **Personality Rights**:
   - **Definition**: Personality rights, also known as the right of publicity, allow individuals to control the commercial use of their identity, including their name, image, and likeness (Source 4).
   - **Legal Framework**: These rights are often considered property rights and can extend beyond an individual's death, depending on jurisdiction. They are divided into the right of publicity (preventing unauthorized commercial exploitation) and the right to privacy (protecting individuals from unauthorized public representation) (Source 4, 6).
   - **Jurisdictional Variations**: Different jurisdictions have varying approaches to personality rights, with common law jurisdictions focusing on torts like passing off, and civil law jurisdictions having specific legal provisions (Source 4).

2. **Contract Disputes**:
   - **Definition**: Contract disputes arise when there is a disagreement over the terms, execution, or validity of a contract. These disputes often involve issues like breach of contract, misrepresentation, or non-performance.
   - **Legal Principles**: Key principles include offer, acceptance, consideration, and mutual consent. Courts often look at the intent of the parties and the terms of the contract to resolve disputes.

3. **Marriage and Family Issues**:
   - **Definition**: These disputes involve legal issues related to marriage, divorce, child custody, and other family matters.
   - **Legal Considerations**: Family law often involves considerations of the best interests of children, equitable distribution of assets, and spousal support.

### Typical Solution Process / Workflow

1. **Understanding Legal Context**:
   - Familiarize yourself with the specific legal terminology and principles relevant to the type of dispute (personality rights, contract law, family law).
   - Review applicable laws and precedents that may influence the case outcome.

2. **Analyzing Evidence**:
   - Examine the evidence provided by both parties, including documents, receipts, and other materials.
   - Assess the validity and relevance of the evidence, considering any disputes over its authenticity.

3. **Assessing Claims and Defenses**:
   - Evaluate the plaintiff's claims and the defendant's defenses, focusing on the credibility and consistency of each party's narrative.
   - Consider any counterarguments or alternative narratives presented by the defendant.

4. **Applying Legal Principles**:
   - Apply relevant legal principles and precedents to the facts of the case.
   - Consider how courts have previously ruled in similar cases and the reasoning behind those decisions.

5. **Judgment Prediction**:
   - Based on the analysis, predict the likely judgment, including any compensation awarded or actions mandated.
   - Consider the balance between protecting individual rights and upholding public interest, especially in personality rights cases (Source 2, 3).

By following this structured approach, one can systematically analyze legal case data and make informed predictions about potential judgments. Understanding the nuances of each type of dispute and the legal frameworks involved is crucial for accurate analysis and prediction.
"""


EXAMPLE = {
    "thought": """
    **Insights:**\nProvide your reasoning for the next effective agent architecture (an architecture may contain multiple agents), along with an explanation of the overall concept behind the design. 
    **Decomposition:**\n Give the resulting new sub-task 1, sub-task 2, ..., sub-task n. Your final decomposition should include all the sub-tasks. 
    Please explain 
    (1) How do you make sure the sub-task is easier enough and thus solvable by the given or proposed blocks and agents; 
    (2) Justify how these sub-tasks can achieve the final answer to the original questions.

    **Overall Architecture to solve each subquestion:**
    You overall architecture and explain how this architecture can solve each of the resulting sub-task
    "**Implementation:**describe the implementation step by step."
    """,
    "name": "Name of your proposed architecture",

    "code": """async def forward(self, taskInfo, extra_info):
    from collections import Counter # must have this and always make sure you import everything needed
    # Your code here. IMPORTANT  
    # (1) You cannot call the existing architecture from the archive but you have to implement it from the code in the Discovered architecture archive
    # for example:
    # you cannot call 'COT' but you have to implement it exactly the same as the code in the Discovered architecture archive. 
    # name an agent 'reflexion' or 'debate' or 'cot_sc' also do not implement the blocks. Make sure you ACTUALLY IMPLEMENT them
    # for example, You need to implement the for-loop in COT-SC, LLM Debate and Reflexion
    # You should only change how they connect but not the function inside (setting and instruction can be different)

    # (2) To creat an agent, call the LLMAgentBase, must have 'model=self.node_model' in the parameters
    # the return of the call is always the same as the output fields,
    # for example: 
    # reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent', model=self.node_model)

    # (3) Since the agent is handling with a sub_question made by question decomposition, use 'is_sub_task=True' when calling the agent.

    # (4) You need to specify the sub-task dependencies by specify the dependent sub-tasks in the instruction. You also need to give each sub-task an ID. This means you need to state clearly what the current sub-task is based on and also give all required information (thinking and answer) in the input task info
    # for example:
    # cot_instruction = ("Sub-task 3: Based on the output of sub-task 1 and sub-task 2.....")
    # thinking, answer = await reasoning_agent([taskInfo] + [thinking1, answer1, thinking2, answer2, ..., ], extra_info, cot_instruction, is_sub_task=True)


    # (5) You need to keep tack of sub-task output, for each sub-task output, please append it to a list named `sub_tasks` (please initialize it as `sub_tasks = []` at the begining of your function) so that we can keep track of the performance of each sub-task. When you do self.make_final_answer, please also add `sub_tasks` as the second last parameterss 
    # for example: 
    # sub_tasks = []
    # ...
    # thinking, answer = await reasoning_agent([taskInfo] + [sub-task thinking, sub-task answer], reasoning_instruction, is_sub_task=True)
    # sub_tasks.append(f'the output of sub task n: thinking {thinking.content}; final answer: {answer.content}')

    # (6) You need to keep track of agent, for each agent inside the sub-architecture (or block or node) output, please append it to a list named `agents` (please initialize it as `agents = []` at the beining of your function) so that we can keep track of the performance of each agents. It should contain the agent name (if setting is important to identify the agent, please include as well, e.g., round, ID), agent purpose, and the thinking and output of the agent. When you do self.make_final_answer, please also add `agents` as the last parameters
    # Takes debate as an example: 
    # debate_instruction = ("Sub-task i: Based on the output of...")
    # max_round = ...(the max round you determine)
    # debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in ...(the role you design)]
    # all_thinking = []
    # all_answer = []
    # for r in range(max_round):
    #     round_thinking = []
    #     round_answer = []
    #     for i, agent in enumerate(debate_agents):
    #         if r == 0:
    #             t, a = await agent([taskInfo, thinking1, answer1], extra_info, cot_instruction, is_sub_task=True)
    #             agents.append(f'Debate agent {agent.id}, round {r}, on the purpose of..., thinking: {t.content}; answer: {a.content}')
    #         else:
    #             t, a = await agent([taskInfo, thinking1, answer1] + all_thinking[r-1], extra_info, debate_instruction, is_sub_task=True)
    #             agents.append(f'Debate agent {agent.id}, round {r}, on the purpose of..., thinking: {t.content}; answer: {a.content}')
    #         round_thinking.append(t)
    #         round_answer.append(a)
    #     all_thinking.append(round_thinking)
    #     all_answer.append(round_answer)
    # final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    # thinking2, answer2 = await final_decision_agent(
    #     [taskInfo] + all_thinking[-1] + all_answer[-1],
    #     extra_info,
    #     "Sub-task i: Based on the output of...",
    #     is_sub_task=True
    # )
    # agents.append(f'Final Decision agent, on the purpose of..., thinking: {thinking2.content}; answer: {answer2.content}')
    # sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")

    # Take reflexion as another example:
    # cot_reflect_instruction = "Sub-task i: Based on the output of..."
    # cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)

    # # Instruction for providing feedback and correcting the answer
    # critic_instruction = "Sub-task i: Based on the output of...,"
    # critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    # N_max = ...(the max round you determine) # Maximum number of attempts
    # # Initial attempt
    # cot_inputs = [taskInfo]
    # thinking, answer = await cot_agent(cot_inputs, extra_info, cot_initial_instruction, 0, is_sub_task=True)
    # agents.append(f'CoT agent {cot_agent.id}, on the purpose of..., thinking: {thinking.content}; answer: {answer.content}')

    # for i in range(N_max):
    #     # Get feedback and correct status from the critic
    #     feedback, correct = await critic_agent([taskInfo, thinking, answer], extra_info, critic_instruction, i, is_sub_task=True)
    #     agents.append(f'Critic agent {critic_agent.id}, on the purpose of..., thinking: {feedback.content}; answer: {correct.content}')
    #     if correct.content == 'True':
    #         break
            
    #     # Add feedback to the inputs for the next iteration
    #     cot_inputs.extend([thinking, answer, feedback])

    #     # Reflect on previous attempts and refine the answer
    #     thinking, answer = await cot_agent(cot_inputs, extra_info, cot_reflect_instruction, i + 1, is_sub_task=True)
    #     agents.append(f'CoT agent {cot_agent.id}, on the purpose of..., thinking: {thinking.content}; answer: {answer.content}')
    # sub_tasks.append(f"Sub-task i output: thinking - {thinking.content}; answer - {answer.content}")

    # Take self-consistency as another example:
    # cot_instruction = 'Sub-task i: Based on the output of...'
    # # Initialize multiple CoT agents with a higher temperature for varied reasoning
    # cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent',  model=self.node_model, temperature=0.5) for _ in range(N)]
    
    # thinking_mapping = {}
    # answer_mapping = {}
    # possible_answers = []
    # for i in range(N):
    #     thinking, answer = await cot_agents[i]([taskInfo], extra_info, cot_instruction, is_sub_task=True)
    #     agents.append(f'CoT agent {cot_agents.id}, on the purpose of..., thinking: {thinking.content}; answer: {answer.content}')
    #     possible_answers.append(answer.content)
    #     thinking_mapping[answer.content] = thinking
    #     answer_mapping[answer.content] = answer
    # answer = majority_voting(possible_answers)

    # thinking = thinking_mapping[answer]
    # answer = answer_mapping[answer]
    # sub_tasks.append(f"Sub-task i output: thinking - {thinking.content}; answer - {answer.content}")

    # (7) Put the saved sub_tasks and agents to the final elf.make_final_answer. Make sure you have `is_sub_task=True` when calling an agent, and keep track of `sub_task`, `agents`, include `sub_task` dependency and detailed steps for the sub-task ('Sub-task i: based on sub-task,') in the sub-task instruction, and actually implmenet the blocks by yourself (for-loop if COT-SC, Debate and Reflextion).

    final_answer = self.make_final_answer(thinking, answer, sub_tasks, agents)
    # Return only the final answer
    return final_answer
"""
}

# ###The task background: {background_j1eval}
base = f"""

# Overview
You are an expert machine learning researcher testing various agentic systems. Given a set of architectures in the archive and the question. Note that architecture can contain multiple agents, and agent means a LLM that is used for specifical objectives by specifclaied setting (instruction, tempreture...)

Your objective is to 

(1) Perform task decomposition. Specifically, decompose the give question significantly so that the sub-architecture (or node or block) can perform each of the sub-tasks. The output should be sub-task 1, sub-task 2, ... sub-task n. Do not solve the task for the sub-architecture and do not leak the expected answer in your sub-task description/instruction/question (a short-cut like 'output exactly the following...' is also leakage and should be avoided). Instead, decompose the task that easy enough for the sub-architecture to solve. You need to justify how these sub-tasks can achieve the final answer to the orginal questions.

Make sure 

(a) Include sub-task ID  and 'Based on (task i)' in the instruction of a sub-task. 
For example, 
Similarly, if sub-task 2 requires the output of task 1, then sub-task 2's instruction should be
'Sub-task 2: Based on the outputs from sub-task 1, ....(origin sub-task 1's instruction)'

Similarly, if sub-task 3 requires the output of task 1 and 2, then sub-task 3's instruction should be
'Sub-task 3: Based on the outputs from sub-task 1 and 2, ....(origin sub-task 3's instruction)'
This helps each sub-task connects to its prerequisite sub-tasks so that there is enough information to solve it.

(b) Each sub-task should be specific and detailed enough to solve and to help achieve the final answer to the given question. The output should be helpful to solve the next sub-task. You need to include details steps (but not the answer) to the sub-task 
For example,
`Sub-task 3: Based on the output of sub-task 1 and 2....`
You can see it clearly states 'based on what sub-tasks'

(c) The answer to the last sub-task should be the same as the answer to the final question, so that the architecture successfully solve the complex question by solving each of the sub-task. 

(2) Given the resulting sub-task 1, sub-task 2, ... sub-task n, design connections between existing blocks to address each of them. 
You should structure the architecture as a multi-layered network. Each existing architecture (or blocks) serves as a node, while connections between them act as edges, forming a structured hierarchy of interactions. Additionally, you must determine the number of layers in the network.

For example, if the exising architectures are 'COT, COT_SC, Reflexion, LLM_debate' and you determine that there can be 3 layers. There are 3 resulting sub-task from (1) sub-task 1, sub-task 2, sub-task 1, sub-task 3:

Example Setup

Resulting sub-tasks:
sub-task 1, sub-task 2, sub-task 3, sub-task 4

Available architectures:
COT, COT_SC, Reflexion, LLM_debate

Network with 3 Layers:

Layer 1: COT  COT_SC  Reflexion  LLM_debate  
Layer 2: COT  COT_SC  Reflexion  LLM_debate   
Layer 3: COT  COT_SC  Reflexion  LLM_debate  

Connection Strategies:

1. Linear Connection: Directly link two block to pass information forward.
Example: [COT] (address sub-task 1) -> [LLM_debate] (address sub-task 2) (Single connection and exit)

2. Multi-Layer Connection: An block can appear in multiple layers, forming deeper reasoning structures.
Example: [COT] (address sub-task 1) -> [LLM_debate] (address sub-task 2) -> [COT -> Reflexion] (address sub-task 3) (COT appears in both Layer 1 and Layer 3) (the whole [COT -> Reflexion] is a sub-task architecture that aims to address sub-task 3)

IMPORTANT:

1. Decomposition itself should not be included in the architecture as the question has been decomposed at step (1). Do not assign one block to perform all the sub-tasks (if you put all decomposed sub-tasks into a single instruction for an block, it is very wrong). Instead, assign different block to address each of the sub-task instead.

2. If your previous attempts in Discovered architecture archive are incorrect (fitness value equals to 0), it means the sub-tasks are still too difficult to the corresponidng blocka. Please further decompose the question to easier sub-tasks. 

Your aim is to design an optimal block connection that can perform well on each of the sub-task.Your code should implement the exising blocks given in the archive (the 'code' entry of blocks) as it-is without modication: Do not propose new blocks or modify existing ones and only change the connections between the given blocks, but block setting like instruction, temperature are allowed to modify


{util_code}

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for and it should capture your thought process for reconnecting the existing blocks in archived. 

In the "(thought)" section, include the following:

(1) **Decomposition**: Given the new sub-task from (1). Form your final decomposition, which should include all of the new sub-task. Explain in details how do you decompose the question and how such decomposition is eaiser enough such that the subtask is solvable by the given agent, blocks and architecture 

(2) **Overall Architecture**: 

Given the resulting sub-task 1, sub-task 2, ... sub-task n, design connections between existing blocks to address each of them. describe your reasoning and the overall concept behind the connection design and finally detail the implementation steps. All connection must between exising blocks in the archive and no new blocks can be made. The format must strickly follow: 

(a) Use '->' for connection. for example, 'CoT (address sub-task 1) (existing block name) -> LLM debate (address sub-task 2) (another exising block name)' means connect the CoT block and the LLM debate block to address sub-task 1 and sub-task 2 correspondingly.

The second key ("name") corresponds to the name of your block connection architecture. 
Finally, the last key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project (so do not implement any other part), so please implement complete, reliable, reusable code snippets. You cannot call the exising blocks (e.g., COT, COT-SC) by its name, but must implement them as it is in the achive. If the block is handling a sub-task, add 'sub_task=True' when calling the block.

Here is an example of the output format for the new connected block architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. DON'T try to use some function that doesn't exisit.
Also, it could be helpful to set the LLM’s role and temperature to further control the LLM’s response. Note that the LLMAgentBase() will automatically parse the output and return a list of “Infos”. You can get the content by Infos.content. 
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

{wrong_implementation}


# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize the specified performance metrics by reconnecting the exisitng block in archived. Do not try to propose new block or modify the exising block, and only change the connection but block setting like instruction, tempreture are allowed to modify
Observe the discovered blocks carefully and think about what insights, lessons, or stepping stones can be learned from them.
You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
Use the knowledge from the archive and inspiration from academic literature to propose the new connection.

Below is the question to solve:\n\n[QUESTION]
"""