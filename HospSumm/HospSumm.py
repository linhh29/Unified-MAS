import argparse
import openai
from openai import AsyncOpenAI
import asyncio
import re, random, json, os, time, datetime
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Optional   

# Global async clients
_async_openai_client = None
_concurrency_semaphore = None

# OpenAI pricing per 1K tokens (as of 2024)
OPENAI_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-5-mini": {
        'input': 0.00025,
        'output': 0.002
    },
    "gemini-3-flash-preview": {
        'input': 0.0005,
        'output': 0.003
    },
    "deepseek-v3.2": {
        'input': 0.000284,
        'output': 0.000426
    },
    "qwen3-30b-a3b-instruct-2507": {
        'input': 0.0001065,
        'output': 0.000426
    },
}

@dataclass
class CostInfo:
    """Token usage and cost information for a single API call"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    model_name: str = ""


class ScenarioCostTracker:
    """Tracks costs for a single scenario, separated by agent type"""
    
    def __init__(self, scenario_id: int):
        self.scenario_id = scenario_id
        # Separate cost tracking for each agent type
        self.doctor_cost = CostInfo()
        self.patient_cost = CostInfo()
        self.measurement_cost = CostInfo()
        self.judge_cost = CostInfo()
        
        # Detailed call history for each agent
        self.doctor_calls: list = []
        self.patient_calls: list = []
        self.measurement_calls: list = []
        self.judge_calls: list = []
    
    def add_cost(self, agent_type: str, cost_info: CostInfo, call_detail: Optional[Dict] = None):
        """Add cost to the appropriate agent tracker"""
        if agent_type == "doctor":
            self.doctor_cost.input_tokens += cost_info.input_tokens
            self.doctor_cost.output_tokens += cost_info.output_tokens
            self.doctor_cost.total_tokens += cost_info.total_tokens
            self.doctor_cost.input_cost += cost_info.input_cost
            self.doctor_cost.output_cost += cost_info.output_cost
            self.doctor_cost.total_cost += cost_info.total_cost
            if call_detail:
                self.doctor_calls.append(call_detail)
        elif agent_type == "patient":
            self.patient_cost.input_tokens += cost_info.input_tokens
            self.patient_cost.output_tokens += cost_info.output_tokens
            self.patient_cost.total_tokens += cost_info.total_tokens
            self.patient_cost.input_cost += cost_info.input_cost
            self.patient_cost.output_cost += cost_info.output_cost
            self.patient_cost.total_cost += cost_info.total_cost
            if call_detail:
                self.patient_calls.append(call_detail)
        elif agent_type == "measurement":
            self.measurement_cost.input_tokens += cost_info.input_tokens
            self.measurement_cost.output_tokens += cost_info.output_tokens
            self.measurement_cost.total_tokens += cost_info.total_tokens
            self.measurement_cost.input_cost += cost_info.input_cost
            self.measurement_cost.output_cost += cost_info.output_cost
            self.measurement_cost.total_cost += cost_info.total_cost
            if call_detail:
                self.measurement_calls.append(call_detail)
        elif agent_type == "judge":
            self.judge_cost.input_tokens += cost_info.input_tokens
            self.judge_cost.output_tokens += cost_info.output_tokens
            self.judge_cost.total_tokens += cost_info.total_tokens
            self.judge_cost.input_cost += cost_info.input_cost
            self.judge_cost.output_cost += cost_info.output_cost
            self.judge_cost.total_cost += cost_info.total_cost
            if call_detail:
                self.judge_calls.append(call_detail)
    
    def get_total_cost(self) -> float:
        """Get total cost for this scenario"""
        return (self.doctor_cost.total_cost + self.patient_cost.total_cost + 
                self.measurement_cost.total_cost + self.judge_cost.total_cost)
    
    def get_summary(self) -> Dict:
        """Get cost summary for this scenario"""
        return {
            "scenario_id": self.scenario_id,
            "doctor": {
                "input_tokens": self.doctor_cost.input_tokens,
                "output_tokens": self.doctor_cost.output_tokens,
                "total_tokens": self.doctor_cost.total_tokens,
                "input_cost": round(self.doctor_cost.input_cost, 6),
                "output_cost": round(self.doctor_cost.output_cost, 6),
                "total_cost": round(self.doctor_cost.total_cost, 6),
                "num_calls": len(self.doctor_calls)
            },
            "patient": {
                "input_tokens": self.patient_cost.input_tokens,
                "output_tokens": self.patient_cost.output_tokens,
                "total_tokens": self.patient_cost.total_tokens,
                "input_cost": round(self.patient_cost.input_cost, 6),
                "output_cost": round(self.patient_cost.output_cost, 6),
                "total_cost": round(self.patient_cost.total_cost, 6),
                "num_calls": len(self.patient_calls)
            },
            "measurement": {
                "input_tokens": self.measurement_cost.input_tokens,
                "output_tokens": self.measurement_cost.output_tokens,
                "total_tokens": self.measurement_cost.total_tokens,
                "input_cost": round(self.measurement_cost.input_cost, 6),
                "output_cost": round(self.measurement_cost.output_cost, 6),
                "total_cost": round(self.measurement_cost.total_cost, 6),
                "num_calls": len(self.measurement_calls)
            },
            "judge": {
                "input_tokens": self.judge_cost.input_tokens,
                "output_tokens": self.judge_cost.output_tokens,
                "total_tokens": self.judge_cost.total_tokens,
                "input_cost": round(self.judge_cost.input_cost, 6),
                "output_cost": round(self.judge_cost.output_cost, 6),
                "total_cost": round(self.judge_cost.total_cost, 6),
                "num_calls": len(self.judge_calls)
            },
            "total": {
                "total_cost": round(self.get_total_cost(), 6),
                "total_tokens": (self.doctor_cost.total_tokens + self.patient_cost.total_tokens + 
                                self.measurement_cost.total_tokens + self.judge_cost.total_tokens)
            }
        }

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> CostInfo:
    """Calculate cost for a given model and token usage"""
    # Normalize model name
    model_key = model_name.lower()
    
    # Try to find matching pricing
    pricing = OPENAI_PRICING.get(model_name)
    input_cost = (input_tokens / 1000.0) * pricing["input"]
    output_cost = (output_tokens / 1000.0) * pricing["output"]
    total_cost = input_cost + output_cost
    total_tokens = input_tokens + output_tokens
    
    return CostInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        model_name=model_name
    )


def _lcs_length(x: str, y: str) -> int:
    """Compute length of Longest Common Subsequence between two strings (token-based)."""
    x_tokens = x.split()
    y_tokens = y.split()
    m, n = len(x_tokens), len(y_tokens)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        xi = x_tokens[i - 1]
        for j in range(1, n + 1):
            if xi == y_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return dp[m][n]


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between hypothesis and reference (token-based).
    """
    hypothesis = hypothesis.strip()
    reference = reference.strip()
    if not hypothesis or not reference:
        return 0.0
    lcs = _lcs_length(hypothesis, reference)
    m = len(reference.split())
    n = len(hypothesis.split())
    if m == 0 or n == 0 or lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    beta_sq = 1.0
    f1 = (1 + beta_sq) * prec * rec / (rec + beta_sq * prec)
    return f1


async def llm_judge_score(hypothesis: str, reference: str, cost_tracker: Optional[ScenarioCostTracker] = None) -> float:
    """
    Use LLM-as-judge to evaluate the quality of hypothesis summary compared to reference.
    
    Args:
        hypothesis: The generated summary from MAS
        reference: The reference summary
        cost_tracker: Optional cost tracker to record judge API calls
    
    Returns:
        float: Score between 0.0 and 1.0 (or other scale as defined by judge)
    """
    hypothesis = hypothesis.strip()
    reference = reference.strip()
    if not hypothesis or not reference:
        return 0.0
    
    # ========== JUDGE PROMPT PLACEHOLDER - START ==========
    # TODO: Replace the following prompts with your actual judge prompts
    
    system_prompt = """
    You are an expert Board-Certified Physician and a specialized Medical Scribe evaluator. Your task is to evaluate the quality of a generated "Hospitalization Summary" by comparing it against a ground-truth "Reference Summary" written by a human doctor.
    """
    
    judge_prompt = f"""
    The goal of the summarization task is to: "Summarize the key diagnostic information and significant results based on the patient's multiple health records during hospitalization."

    Please evaluate the [Generated Summary] based on the [Reference Summary] across the following four dimensions. For each dimension, assign a score from 0 to 10.

    ### Evaluation Dimensions:

    **1. Factual Accuracy (0-10)**
    *   **Definition:** Does the generated summary contain any factual errors, hallucinations, or contradictions compared to the reference?
    *   **Scoring:**
        *   10: Absolutely no factual errors. All claims are supported by the reference.
        *   0: Contains dangerous medical errors (e.g., stating a patient has a disease they do not have, or flipping a negative result to positive).

    **2. Completeness of Diagnostics (0-10)**
    *   **Definition:** Does the generated summary capture all the "Key Diagnostic Information" and "Significant Results" found in the reference? Look for specific diseases, procedures, and critical lab values.
    *   **Scoring:**
        *   10: Captures 100% of the key diagnoses and significant findings mentioned in the reference.
        *   0: Misses the primary reason for hospitalization or the main diagnosis.

    **3. Coherence & Medical Professionalism (0-10)**
    *   **Definition:** Is the summary well-structured, logically organized, and written in professional medical language?
    *   **Scoring:**
        *   10: Reads like a professional discharge summary. Perfect terminology and flow.
        *   0: Incoherent, disjointed, or uses unprofessional/layman language inappropriately.

    **4. Conciseness (0-10)**
    *   **Definition:** Does the summary avoid redundancy and irrelevant details?
    *   **Scoring:**
        *   10: Very concise. Every sentence adds value to the clinical picture.
        *   0: Extremely repetitive or filled with irrelevant nursing care details (e.g., "patient slept well") that are not diagnostic.
    
    Reference Summary:
    {reference}
    
    Generated Summary:
    {hypothesis}
    
    Please output your evaluation in the following JSON format only, without additional commentary outside the JSON:

    {{
    "reasoning": "A brief explanation of the evaluation (max 3 sentences), highlighting specific missing facts or errors.",
    "scores": {{
        "factual_accuracy": <float>,
        "completeness": <float>,
        "coherence": <float>,
        "conciseness": <float>
    }}
    }}

    """
    # ========== JUDGE PROMPT PLACEHOLDER - END ==========
    
    # Call LLM judge (using doctor_llm or a separate judge model)
    # For now, we'll use the same backend as doctor_llm, but you can customize this
    judge_model = "gpt-4o"  # TODO: Make this configurable if needed
    
    try:
        answer, cost_info = await query_model(
            judge_model,
            judge_prompt,
            system_prompt
        )
        
        # Record cost if tracker is provided
        if cost_tracker:
            cost_tracker.add_cost("judge", cost_info, {
                "model": cost_info.model_name,
                "input_tokens": cost_info.input_tokens,
                "output_tokens": cost_info.output_tokens,
                "cost": cost_info.total_cost
            })
        
        # Parse score from LLM response
        # Try to extract a numeric score from the response
        # This is a simple extraction - you may want to improve this based on your prompt format
        response_json = json.loads(answer)
        factual_accuracy = response_json.get("scores", {}).get("factual_accuracy", 0.0)
        completeness = response_json.get("scores", {}).get("completeness", 0.0)
        coherence = response_json.get("scores", {}).get("coherence", 0.0)
        conciseness = response_json.get("scores", {}).get("conciseness", 0.0)
        weighted_average_score = (factual_accuracy * 0.25 + completeness * 0.25 + coherence * 0.25 + conciseness * 0.25) / 10.0
        return weighted_average_score
    except Exception as e:
        print(f"Error in LLM judge evaluation: {e}")
        return 0.0


def init_clients(api_key=None, max_concurrent=10, base_url=None):
    """Initialize async clients and concurrency control"""
    global _async_openai_client, _concurrency_semaphore
    if api_key:
        _async_openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    _concurrency_semaphore = asyncio.Semaphore(max_concurrent)


async def query_model(model_str, prompt, system_prompt, tries=30, timeout=20.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    """Async version of query_model with concurrency control
    
    Returns:
        tuple: (answer, cost_info) where cost_info is a CostInfo object
    """
    
    # Use semaphore to control concurrency
    if _concurrency_semaphore:
        async with _concurrency_semaphore:
            return await _query_model_impl(model_str, prompt, system_prompt, tries, timeout, image_requested, scene, max_prompt_len, clip_prompt)
    else:
        return await _query_model_impl(model_str, prompt, system_prompt, tries, timeout, image_requested, scene, max_prompt_len, clip_prompt)


async def _query_model_impl(model_str, prompt, system_prompt, tries=30, timeout=20.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    """Internal implementation of async query_model
    
    Returns:
        tuple: (answer, cost_info) where cost_info is a CostInfo object
    """

    if clip_prompt: 
        prompt = prompt[:max_prompt_len]
    answer = None
    cost_info = None
    
    # Handle text-only requests
    if answer is None:
        if model_str in OPENAI_PRICING.keys():
            if not _async_openai_client:
                raise Exception("OpenAI client not initialized")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            if model_str == 'gpt-5-mini':
                response = await _async_openai_client.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768,
                    reasoning_effort='low',
                )
            elif model_str == 'gemini-3-flash-preview':
                response = await _async_openai_client.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            elif model_str == 'deepseek-v3.2':
                response = await _async_openai_client.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            elif model_str == 'qwen3-30b-a3b-instruct-2507':
                response = await _async_openai_client.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            elif model_str == 'gpt-4o':
                response = await _async_openai_client.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=1,
                    response_format={"type": "json_object"}
                )
            # print(response)
            
            answer = response.choices[0].message.content
            answer = re.sub(r"\s+", " ", answer)
            
            # Extract token usage
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost_info = calculate_cost(model_str, input_tokens, output_tokens)
    
    if answer:
        # Return answer and cost_info (create zero cost if not calculated)
        if cost_info is None:
            cost_info = CostInfo(model_name=model_str)
        return answer, cost_info
                
# demonstration = """
# Patient Records:
# Input:\nage: 47. gender: F.  \nName:  ___.                   Unit No:   ___\n \nAdmission Date:  ___              Discharge Date:   ___\n \nDate of Birth:  ___             Sex:   F\n \nService: SURGERY\n \nAllergies: \nGadolinium-Containing Agents\n \nAttending: ___.\n \nChief Complaint:\nabd pain\n \nMajor Surgical or Invasive Procedure:\nExploratory laparotomy, lysis adhesions.\n\n \nHistory of Present Illness:\n___ F who is 7 months s/p low anterior resection for colon\nadenocarcinoma who presents with severe abdominal pain, nausea,\nand vomiting since 9 ___ last night. She had been doing very well\nuntil that time. She had pizza for dinner, and then her pain \ncame\non quite suddenly. She describes it as sharp and severe,\nradiating to her back. She reports that it's the same as when \nshe\nhad an SBO in ___ after her colectomy. She has vomited\nmultiple times. Denies any hematemesis, fevers, chills, or\nurinary symptoms. Passed gas last night, but not today.\n.\n\n \nPast Medical History:\nBreast CA (full details in OMR)\nCervical DDD\nDepression\n \nSocial History:\n___\nFamily History:\nShe has no known family history of breast or ovarian cancer.  \nShe does have a\nrelatively large family and has a maternal grandmother with \ncolon cancer as the only known cancer diagnosis.\n \nPhysical Exam:\nGen: a and o x 3, nad\nV.S.S\nCV: RRR no m/r/g\nRESP: LSCTA, bilat\nABD: soft, nt, nd, + bs\nIncision OTA with steri strips, no s/s of infections\nEXT: no c/c/e\n \nPertinent Results:\n___ 06:15AM BLOOD WBC-4.3# RBC-3.36* Hgb-9.9* Hct-30.4* \nMCV-91 MCH-29.5 MCHC-32.5 RDW-13.7 Plt ___\n___ 11:20AM BLOOD WBC-8.7# RBC-4.24 Hgb-12.4 Hct-37.7 \nMCV-89 MCH-29.2 MCHC-32.9 RDW-13.4 Plt ___\n___ 11:20AM BLOOD Neuts-91.8* Lymphs-6.7* Monos-1.0* \nEos-0.3 Baso-0.1\n___ 06:15AM BLOOD Glucose-112* UreaN-9 Creat-0.7 Na-139 \nK-4.1 Cl-104 HCO3-28 AnGap-11\n___ 11:20AM BLOOD Glucose-121* UreaN-15 Creat-0.7 Na-139 \nK-3.8 Cl-103 HCO3-25 AnGap-15\n___ 11:20AM BLOOD ALT-14 AST-22 AlkPhos-79 TotBili-0.2\n___ 06:15AM BLOOD Calcium-8.2* Phos-2.3* Mg-1.9\n.\nCT ABd/Pelvis ___: Oral contrast has only minimally progressed \nsince study three hours prior with continued dilated loops of \nsmall bowel and transition point in the low to mid pelvis. These \nfindings are consistent with small-bowel obstruction and \npossible internal hernia. No contrast has transited into the \ncolon. Proximal loops of bowel in the left upper quadrant are \nnot dilated. \n \nBrief Hospital Course:\nPatient was admitted to the surgical service secondary to SBO. \nShe was taken to the OR for ex-lap and loa. She returned to the \nfloor and maintained NPO with IVF/MEDS. \nPain was well controlled. With the return of bowel function and \nflatus her diet was slowly advanced from sips to regular, \ntolerated well. She was than started on PO/Home meds and her \nfoley was removed. Her staples were removed and steri strips \nwere applied. \nAll d/c paperwork was reviewed and the patient will follow up \nwith Dr. ___ in ___ weeks. \n \nMedications on Admission:\nativan 1 QHS prn, naproxen 220\", trazadone 100 QHS, famotidine \n20\", venlafaxine XR 150'\n\n \nDischarge Medications:\n1. Venlafaxine 75 mg Capsule, Sust. Release 24 hr Sig: Two (2) \nCapsule, Sust. Release 24 hr PO DAILY (Daily).  \n2. Famotidine 20 mg Tablet Sig: One (1) Tablet PO Q12H (every 12 \nhours).  \n3. Docusate Sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 \ntimes a day) as needed for constipation for 1 months: Take with \npain meds.\nDisp:*60 Capsule(s)* Refills:*0*\n4. Hydromorphone 2 mg Tablet Sig: ___ Tablets PO Q3H (every 3 \nhours) as needed for pain for 2 weeks.\nDisp:*45 Tablet(s)* Refills:*0*\n\n \nDischarge Disposition:\nHome\n \nDischarge Diagnosis:\nStrangulating small bowel obstruction.\n\n \nDischarge Condition:\nStable.\nTolerating regular diet.\nPain well controlled with oral medications.\n\n \nDischarge Instructions:\nPlease call your doctor or return to the ER for any of the \nfollowing:\n* You experience new chest pain, pressure, squeezing or \ntightness.\n* New or worsening cough or wheezing.\n* If you are vomiting and cannot keep in fluids or your \nmedications.\n* You are getting dehydrated due to continued vomiting, \ndiarrhea or other reasons. Signs of dehydration include dry \nmouth, rapid heartbeat or feeling dizzy or faint when standing.\n* You see blood or dark/black material when you vomit or have a \nbowel movement.\n* Your pain is not improving within ___ hours or not gone \nwithin 24 hours. Call or return immediately if your pain is \ngetting worse or is changing location or moving to your chest or \n\nback.\n*Avoid lifting objects > 5lbs until your follow-up appointment \nwith the surgeon.\n*Avoid driving or operating heavy machinery while taking pain \nmedications.\n* You have shaking chills, or a fever greater than 101.5 (F) \ndegrees or 38(C) degrees.\n* Any serious change in your symptoms, or any new symptoms that \nconcern you. \n* Please resume all regular home medications and take any new \nmeds \nas ordered. \n* Continue to ambulate several times per day.\n.\nIncision Care:\n-Your staple will be removed at your follow up appointment. \n-Steri-strips will be applied and will fall off on their own. \nPlease remove any remaining strips ___ days after appllicaiton.\n-You may shower, and wash surgical incisions.\n-Avoid swimming and baths until your follow-up appointment.\n-Please call the doctor if you have increased pain, swelling, \nredness, or drainage from the incision sites.\n.\n\n \nFollowup Instructions:\n___\n \n\nOutput:

# Answer:
# The patient is a 47-year-old female who was admitted to the hospital with severe abdominal pain, nausea, and vomiting. She had a history of colon adenocarcinoma and had undergone a low anterior resection 7 months prior. She presented with sudden onset of severe abdominal pain, which was similar to a previous small bowel obstruction (SBO) she experienced after her colectomy. \n\nThe patient's past medical history includes breast cancer, cervical DDD, and depression. She has no known family history of breast or ovarian cancer, but her maternal grandmother had colon cancer. \n\nThe patient underwent an exploratory laparotomy and lysis of adhesions, and was found to have a strangulating small bowel obstruction. She was managed with IV fluids, medications, and a nasogastric tube, and was eventually able to tolerate a regular diet. \n\nThe patient was discharged with a diagnosis of strangulating small bowel obstruction, and was prescribed medications including venlafaxine, famotidine, docusate sodium, and hydromorphone. She was advised to follow up with her doctor in 6 weeks and to avoid lifting heavy objects, driving, or operating heavy machinery while taking pain medications.
# """

class ScenarioHospitalSummarization:
    """
    Scenario for hospitalization summarization.
    We reuse the existing MAS interface:
      - examiner_information() -> full instruct (for physician agent)
      - patient_information()  -> full instruct (for nursing notes agent)
      - exam_information()     -> full instruct (for medication agent)
      - diagnosis_information() -> reference summary (answer)
    
    All agents receive the complete hospitalization record (instruct),
    but each agent focuses on their specialized domain through system prompts.
    """
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.hadm_id = scenario_dict.get("hadm_id")
        self.subject_id = scenario_dict.get("subject_id")
        self.gender = scenario_dict.get("gender")
        self.age = scenario_dict.get("age")
        self.instruct: str = scenario_dict.get("instruct", "")
        self.answer: str = scenario_dict.get("answer", "")

        # Parse instruct into physician notes, nursing notes, and medications
        self.physician_notes, self.nursing_notes, self.medication_notes = self._parse_instruct(self.instruct)

    @staticmethod
    def _parse_instruct(instruct: str):
        """
        Heuristically split the long hospitalization note into:
          - physician_notes: main clinical narrative (everything before 'Discharge Medications:' if present)
          - medication_notes: 'Discharge Medications:' section
          - nursing_notes: 'Discharge Instructions:' section
        If markers are missing, fall back gracefully.
        """
        physician_notes = instruct
        nursing_notes = ""
        medication_notes = ""

        text = instruct

        # Discharge Medications section
        med_marker = "Discharge Medications:"
        med_idx = text.find(med_marker)
        if med_idx != -1:
            physician_notes = text[:med_idx].strip()
            end_idx = len(text)
            for marker in [
                "\n\nDischarge Disposition:",
                "\nDischarge Disposition:",
                "\n\nDischarge Diagnosis:",
                "\nDischarge Diagnosis:",
                "\n\nDischarge Condition:",
                "\nDischarge Condition:",
                "\n\nDischarge Instructions:",
                "\nDischarge Instructions:",
            ]:
                pos = text.find(marker, med_idx)
                if pos != -1 and pos < end_idx:
                    end_idx = pos
            medication_notes = text[med_idx:end_idx].strip()

        # Discharge Instructions section
        instr_marker = "Discharge Instructions:"
        instr_idx = text.find(instr_marker)
        if instr_idx != -1:
            nursing_notes = text[instr_idx:].strip()

        if not physician_notes:
            physician_notes = instruct.strip()
        return physician_notes, nursing_notes, medication_notes

    def patient_information(self) -> str:
        # Return full instruct for nursing notes agent
        return self.instruct

    def examiner_information(self) -> str:
        # Return full instruct for physician agent
        return self.instruct

    def exam_information(self) -> str:
        # Return full instruct for medication agent
        return self.instruct

    def diagnosis_information(self) -> str:
        # Reference summary for evaluation
        return self.answer


class ScenarioLoaderHospitalSummarization:
    def __init__(self) -> None:
        path = "/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/hosp_summ_test_16.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioHospitalSummarization(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]




class PatientAgent:
    def __init__(self, scenario, backend_str="gpt-5-mini", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    async def inference_patient(self, question, cost_tracker: Optional[ScenarioCostTracker] = None) -> str:
        answer, cost_info = await query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        # Record cost if tracker is provided
        if cost_tracker:
            cost_tracker.add_cost("patient", cost_info, {
                "model": cost_info.model_name,
                "input_tokens": cost_info.input_tokens,
                "output_tokens": cost_info.output_tokens,
                "cost": cost_info.total_cost
            })
        return answer

    def system_prompt(self) -> str:
        # Special prompt for hospitalization summarization (nursing notes agent)
        if isinstance(self.scenario, ScenarioHospitalSummarization):
            base = (
                "You are a hospital nurse responsible for providing nursing notes and discharge instructions. "
                "Your role is CRITICAL: you are the expert on patient care, nursing observations, and discharge instructions. "
                "You have access to the complete hospitalization record, but you MUST focus ONLY on your area of expertise:\n"
                "- Nursing observations and patient care activities\n"
                "- Discharge instructions and patient education\n"
                "- Nursing-related details from the hospitalization record\n\n"
                "When the physician asks you questions, respond ONLY with information relevant to nursing care and discharge instructions. "
                "Keep your responses brief (1-3 sentences) and focused on YOUR domain. "
                "Do NOT provide physician notes, medication details, or write the final discharge summary yourself. "
                "Your responses should help the physician write a comprehensive discharge summary."
            )
            info = (
                "\n\nBelow is the complete hospitalization record available to you:\n"
                f"{self.symptoms}\n\n"
                "Remember: You are a NURSE. Focus ONLY on nursing-related information when answering questions. "
                "If asked about medications, physician notes, or other non-nursing topics, politely redirect or say you don't have that information."
            )
            return base + info

        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt-5-mini", max_infs=20, min_infs=10, bias_present=None, img_request=False) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # minimum number of inference calls before summary can be output
        self.MIN_INFS = min_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    async def inference_doctor(self, question, image_requested=False, cost_tracker: Optional[ScenarioCostTracker] = None) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        answer, cost_info = await query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        # Record cost if tracker is provided
        if cost_tracker:
            cost_tracker.add_cost("doctor", cost_info, {
                "model": cost_info.model_name,
                "input_tokens": cost_info.input_tokens,
                "output_tokens": cost_info.output_tokens,
                "cost": cost_info.total_cost,
                "inference_num": self.infs
            })
        return answer

    def system_prompt(self) -> str:
        # Special prompt for hospitalization summarization (physician notes / summarization agent)
        if isinstance(self.scenario, ScenarioHospitalSummarization):
            base = (
                "You are a hospital PHYSICIAN responsible for writing a concise, paragraph-style discharge summary. "
                "Your role is CRITICAL: you are the LEADER of the medical team and must synthesize information from multiple sources. "
                f"You are allowed to interact for up to {self.MAX_INFS} turns; you have used {self.infs} turns so far.\n\n"
                f"IMPORTANT: You MUST conduct at least {self.MIN_INFS} rounds of dialogue before outputting the final summary. "
                f"You have only completed {self.infs} rounds so far. "
                f"You need at least {max(0, self.MIN_INFS - self.infs)} more rounds of interaction with the nursing and medication agents "
                f"to gather sufficient information before you can write the summary.\n\n"
                "You have access to the complete hospitalization record, but you work with specialized team members:\n"
                "- The NURSING AGENT: Expert on nursing care, patient observations, and discharge instructions\n"
                "- The MEDICATION AGENT: Expert on discharge medications and medication-related information\n\n"
                "Your responsibilities:\n"
                "1. Review the complete hospitalization record (available below)\n"
                "2. Ask the nursing agent questions about nursing care and discharge instructions using natural language\n"
                "3. Query the medication agent using \"REQUEST TEST: [any text]\" to obtain medication information\n"
                "4. Synthesize ALL information to write a CONCISE, PARAGRAPH-STYLE summary\n\n"
                "CRITICAL FORMAT REQUIREMENTS:\n"
                "- Your summary MUST be written as a CONCISE, FLOWING PARAGRAPH (or a few short paragraphs), NOT a detailed structured discharge summary\n"
                "- DO NOT use structured formats like \"**DISCHARGE SUMMARY**\", \"**DIAGNOSES:**\", \"**HOSPITAL COURSE:**\", \"**DISCHARGE MEDICATIONS:**\", etc.\n"
                "- DO NOT list medications, diagnoses, or procedures in bullet points or numbered lists\n"
                "- Write naturally, summarizing key information in paragraph form\n"
                "- Focus on: patient demographics/history, chief complaints, key diagnoses and treatments during hospitalization, and discharge condition\n"
                "- Keep it concise (typically 2-4 sentences per key topic area)\n"
                "- The summary should read like a narrative summary, not a structured medical document\n\n"
                "Your intermediate turns should be short (1-3 sentences) and focused on gathering necessary information. "
                f"DO NOT output \"SUMMARY READY\" until you have completed at least {self.MIN_INFS} rounds of dialogue. "
                f"When you have gathered sufficient information (after at least {self.MIN_INFS} rounds), "
                "respond with \"SUMMARY READY: [your concise paragraph-style summary text]\"."
            )
            presentation = (
                "\n\nBelow is the complete hospitalization record available to you:\n"
                f"{self.presentation}\n\n"
                "Use this information together with information from the nursing agent and medication agent to write a CONCISE, PARAGRAPH-STYLE summary. "
                "Remember: You are writing a narrative summary, NOT a structured discharge summary document. "
                "Summarize the key diagnostic information and significant results in flowing paragraph form, similar to how a physician would verbally summarize a patient's hospitalization."
            )
            return base + presentation

        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt-5-mini") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    async def inference_measurement(self, question, cost_tracker: Optional[ScenarioCostTracker] = None) -> str:
        answer = str()
        answer, cost_info = await query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        # Record cost if tracker is provided
        if cost_tracker:
            cost_tracker.add_cost("measurement", cost_info, {
                "model": cost_info.model_name,
                "input_tokens": cost_info.input_tokens,
                "output_tokens": cost_info.output_tokens,
                "cost": cost_info.total_cost
            })
        return answer

    def system_prompt(self) -> str:
        # Special prompt for hospitalization summarization (medication agent)
        if isinstance(self.scenario, ScenarioHospitalSummarization):
            base = (
                "You are a MEDICATION SPECIALIST responsible for providing medication-related information. "
                "Your role is CRITICAL: you are the expert on all medication-related details from the hospitalization record. "
                "You have access to the complete hospitalization record, but you MUST focus ONLY on your area of expertise:\n"
                "- Discharge medications and medication lists\n"
                "- Medication dosages, frequencies, and instructions\n"
                "- Medication-related information from the hospitalization record\n\n"
                "When the physician requests medication information (using \"REQUEST TEST: ...\"), "
                "respond with the patient's discharge medications in the format "
                "\"RESULTS: [medication list here]\". "
                "Extract medication information from the complete record, focusing on sections like "
                "\"Discharge Medications:\" or medication-related content.\n"
                "If no medication information is available in the record, respond with "
                "\"RESULTS: NO MEDICATION INFORMATION AVAILABLE\".\n\n"
                "Remember: You are a MEDICATION SPECIALIST. Focus ONLY on medication-related information. "
                "Do NOT provide nursing notes, physician notes, or other non-medication information."
            )
            presentation = (
                "\n\nBelow is the complete hospitalization record available to you:\n"
                f"{self.information}\n\n"
                "Extract and provide ONLY medication-related information when requested by the physician."
            )
            return base + presentation

        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def save_scenario_results(scenario_id, scenario, dialogue_history, diagnosis_text, 
                          correct_diagnosis, is_correct, cost_tracker, output_dir, llm_judge_score=None):
    """Save scenario results to files (synchronous file operations)"""
    # Create scenario directory
    scenario_dir = os.path.join(output_dir, f"scenario_{scenario_id}")
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Save dialogue history
    dialogue_file = os.path.join(scenario_dir, "dialogue.json")
    with open(dialogue_file, "w", encoding="utf-8") as f:
        json.dump({
            "scenario_id": scenario_id,
            "dialogue_history": dialogue_history,
            "total_rounds": len(dialogue_history)
        }, f, indent=2, ensure_ascii=False)
    
    # Save result
    result_file = os.path.join(scenario_dir, "result.json")
    # Get scenario data safely - try to serialize scenario_dict
    scenario_data = {}
    try:
        if hasattr(scenario, 'scenario_dict'):
            # Try to serialize scenario_dict directly
            scenario_data = scenario.scenario_dict
            # Convert to JSON-serializable format if needed
            scenario_data = json.loads(json.dumps(scenario_data, default=str))
        else:
            # Try to get serializable data from scenario methods
            scenario_data = {
                "patient_info": scenario.patient_information() if hasattr(scenario, 'patient_information') else None,
                "examiner_info": scenario.examiner_information() if hasattr(scenario, 'examiner_information') else None,
                "diagnosis": scenario.diagnosis_information() if hasattr(scenario, 'diagnosis_information') else None,
            }
            # Convert to JSON-serializable format
            scenario_data = json.loads(json.dumps(scenario_data, default=str))
    except Exception as e:
        # If serialization fails, save minimal information
        scenario_data = {
            "error": f"Failed to serialize scenario data: {str(e)}",
            "diagnosis": correct_diagnosis
        }
    
    result_data = {
        "scenario_id": scenario_id,
        "diagnosis_made": bool(diagnosis_text),
        "doctor_diagnosis": diagnosis_text if diagnosis_text else None,
        "correct_diagnosis": correct_diagnosis,
        "is_correct": is_correct,  # True if diagnosis is correct, False if incorrect or no diagnosis
        "diagnosis_status": "correct" if is_correct and diagnosis_text else ("incorrect" if diagnosis_text else "no_diagnosis"),
        "llm_judge_score": llm_judge_score,  # LLM-as-judge score for hospitalization summarization (None for other datasets)
        "scenario_data": scenario_data
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # Save cost information
    cost_file = os.path.join(scenario_dir, "cost.json")
    cost_summary = cost_tracker.get_summary()
    # Add detailed call history
    cost_summary["detailed_calls"] = {
        "doctor": cost_tracker.doctor_calls,
        "patient": cost_tracker.patient_calls,
        "measurement": cost_tracker.measurement_calls,
        "judge": cost_tracker.judge_calls
    }
    # Add diagnosis correctness to cost file for easy reference
    cost_summary["diagnosis_result"] = {
        "diagnosis_made": bool(diagnosis_text),
        "is_correct": is_correct,
        "doctor_diagnosis": diagnosis_text if diagnosis_text else None,
        "correct_diagnosis": correct_diagnosis
    }
    with open(cost_file, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2, ensure_ascii=False)


async def process_single_scenario(scenario_id, scenario_loader, inf_type, doctor_bias, patient_bias, 
                                  doctor_llm, patient_llm, measurement_llm, 
                                  dataset, img_request, total_inferences, min_infs=10, output_dir=None, 
                                  scenario_semaphore=None):
    """Process a single scenario asynchronously"""
    # Use semaphore to control scenario-level concurrency
    if scenario_semaphore:
        async with scenario_semaphore:
            return await _process_single_scenario_impl(
                scenario_id, scenario_loader, inf_type, doctor_bias, patient_bias,
                doctor_llm, patient_llm, measurement_llm,
                dataset, img_request, total_inferences, min_infs, output_dir
            )
    else:
        return await _process_single_scenario_impl(
            scenario_id, scenario_loader, inf_type, doctor_bias, patient_bias,
            doctor_llm, patient_llm, measurement_llm,
            dataset, img_request, total_inferences, min_infs, output_dir
        )


async def _process_single_scenario_impl(scenario_id, scenario_loader, inf_type, doctor_bias, patient_bias, 
                                  doctor_llm, patient_llm, measurement_llm, 
                                  dataset, img_request, total_inferences, min_infs=10, output_dir=None):
    """Internal implementation of process_single_scenario"""
    # Initialize cost tracker for this scenario
    cost_tracker = ScenarioCostTracker(scenario_id)
    
    # Initialize dialogue history for recording
    dialogue_history = []
    
    pi_dialogue = str()
    # Initialize scenarios (MedQA/NEJM)
    scenario = scenario_loader.get_scenario(id=scenario_id)
    # Initialize agents
    meas_agent = MeasurementAgent(
        scenario=scenario,
        backend_str=measurement_llm)
    patient_agent = PatientAgent(
        scenario=scenario, 
        bias_present=patient_bias,
        backend_str=patient_llm)
    doctor_agent = DoctorAgent(
        scenario=scenario, 
        bias_present=doctor_bias,
        backend_str=doctor_llm,
        max_infs=total_inferences,
        min_infs=min_infs,
        img_request=img_request)

    doctor_dialogue = ""
    diagnosis_made = False
    diagnosis_text = ""
    correct_diagnosis = ""
    is_correct = False
    llm_judge_score_value = None
    
    for _inf_id in range(total_inferences):
        # Check for medical image request
        if dataset == "NEJM":
            if img_request:
                imgs = "REQUEST IMAGES" in doctor_dialogue
            else: 
                imgs = True
        else: 
            imgs = False
        
        # Check if final inference
        if _inf_id == total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
        
        # Obtain doctor dialogue (human or llm agent)
        if inf_type == "human_doctor":
            doctor_dialogue = input("\nQuestion for patient: ")
        else: 
            doctor_dialogue = await doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs, cost_tracker=cost_tracker)
        
        print("Scene {} - Doctor [{}%]:".format(scenario_id, int(((_inf_id+1)/total_inferences)*100)), doctor_dialogue)
        
        # Record doctor dialogue
        dialogue_history.append({
            "round": _inf_id + 1,
            "type": "doctor",
            "content": doctor_dialogue,
            "progress": int(((_inf_id+1)/total_inferences)*100)
        })
        
        # Doctor has arrived at a diagnosis / summary, check correctness / score
        trigger_phrase = "SUMMARY READY" if isinstance(scenario, ScenarioHospitalSummarization) else "DIAGNOSIS READY"
        
        # Check if summary/diagnosis is ready
        summary_ready = trigger_phrase in doctor_dialogue
        diagnosis_ready = "DIAGNOSIS READY" in doctor_dialogue and not isinstance(scenario, ScenarioHospitalSummarization)
        
        # For hospitalization summarization, enforce minimum rounds requirement
        if isinstance(scenario, ScenarioHospitalSummarization) and summary_ready:
            if doctor_agent.infs < doctor_agent.MIN_INFS:
                # Ignore SUMMARY READY if minimum rounds not reached, continue dialogue
                print(f"\nScene {scenario_id} - Warning: Doctor attempted to output summary at round {doctor_agent.infs}, "
                      f"but minimum {doctor_agent.MIN_INFS} rounds required. Continuing dialogue...")
                summary_ready = False  # Ignore the summary ready signal
        
        if summary_ready or diagnosis_ready:
            diagnosis_made = True
            # Extract diagnosis from doctor dialogue
            diagnosis_text = doctor_dialogue
            # Prefer dataset-specific trigger phrase when present
            if trigger_phrase in doctor_dialogue:
                parts = doctor_dialogue.split(trigger_phrase)
                if len(parts) > 1:
                    diagnosis_text = parts[-1].strip().lstrip(":").strip()
            elif "DIAGNOSIS READY:" in doctor_dialogue:
                diagnosis_text = doctor_dialogue.split("DIAGNOSIS READY:")[-1].strip()
            elif "DIAGNOSIS READY" in doctor_dialogue:
                # Try to extract text after DIAGNOSIS READY
                parts = doctor_dialogue.split("DIAGNOSIS READY")
                if len(parts) > 1:
                    diagnosis_text = parts[-1].strip().lstrip(":").strip()
            
            diagnosis_info = scenario.diagnosis_information()
            correct_diagnosis = diagnosis_info if isinstance(diagnosis_info, str) else str(diagnosis_info)

            # Hospitalization summarization: use LLM-as-judge for evaluation
            if isinstance(scenario, ScenarioHospitalSummarization):
                llm_judge_score_value = await llm_judge_score(diagnosis_text, correct_diagnosis, cost_tracker)
                is_correct = False  # we don't use binary correctness here
                print("\nScene {} - Reference summary:".format(scenario_id), correct_diagnosis)
                print("Scene {} - MAS summary:".format(scenario_id), diagnosis_text)
                print("Scene {} - LLM Judge Score: {:.4f}".format(scenario_id, llm_judge_score_value))
            else:
                # For other datasets, we don't evaluate correctness without moderator
                # This code path should not be used for HospSumm dataset
                is_correct = False
                print("\nScene {} - Correct answer:".format(scenario_id), correct_diagnosis)
                print("Scene {} - Doctor diagnosis:".format(scenario_id), diagnosis_text)
                print("Scene {} - Note: Correctness evaluation not available without moderator".format(scenario_id))
            
            # Print cost summary for this scenario
            cost_summary = cost_tracker.get_summary()
            print("\n" + "="*60)
            print(f"Scene {scenario_id} - Cost Summary:")
            print("="*60)
            print(f"Doctor Agent:")
            print(f"  - Calls: {cost_summary['doctor']['num_calls']}")
            print(f"  - Input tokens: {cost_summary['doctor']['input_tokens']}")
            print(f"  - Output tokens: {cost_summary['doctor']['output_tokens']}")
            print(f"  - Total tokens: {cost_summary['doctor']['total_tokens']}")
            print(f"  - Cost: ${cost_summary['doctor']['total_cost']:.6f}")
            print(f"Patient Agent:")
            print(f"  - Calls: {cost_summary['patient']['num_calls']}")
            print(f"  - Input tokens: {cost_summary['patient']['input_tokens']}")
            print(f"  - Output tokens: {cost_summary['patient']['output_tokens']}")
            print(f"  - Total tokens: {cost_summary['patient']['total_tokens']}")
            print(f"  - Cost: ${cost_summary['patient']['total_cost']:.6f}")
            print(f"Measurement Agent:")
            print(f"  - Calls: {cost_summary['measurement']['num_calls']}")
            print(f"  - Input tokens: {cost_summary['measurement']['input_tokens']}")
            print(f"  - Output tokens: {cost_summary['measurement']['output_tokens']}")
            print(f"  - Total tokens: {cost_summary['measurement']['total_tokens']}")
            print(f"  - Cost: ${cost_summary['measurement']['total_cost']:.6f}")
            print(f"Judge Agent:")
            print(f"  - Calls: {cost_summary['judge']['num_calls']}")
            print(f"  - Input tokens: {cost_summary['judge']['input_tokens']}")
            print(f"  - Output tokens: {cost_summary['judge']['output_tokens']}")
            print(f"  - Total tokens: {cost_summary['judge']['total_tokens']}")
            print(f"  - Cost: ${cost_summary['judge']['total_cost']:.6f}")
            print(f"Total Cost: ${cost_summary['total']['total_cost']:.6f}")
            print(f"Total Tokens: {cost_summary['total']['total_tokens']}")
            print("="*60 + "\n")
            
            # Save results to file if output_dir is provided
            if output_dir:
                # Run file operations in thread to avoid blocking
                await asyncio.to_thread(
                    save_scenario_results,
                    scenario_id, scenario, dialogue_history, diagnosis_text, 
                    correct_diagnosis, is_correct, cost_tracker, output_dir, llm_judge_score_value
                )
            
            # Return diagnosis result with full information
            return {
                "scenario_id": scenario_id,
                "is_correct": is_correct,
                "diagnosis_made": True,
                "doctor_diagnosis": diagnosis_text,
                "correct_diagnosis": correct_diagnosis,
                "diagnosis_status": "correct" if is_correct else "incorrect",
                "llm_judge_score": llm_judge_score_value
            }, cost_tracker
        
        # Obtain medical exam from measurement reader
        if "REQUEST TEST" in doctor_dialogue:
            pi_dialogue = await meas_agent.inference_measurement(doctor_dialogue, cost_tracker=cost_tracker)
            print("Scene {} - Measurement [{}%]:".format(scenario_id, int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
            patient_agent.add_hist(pi_dialogue)
            # Record measurement result
            dialogue_history.append({
                "round": _inf_id + 1,
                "type": "measurement",
                "content": pi_dialogue,
                "progress": int(((_inf_id+1)/total_inferences)*100)
            })
        # Obtain response from patient
        else:
            if inf_type == "human_patient":
                pi_dialogue = input("\nResponse to doctor: ")
            else:
                pi_dialogue = await patient_agent.inference_patient(doctor_dialogue, cost_tracker=cost_tracker)
            print("Scene {} - Patient [{}%]:".format(scenario_id, int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
            meas_agent.add_hist(pi_dialogue)
            # Record patient response
            dialogue_history.append({
                "round": _inf_id + 1,
                "type": "patient",
                "content": pi_dialogue,
                "progress": int(((_inf_id+1)/total_inferences)*100)
            })
        
        # Small delay to prevent API rate limits (can be reduced or removed with proper concurrency control)
        await asyncio.sleep(0.1)
    
    # If reached max inferences without diagnosis, consider it incorrect
    diagnosis_info = scenario.diagnosis_information()
    correct_diagnosis = diagnosis_info if isinstance(diagnosis_info, str) else str(diagnosis_info)
    is_correct = False  # No diagnosis/summary made, treat as incorrect / score 0
    
    # Print cost summary even if no diagnosis was made
    cost_summary = cost_tracker.get_summary()
    print("\n" + "="*60)
    print(f"Scene {scenario_id} - Cost Summary (No Diagnosis):")
    print("="*60)
    print(f"Doctor Agent: ${cost_summary['doctor']['total_cost']:.6f} ({cost_summary['doctor']['num_calls']} calls)")
    print(f"Patient Agent: ${cost_summary['patient']['total_cost']:.6f} ({cost_summary['patient']['num_calls']} calls)")
    print(f"Measurement Agent: ${cost_summary['measurement']['total_cost']:.6f} ({cost_summary['measurement']['num_calls']} calls)")
    print(f"Judge Agent: ${cost_summary['judge']['total_cost']:.6f} ({cost_summary['judge']['num_calls']} calls)")
    print(f"Total Cost: ${cost_summary['total']['total_cost']:.6f}")
    print("="*60 + "\n")
    
    # Save results to file if output_dir is provided (even if no diagnosis)
    if output_dir:
        # Run file operations in thread to avoid blocking
        no_diagnosis_llm_judge_score = 0.0 if isinstance(scenario, ScenarioHospitalSummarization) else None
        await asyncio.to_thread(
            save_scenario_results,
            scenario_id, scenario, dialogue_history, diagnosis_text, 
            correct_diagnosis, is_correct, cost_tracker, output_dir, no_diagnosis_llm_judge_score
        )
    
    # Return diagnosis result with full information
    return {
        "scenario_id": scenario_id,
        "is_correct": False,  # No diagnosis made, so it's incorrect
        "diagnosis_made": False,
        "doctor_diagnosis": None,
        "correct_diagnosis": correct_diagnosis,
        "diagnosis_status": "no_diagnosis",
        "llm_judge_score": 0.0 if isinstance(scenario, ScenarioHospitalSummarization) else None
    }, cost_tracker


async def main_async(api_key, inf_type, doctor_bias, patient_bias, 
                     doctor_llm, patient_llm, measurement_llm, 
                     num_scenarios, dataset, img_request, total_inferences, 
                     min_infs=10, max_concurrent=10, output_dir=None, base_url=None):
    """Async main function with concurrency control
    
    Args:
        max_concurrent: Maximum number of scenarios to process concurrently
        API calls within each scenario are limited to 10 concurrent calls
    """
    # Initialize async clients with fixed API-level concurrency of 10
    # This controls how many API calls can run concurrently within each scenario
    init_clients(api_key=api_key, max_concurrent=10, base_url=base_url)
    
    # Create scenario-level semaphore to control how many scenarios run concurrently
    # This is controlled by the --max_concurrent parameter
    scenario_semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Load MedQA, MIMICIV, NEJM or Hospital Summarization agent case scenarios
    if dataset == "HospSumm":
        scenario_loader = ScenarioLoaderHospitalSummarization()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    
    total_correct = 0
    total_presents = 0
    llm_judge_scores = []

    if num_scenarios is None: 
        num_scenarios = scenario_loader.num_scenarios
    
    # Create tasks for all scenarios
    scenario_ids = list(range(0, min(num_scenarios, scenario_loader.num_scenarios)))
    
    # Process scenarios with concurrency control
    # Use semaphore to limit how many scenarios run concurrently
    tasks = []
    for scenario_id in scenario_ids:
        task = process_single_scenario(
            scenario_id, scenario_loader, inf_type, doctor_bias, patient_bias,
            doctor_llm, patient_llm, measurement_llm,
            dataset, img_request, total_inferences, min_infs, output_dir=output_dir,
            scenario_semaphore=scenario_semaphore
        )
        tasks.append(task)
    
    # Execute tasks with progress bar
    results = []
    cost_trackers = []
    scenario_results = []  # Store detailed results for each scenario
    
    # Store scenario diagnosis results for summary
    scenario_diagnosis_results = {}  # scenario_id -> {diagnosis_text, correct_diagnosis, is_correct}
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing scenarios"):
        diagnosis_result, cost_tracker = await coro
        results.append(diagnosis_result["is_correct"])
        cost_trackers.append(cost_tracker)
        
        # Store detailed scenario result
        scenario_result = {
            "scenario_id": diagnosis_result["scenario_id"],
            "correct": diagnosis_result["is_correct"],  # Boolean: True if correct, False if incorrect or no diagnosis
            "is_correct": diagnosis_result["is_correct"],
            "diagnosis_made": diagnosis_result["diagnosis_made"],
            "doctor_diagnosis": diagnosis_result["doctor_diagnosis"],
            "correct_diagnosis": diagnosis_result["correct_diagnosis"],
            "diagnosis_status": diagnosis_result["diagnosis_status"],
            "llm_judge_score": diagnosis_result.get("llm_judge_score"),
            "cost_summary": cost_tracker.get_summary()
        }
        scenario_results.append(scenario_result)
        
        total_presents += 1
        if diagnosis_result["is_correct"]:
            total_correct += 1
        if dataset == "HospSumm" and diagnosis_result.get("llm_judge_score") is not None:
            llm_judge_scores.append(diagnosis_result["llm_judge_score"])
        
        status_msg = diagnosis_result["diagnosis_status"].upper()
        print(f"\nScene {diagnosis_result['scenario_id']}: {status_msg}")
        if dataset != "HospSumm":
            print(f"Progress: {total_correct}/{total_presents} correct ({int((total_correct/total_presents)*100)}%)")
        else:
            avg_llm_judge = sum(llm_judge_scores) / len(llm_judge_scores) if llm_judge_scores else 0.0
            print(f"Progress: {len(llm_judge_scores)}/{total_presents} scored, current avg LLM Judge Score: {avg_llm_judge:.4f}")
    
    # Calculate total costs across all scenarios
    total_doctor_cost = sum(ct.doctor_cost.total_cost for ct in cost_trackers)
    total_patient_cost = sum(ct.patient_cost.total_cost for ct in cost_trackers)
    total_measurement_cost = sum(ct.measurement_cost.total_cost for ct in cost_trackers)
    total_judge_cost = sum(ct.judge_cost.total_cost for ct in cost_trackers)
    total_all_cost = total_doctor_cost + total_patient_cost + total_measurement_cost + total_judge_cost
    
    # Calculate total tokens
    total_doctor_tokens = sum(ct.doctor_cost.total_tokens for ct in cost_trackers)
    total_patient_tokens = sum(ct.patient_cost.total_tokens for ct in cost_trackers)
    total_measurement_tokens = sum(ct.measurement_cost.total_tokens for ct in cost_trackers)
    total_judge_tokens = sum(ct.judge_cost.total_tokens for ct in cost_trackers)
    total_all_tokens = total_doctor_tokens + total_patient_tokens + total_measurement_tokens + total_judge_tokens
    
    print(f"\n{'='*60}")
    print(f"=== Final Results ===")
    print(f"{'='*60}")
    print(f"Total scenarios: {total_presents}")
    if dataset == "HospSumm":
        avg_llm_judge = sum(llm_judge_scores) / len(llm_judge_scores) if llm_judge_scores else 0.0
        print(f"Average LLM Judge Score (MAS summary vs reference): {avg_llm_judge:.4f}")
    else:
        print(f"Correct diagnoses: {total_correct}")
        print(f"Accuracy: {int((total_correct/total_presents)*100)}%")
    print(f"\n=== Total Cost Summary (All Scenarios) ===")
    print(f"Doctor Agent Total Cost: ${total_doctor_cost:.6f}")
    print(f"Patient Agent Total Cost: ${total_patient_cost:.6f}")
    print(f"Measurement Agent Total Cost: ${total_measurement_cost:.6f}")
    print(f"Judge Agent Total Cost: ${total_judge_cost:.6f}")
    print(f"Total Cost: ${total_all_cost:.6f}")
    print(f"Total Tokens: {total_all_tokens}")
    print(f"{'='*60}\n")
    
    # Save summary to file if output_dir is provided
    if output_dir:
        summary_file = os.path.join(output_dir, "summary.json")
        summary = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "dataset": dataset,
                "doctor_llm": doctor_llm,
                "patient_llm": patient_llm,
                "measurement_llm": measurement_llm,
                "doctor_bias": doctor_bias,
                "patient_bias": patient_bias,
                "total_inferences": total_inferences,
                "num_scenarios": num_scenarios,
                "max_concurrent": max_concurrent
            },
            "results": {
                "total_scenarios": total_presents,
                "correct_diagnoses": total_correct if dataset != "HospSumm" else None,
                "incorrect_diagnoses": (total_presents - total_correct) if dataset != "HospSumm" else None,
                "accuracy": round((total_correct/total_presents)*100, 2) if (total_presents > 0 and dataset != "HospSumm") else None,
                "accuracy_percentage": (f"{round((total_correct/total_presents)*100, 2)}%" if (total_presents > 0 and dataset != "HospSumm") else None),
                "average_llm_judge_score": (round(sum(llm_judge_scores) / len(llm_judge_scores), 4) if (dataset == "HospSumm" and llm_judge_scores) else None)
            },
            "scenario_results": scenario_results,  # Detailed results for each scenario including correctness / LLM Judge Score
            "costs": {
                "doctor": {
                    "total_cost": round(total_doctor_cost, 6),
                    "total_tokens": total_doctor_tokens,
                    "average_cost_per_scenario": round(total_doctor_cost/total_presents, 6) if total_presents > 0 else 0
                },
                "patient": {
                    "total_cost": round(total_patient_cost, 6),
                    "total_tokens": total_patient_tokens,
                    "average_cost_per_scenario": round(total_patient_cost/total_presents, 6) if total_presents > 0 else 0
                },
                "measurement": {
                    "total_cost": round(total_measurement_cost, 6),
                    "total_tokens": total_measurement_tokens,
                    "average_cost_per_scenario": round(total_measurement_cost/total_presents, 6) if total_presents > 0 else 0
                },
                "judge": {
                    "total_cost": round(total_judge_cost, 6),
                    "total_tokens": total_judge_tokens,
                    "average_cost_per_scenario": round(total_judge_cost/total_presents, 6) if total_presents > 0 else 0
                },
                "total": {
                    "total_cost": round(total_all_cost, 6),
                    "total_tokens": total_all_tokens,
                    "average_cost_per_scenario": round(total_all_cost/total_presents, 6) if total_presents > 0 else 0
                }
            },
            "scenario_results": scenario_results
        }
        # Save summary file (run in thread to avoid blocking)
        def save_summary():
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        await asyncio.to_thread(save_summary)
        print(f"Summary saved to: {summary_file}")
    
    return total_correct, total_presents, cost_trackers


def main(api_key, inf_type, doctor_bias, patient_bias, 
         doctor_llm, patient_llm, measurement_llm, 
         num_scenarios, dataset, img_request, total_inferences, 
         min_infs=10, max_concurrent=10, output_dir=None, base_url=None):
    """Synchronous wrapper for async main function"""
    return asyncio.run(main_async(
        api_key, inf_type, doctor_bias, patient_bias,
        doctor_llm, patient_llm, measurement_llm,
        num_scenarios, dataset, img_request, total_inferences,
        min_infs, max_concurrent, output_dir, base_url
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--base_url', type=str, required=True, help='OpenAI API Base URL')    
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='gpt-5-mini')
    parser.add_argument('--patient_llm', type=str, default='gpt-5-mini')
    parser.add_argument('--measurement_llm', type=str, default='gpt-5-mini')
    parser.add_argument('--agent_dataset', type=str, default='HospSumm') # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=40, required=False, help='Number of inferences between patient and doctor')
    parser.add_argument('--min_inferences', type=int, default=10, required=False, help='Minimum number of inference rounds before doctor can output summary (default: 10)')
    parser.add_argument('--max_concurrent', type=int, default=50, required=False, help='Maximum number of scenarios to process concurrently (default: 1). API calls within each scenario are limited to 10 concurrent calls.')
    parser.add_argument('--output_dir', type=str, default='./async_results/', required=False, help='Output directory to save results (default: None, no saving)')
    
    args = parser.parse_args()
    print(args)

    main(args.openai_api_key, args.inf_type, args.doctor_bias, args.patient_bias, 
         args.doctor_llm, args.patient_llm, args.measurement_llm, 
         args.num_scenarios, args.agent_dataset, args.doctor_image_request, args.total_inferences, 
         args.min_inferences, args.max_concurrent, args.output_dir, args.base_url)
