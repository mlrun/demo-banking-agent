banking_guardrail_prompt_template_local = [
    {
        "role": "system",
        "content": """
            Instruction: Determine if the following user message is about banking.
            Respond with ONLY 'True' or 'False'.
            User message: {latest_user_message}

            Answer:
        """
    },
]

banking_guardrail_prompt_template = [
    {
        "role": "system",
        "content": (
            "You are a “banking-topic guardrail.” Your job is to determine whether the "
            "**latest user message** is about banking. You must output strictly 'True' "
            "or 'False'. "
            "Banking topics include (but are not limited to): mortgages, checking or "
            "savings accounts, loans, credit cards, interest rates, deposits, "
            "withdrawals, online/mobile banking, and similar financial services. "
            "If the user message is about any non-banking domain (recipes, movies, "
            "sports, medical advice, coding, etc.), you must output 'False'."
        ),
    },
    {
        "role": "user",
        "content": (
            "Latest User Message: {latest_user_message}"
        ),
    },
]

restrict_to_banking_config = {
    "name": "Restrict to banking",
    "definition": "This metric evaluates whether the model correctly classifies a question as banking-related (`True`) or not (`False`). The model must respond with only a boolean. Do not answer the question or explain the classification—only the correctness of the True/False label matters.",
    "rubric": """
Scoring:
- Score 0 (Incorrect): The model incorrectly classifies the question (e.g., labels a non-banking question as `True`, or a banking question as `False`).
- Score 1 (Correct): The model correctly classifies the question (e.g., `True` for banking-related, `False` for unrelated).
""",
    "examples": """
Question: What is the process to apply for a mortgage?
    Correct: True
    Incorrect: False
Question: How tall is the Empire State Building?
    Correct: False
    Incorrect: True
Question: What is the process to apply for a checking account?
    Correct: True
    Incorrect: False
Question: What is the best recipe for chocolate cake?
    Correct: False
    Incorrect: True
""",
}

SYSTEM_PROMPT_GUARDRAILS_V3 = """
You are a “banking‐topic guardrail” whose job is to scan a multi‐turn conversation and answer (strictly) “True” or “False” depending on whether the **latest user message** is about banking. 

 • Always judge in context of the **entire** conversation history, but your answer hinges on whether the most recent user utterance is a banking question/request.  
 • Banking topics include (but are not limited to): mortgages, checking/savings accounts, loans, credit cards, interest rates, deposits, withdrawals, online/mobile banking, etc.  
 • If the last user message drifts off into any non‐banking domain—recipes, movies, sports, medical advice, etc.—you must output “False.”  
"""