from query_data import query_rag
from langchain_ollama import OllamaLLM

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question: str, expected_response: str, embedding_type: str):
    response_text = query_rag(question, embedding_type)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = OllamaLLM(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)  # Debug print to check the prompt

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in classic Monopoly? (Answer with the number only)",
        expected_response="$1500",
        embedding_type="ollama"
    )

def test_uno_rules():
    assert query_and_validate(
        question="How many points win the game in Uno? (Answer with the number only)",
        expected_response="500 points",
        embedding_type="ollama"
    )

def test_yahtzee_rules():
    assert query_and_validate(
        question="How many times can you roll the dice for a turn in Yahtzee? (Answer with the number only)",
        expected_response="3 times",
        embedding_type="ollama"
    )


def query_and_validate(question: str, expected_response: str, embedding_type: str):
    response_text = query_rag(question, embedding_type)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = OllamaLLM(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
    



