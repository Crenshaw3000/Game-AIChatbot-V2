import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # new

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--embedding-type", type=str, choices=["mistral"], default="mistral",
                        help="Type of embedding to use: mistral.")
    args = parser.parse_args()

    query_text = args.query_text
    embedding_type = args.embedding_type
    
    # Call the query_rag function with the provided inputs
    query_rag(query_text, embedding_type)

def query_rag(question: str, embedding_type: str):
    """Function to query the model and evaluate the response."""
    try:
        # Assuming get_embedding_function is defined and returns a function to get embeddings
        embedding_function = get_embedding_function(embedding_type)
        
        # Example of querying the model. Modify based on how you interact with your data.
        # response_text = some_model.query(question, embedding_function)
        
        # Here, replace with actual query logic
        response_text = "Example response based on the question."  # Placeholder response

        # For demonstration purposes, we'll define the expected response here.
        expected_response = "$1500"  # Replace with the actual expected response logic if needed.

        # Proceed with evaluation
        prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=response_text)

        model = OllamaLLM(model="mistral")
        evaluation_results_str = model.invoke(prompt)
        
        evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

        print(prompt)

        if "true" in evaluation_results_str_cleaned:
            print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
            return True
        elif "false" in evaluation_results_str_cleaned:
            print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
            return False
        else:
            raise ValueError("Invalid evaluation result. Cannot determine if 'true' or 'false'.")
    
    except Exception as e:
        print(f"Error querying the model: {e}")
        return False  # Return False if there is an error

if __name__ == "__main__":
    main()
