__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
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

# Step 1: Define a dictionary of question-answer pairs.
expected_responses = {
    "How much total money does a player start with in classic Monopoly? (Answer with the number only)": "$1500",
    "How many points win the game in Uno? (Answer with the number only)": "500 points",
    "How many times can you roll the dice for a turn in Yahtzee? (Answer with the number only)": "3 times",
    # Add more question-answer pairs as needed
}

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

def query_rag(query_text: str, embedding_type: str):
    """Function to query the model and evaluate the response."""
    try:
        # Prepare the embedding function and database.
        embedding_function = get_embedding_function(embedding_type)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        if not results:
            print("No results found.")
            return None

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Prepare the prompt.
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Invoke the model using the selected embedding type.
        model = OllamaLLM(model=embedding_type)
        response_text = model.invoke(prompt)

        # Get expected response based on query.
        expected_response = expected_responses.get(query_text, "No expected response available.")
        
        # Evaluate the response.
        evaluation_prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=response_text)
        evaluation_results_str = model.invoke(evaluation_prompt).strip().lower()

        # Output the evaluation results.
        if "true" in evaluation_results_str:
            print("\033[92m" + f"Response: {evaluation_results_str}" + "\033[0m")
        elif "false" in evaluation_results_str:
            print("\033[91m" + f"Response: {evaluation_results_str}" + "\033[0m")
        else:
            raise ValueError("Invalid evaluation result. Cannot determine if 'true' or 'false'.")

        # Extract sources from results.
        sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return response_text

    except Exception as e:
        print(f"Error querying the model: {e}")
        return None

if __name__ == "__main__":
    main()
