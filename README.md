# Game Rules AI Chatbot<br />

The Game Rules AI Chatbot is designed to provide specialized knowledge 
on the rules of Monopoly, Uno, and Yahtzee. Users can ask it questions about 
these games, and the chatbot will retrieve accurate answers. It leverages PDFs as 
information sources and employs Retrieval Augmented Generation (RAG) using the 
powerful large language model (LLM), Ollama, to process and respond to queries effectively.



![Game Rules Chatbot](images/gameAIbot.gif "Game Rules Chatbot") <br />

## <a name="technologies"></a> Technologies
* Python
* Ollama
* Chroma
* LangChain
* pypdf
* Streamlit


## Features & Architecture

This project implements a Retrieval Augmented Generation (RAG) chatbot that provides accurate, source-backed answers about the rules of **Monopoly, Uno, and Yahtzee**. The system leverages local PDF documents as its knowledge base and uses a robust pipeline to ensure responses are grounded in official game rules rather than LLM training data.

The system consists of five key components:

1. **PDF Processing (pypdf):** Extracts text from game rule PDFs while preserving document structure, page numbers, and source filenames.
2. **Vector Database (Chroma):** Stores and indexes text chunks as high-dimensional embeddings for fast semantic search.
3. **RAG Pipeline (LangChain):** Orchestrates document retrieval and LLM inference for grounded answers.
4. **LLM Integration (Ollama + Mistral):** Generates natural language responses locally, ensuring privacy and adherence to PDF content.
5. **Web Interface (Streamlit):** Provides a mobile-friendly chat interface for interactive question-answering.

## Technical Workflow

- **Document Ingestion:** PDFs are split into 800-character chunks with 80-character overlaps, converted to 1536-dimensional embeddings, and stored in Chroma with unique identifiers (`source:page:chunk_index`).
- **Query Processing:** User questions are converted to embeddings, compared against the vector database via cosine similarity, and top relevant chunks are combined into a structured context prompt for the LLM.
- **RAG Flow:** User Question → Embeddings → Vector Search → Context Build → Prompt Engineering → LLM Response → Self-Evaluation → Source Attribution.

## Quality Assurance & Validation

- **Prompt Engineering:** Templates constrain the model to use only PDF context.
  ```python
  PROMPT_TEMPLATE = """
  Answer the question based only on the following context:
  {context}
  ---
  Answer the question based on the above context: {question}
  """
  ```
- **Automated Evaluation:** Ground-truth Q&A pairs are used to verify generated answers.
  ```python
  EVAL_PROMPT = """
  Expected Response: {expected_response}
  Actual Response: {actual_response}
  ---
  (Answer with 'true' or 'false') Does the actual response match the expected response? 
  """
  ```
- **Smart Testing:** Includes edge cases such as Monopoly starting money, Uno point totals, and Yahtzee dice rolls.
  ```python
  expected_responses = {
    "How much total money does a player start with in classic Monopoly? (Answer with the number only)": "$1500",
    "How many points win the game in Uno? (Answer with the number only)": "500 points",
    "How many times can you roll the dice for a turn in Yahtzee? (Answer with the number only)": "3 times",} ```
- **Traceability:** Every answer cites PDF pages and chunk indices for verification.

## Deployment & Privacy

The system runs fully locally with Ollama handling both embeddings and text generation, ensuring complete privacy and eliminating reliance on external APIs. Streamlit provides a mobile-accessible web interface for user interactions.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up Ollama with Mistral model
3. Add PDF rule books to the `data/` folder
4. Run `python populate_database.py` to build the vector database
5. Launch the interface: `streamlit run streamlit_app.py`


### Game Rules Chatbot<br />
![Game Rules Chatbot](images/Game_chatbot.png "Game Rules Chatbot") <br/>


### Question Answered by Games Rules<br />
![Question Answered by Chatbot](images/game_chatbotQ1.png "Question Answered by Chatbot") <br />

