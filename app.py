import streamlit as st
from query_data import query_rag  # Assuming query_rag uses get_embedding_function internally

# Title of the app
st.title("Game Rules Chatbot")

# Input field for user query
query_text = st.text_input("Ask a question about Monopoly, Uno, or Yahtzee rules:")

# Select embedding type
embedding_type = st.selectbox("Select embedding type:", ["mistral"], index=0)

# Button to submit the query
if st.button("Get Answer"):
    if query_text:
        # Call the query_rag function and get the response
        response = query_rag(query_text, embedding_type)
        st.write("Response:", response)
    else:
        st.write("Please enter a question.")
