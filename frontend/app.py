import streamlit as st
import requests
from components import display_response

# Streamlit app configuration
st.set_page_config(page_title="Financial Data Retrieval System", layout="wide")

# Title and description
st.title("Financial Data Retrieval System")
st.write("Interact with company reports, mutual fund documents, and market research files using natural language queries.")

# Input for user query
query = st.text_input("Enter your query:")

# Button to submit the query
if st.button("Search"):
    if query:
        # Call the API with the user query
        response = requests.post("http://api:8000/rag", json={"query": query})
        
        if response.status_code == 200:
            data = response.json()
            display_response(data["answer"], data["sources"])
        else:
            st.error("Error: Unable to retrieve data. Please try again later.")
    else:
        st.warning("Please enter a query.")