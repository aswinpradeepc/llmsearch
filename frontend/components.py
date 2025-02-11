import streamlit as st

def display_response(response: str, sources: list):
    """
    Display the response and sources in the Streamlit app.
    
    :param response: The generated response from the API.
    :param sources: List of sources for the response.
    """
    st.write("### Response")
    st.write(response)
    
    st.write("### Sources")
    for source in sources:
        st.write(f"- {source}")