import openai
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(query: str, context: List[str]) -> str:
    """
    Generate a response using GPT-4 Turbo based on the query and context.
    
    :param query: The user's query.
    :param context: List of context strings retrieved from the search.
    :return: Generated response string.
    """
    prompt = f"Answer the following query based on the provided context:\n\nQuery: {query}\n\nContext:\n" + "\n".join(context)
    
    response = openai.Completion.create(
        engine="gpt-4-turbo",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()