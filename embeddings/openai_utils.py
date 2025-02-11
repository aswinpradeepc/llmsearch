import openai
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_embeddings(text: str) -> List[float]:
    """
    Generates embeddings using OpenAI's Ada-002 model for a given text.

    Args:
    - text (str): The text to generate embeddings for.

    Returns:
    - List[float]: The embedding vector for the text.
    """
    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=[text]  # The new API expects a list of inputs
    )
    return response['data'][0]['embedding']
