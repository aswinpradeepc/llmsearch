import openai
import json
from typing import List
from pinecone_manager import insert_embeddings

# Initialize OpenAI API key
openai.api_key = 'your-openai-api-key'

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
        input=text
    )
    return response['data'][0]['embedding']

def process_and_store_embeddings(text_data: List[dict], index_name: str):
    """
    Process text data and generate embeddings, then store them in Pinecone.

    Args:
    - text_data (List[dict]): List of dictionaries containing 'id' and 'text'.
    - index_name (str): Pinecone index name where embeddings will be stored.
    """
    embeddings = []
    for data in text_data:
        embedding = generate_embeddings(data['text'])
        embeddings.append((data['id'], embedding))

    insert_embeddings(index_name, embeddings)
    print(f"Embeddings for {len(text_data)} documents inserted into Pinecone.")
