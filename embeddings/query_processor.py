import openai
from typing import List

# Function to generate query embedding using OpenAI's new API
def generate_query_embedding(query: str) -> List[float]:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query]  # The new API expects a list of inputs
    )
    return response['data'][0]['embedding']
