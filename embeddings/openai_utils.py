import openai

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
