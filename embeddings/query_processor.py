import openai

# Function to generate query embedding using OpenAI's Ada-002 model
def generate_query_embedding(query: str) -> List[float]:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    return response['data'][0]['embedding']
