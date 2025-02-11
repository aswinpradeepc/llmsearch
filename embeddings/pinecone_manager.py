import pinecone

# Initialize Pinecone environment and index
pinecone.init(api_key='your-pinecone-api-key', environment='us-west1-gcp')
index_name = 'financial-search-index'
index = pinecone.Index(index_name)

def insert_embeddings(index_name: str, embeddings: List[tuple]):
    """
    Inserts embeddings into Pinecone index.

    Args:
    - index_name (str): Pinecone index name.
    - embeddings (List[tuple]): A list of (id, embedding) pairs to be inserted into Pinecone.
    """
    index.upsert(vectors=embeddings)

def query_embeddings(query_embedding: List[float], top_k: int = 5) -> List[dict]:
    """
    Queries Pinecone for the most relevant embeddings based on a query embedding.

    Args:
    - query_embedding (List[float]): The query embedding to search for.
    - top_k (int): Number of top results to return.

    Returns:
    - List[dict]: The list of matching results.
    """
    result = index.query(
        vector=query_embedding, 
        top_k=top_k,
        include_metadata=True
    )
    return result['matches']
