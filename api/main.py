from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Financial Document Search API")

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index = pinecone_client.Index("financial-search-index")

# Models
class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5

class Metadata(BaseModel):
    filename: str
    content: str  # Make content required, not optional

class Match(BaseModel):
    id: str
    score: float
    metadata: Metadata

class SearchResponse(BaseModel):
    matches: List[Match]
    query_embedding: List[float]

class RAGQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RAGResponse(BaseModel):
    answer: str
    sources: List[Match]

# Helper functions
def generate_embedding(text: str) -> List[float]:
    """Generate embedding for the input text."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

def format_matches(results: Dict[str, Any]) -> List[Match]:
    """Format Pinecone results to match Pydantic model."""
    matches = []
    for match in results.get("matches", []):
        matches.append(Match(
            id=match["id"],
            score=match["score"],
            metadata=Metadata(**match["metadata"])
        ))
    return matches

def get_rag_response(query: str, context: List[str]) -> str:
    try:
        # Join contexts with proper truncation
        combined_context = " ".join(context)
        if len(combined_context) > 15000:
            combined_context = combined_context[:15000] + "..."

        prompt = f"""Based on the following context, answer the question.
        If the answer cannot be found in the context, say "I cannot answer this based on the available information."
        
        Context:
        {combined_context}
        
        Question: {query}
        
        Answer:"""

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

# Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/search", response_model=SearchResponse)
async def semantic_search(query: SearchQuery):
    """Perform semantic search on documents."""
    try:
        query_embedding = generate_embedding(query.query)
        results = index.query(
            vector=query_embedding,
            top_k=query.top_k,
            include_metadata=True
        )
        
        return SearchResponse(
            matches=format_matches(results),
            query_embedding=query_embedding
        )
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag", response_model=RAGResponse)
async def rag_search(query: RAGQuery):
    """Perform RAG-based search and response."""
    try:
        query_embedding = generate_embedding(query.query)
        results = index.query(
            vector=query_embedding,
            top_k=query.top_k,
            include_metadata=True
        )
        
        matches = format_matches(results)
        contexts = [match.metadata.content for match in matches if match.metadata.content]
        
        answer = get_rag_response(query.query, contexts)
        
        return RAGResponse(
            answer=answer,
            sources=matches
        )
    except Exception as e:
        logger.error(f"Error in RAG search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)