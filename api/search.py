from fastapi import APIRouter, HTTPException
from typing import List, Dict
from .utils import parse_query, filter_metadata, search_in_pinecone
from .models import QueryResponse

router = APIRouter()

@router.post("/search", response_model=QueryResponse)
async def search(query: str):
    try:
        # Step 1: Parse the query to extract keywords, context, and metrics
        parsed_query = parse_query(query)
        
        # Step 2: Perform semantic search using Pinecone
        semantic_results = search_in_pinecone(parsed_query["embedding"])
        
        # Step 3: Apply keyword matching and metadata filters
        filtered_results = filter_metadata(semantic_results, parsed_query)
        
        # Step 4: Return the search results
        return QueryResponse(results=filtered_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

