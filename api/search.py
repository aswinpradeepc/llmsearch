import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from api.utils import perform_search
from api.response_generator import generate_response

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    query: str
    response: str
    sources: List[str]

@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    query = request.query
    try:
        logger.info(f"Received query: {query}")
        
        # Perform the search to get context
        context, sources = perform_search(query)
        logger.info(f"Search context: {context}")
        logger.info(f"Search sources: {sources}")
        
        # Generate the response using GPT-4 Turbo
        response = generate_response(query, context)
        logger.info(f"Generated response: {response}")
        
        return SearchResponse(query=query, response=response, sources=sources)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

