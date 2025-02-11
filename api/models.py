from pydantic import BaseModel
from typing import List, Dict

class QueryResponse(BaseModel):
    results: List[Dict]  # List of results with metadata and relevant financial data
