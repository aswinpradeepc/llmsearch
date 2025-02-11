from fastapi import FastAPI
from .search import router as search_router

app = FastAPI()

# Include the search router
app.include_router(search_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Financial Data Retrieval System API"}