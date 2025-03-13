# Financial Document Search & Retrieval System

## Overview
The **Financial Document Search & Retrieval System** is an AI-powered search tool that allows users to query financial documents (such as company reports, mutual fund documents, and market research reports) using natural language. It leverages **OpenAI embeddings** and **Pinecone vector search** to provide accurate and contextually relevant results.

## Features
- **Semantic Search**: Uses OpenAI's `text-embedding-ada-002` model to generate document embeddings for similarity-based retrieval.
- **Retrieval-Augmented Generation (RAG)**: Combines search results with GPT-4 to generate concise, well-informed answers.
- **API Backend**: Built with FastAPI to handle document search and retrieval requests.
- **Frontend Interface**: Uses Streamlit for an interactive UI to submit queries and display results.
- **PDF Processing**: Extracts text and tables from PDF files using `pdfplumber` and `PyMuPDF`.
- **Containerized Deployment**: Managed using Docker and `docker-compose` for easy scalability.

## Directory Structure
```
├── aswinpradeepc-llmsearch/
│   ├── docker-compose.yml  # Docker Compose file for multi-container setup
│   ├── Dockerfile          # Defines the API and frontend container
│   ├── dockerignore        # Files to ignore during Docker build
│   ├── requirements.txt    # Python dependencies
│   ├── main.py             # Main entry point (if required)
│   ├── api/                # Backend API
│   │   └── main.py         # FastAPI server implementation
│   ├── frontend/           # Streamlit frontend
│   │   ├── app.py          # Main Streamlit application
│   │   └── components.py   # Helper functions for UI rendering
│   ├── scripts/            # Preprocessing scripts
│   │   ├── generate_embeddings.py # Converts documents into vector embeddings
│   │   └── pdf_to_json.py         # Extracts text and tables from PDFs
```

## Setup & Installation
### Prerequisites
Ensure you have the following installed:
- **Python 3.12+**
- **Docker & Docker Compose**
- **An OpenAI API Key**
- **A Pinecone API Key**

### Environment Variables
Create a `.env` file and define the following variables:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_KEY=your_pinecone_api_key
```

### Running the Application
#### 1. Clone the repository
```sh
git clone https://github.com/aswinpradeepc/llmsearch.git
cd llmsearch
```

#### 2. Build and run the containers
```sh
docker-compose up --build
```

#### 3. Access the application
- **Backend API**: Runs on `http://localhost:8000`
- **Frontend UI**: Accessible at `http://localhost:8501`

## API Endpoints
### Health Check
```http
GET /health
```
Response:
```json
{"status": "healthy"}
```

### Semantic Search
```http
POST /search
```
#### Request Body:
```json
{
  "query": "Top mutual funds in 2023",
  "top_k": 5
}
```
#### Response:
```json
{
  "matches": [
    {
      "id": "doc123",
      "score": 0.95,
      "metadata": {
        "filename": "funds_report.pdf",
        "content": "Best performing funds in 2023...",
        "tables": "[Table data here]"
      }
    }
  ]
}
```

### Retrieval-Augmented Generation (RAG)
```http
POST /rag
```
#### Request Body:
```json
{
  "query": "What are the key insights from XYZ report?",
  "top_k": 3
}
```
#### Response:
```json
{
  "answer": "XYZ report highlights revenue growth of 10%...",
  "sources": [
    {
      "id": "doc456",
      "metadata": {
        "filename": "XYZ_report.pdf",
        "content": "Revenue increased by 10%..."
      }
    }
  ]
}
```

## Development & Contribution
### Running Locally
#### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

#### 2. Start API
```sh
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Start Frontend
```sh
streamlit run frontend/app.py --server.port 8501
```

### Modifying the Code
- **Backend (FastAPI)**: Modify `api/main.py`
- **Frontend (Streamlit)**: Modify `frontend/app.py`
- **PDF Processing**: Modify `scripts/pdf_to_json.py`

## Deployment
### Running in Production
To deploy the application, ensure environment variables are set and run:
```sh
docker compose up -d --build
```


