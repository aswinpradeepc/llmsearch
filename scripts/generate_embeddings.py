import os
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

# Constants
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = "financial-search-index"
DATA_DIR = "/home/aswin/geojit/llmsearch/media/processed_data"

def load_document(filepath: str) -> Dict:
    """Load a single JSON document."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return {
                'id': os.path.splitext(os.path.basename(filepath))[0],
                'content': data.get('text', ''),
                'metadata': {
                    'filename': data.get('filename', '')
                }
            }
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None

def generate_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI API."""
    try:
        # Ensure text is not empty and is a string
        if not isinstance(text, str) or not text.strip():
            return None
            
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text.replace("\n", " ")[:8192]  # Truncate to model limit
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def initialize_pinecone():
    """Initialize Pinecone index if it doesn't exist."""
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,  # Ada-002 embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise

def main():
    try:
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        index = initialize_pinecone()
        
        # Process documents
        documents = []
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(DATA_DIR, filename)
                doc = load_document(filepath)
                if doc and doc['content']:  # Only add if content exists
                    documents.append(doc)
        
        logger.info(f"Found {len(documents)} documents to process")
        
        # Generate and store embeddings in batches
        batch = []
        for i, doc in enumerate(documents, 1):
            embedding = generate_embedding(doc['content'])
            if embedding:
                vector = {
                    'id': doc['id'],
                    'values': embedding,
                    'metadata': doc['metadata']
                }
                batch.append(vector)
                
                if len(batch) >= BATCH_SIZE:
                    index.upsert(vectors=batch)
                    logger.info(f"Processed {i}/{len(documents)} documents")
                    batch = []
        
        # Upload remaining vectors
        if batch:
            index.upsert(vectors=batch)
            logger.info("Upload complete")
        
        logger.info("Embedding generation and storage complete!")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()