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

def truncate_text(text: str, max_chars: int = 15000) -> str:
    """Truncate text to stay within Pinecone metadata limits."""
    if not text:
        return ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

def load_document(filepath: str) -> Dict:
    """Load a single JSON document."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            content = data.get('text', '')  # Get text from JSON
            # Truncate content to stay within metadata limits
            truncated_content = truncate_text(content)
            return {
                'id': os.path.splitext(os.path.basename(filepath))[0],
                'content': truncated_content,
                'metadata': {
                    'filename': data.get('filename', ''),
                    'content': truncated_content  # Include content in metadata
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
            
        # Truncate text to OpenAI's limit
        truncated_text = text[:8192].replace("\n", " ")
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=truncated_text
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
                    'metadata': {
                        'filename': doc['metadata']['filename'],
                        'content': doc['content']  # Ensure content is in metadata
                    }
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