from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, List
import logging
import json
from utils import CSVHandler, process_csv_files, create_or_load_index, update_index

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - Using absolute path
static_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize model and configurations
PERSIST_DIR = "index_storage"
DATA_DIR = "data"

def initialize_model():
    try:
        # Initialize Groq LLM with the model from oldmainGroq
        api_key = os.getenv("GROQ_API_KEY")
        llm = Groq(model="llama3-70b-8192", api_key=api_key)
        Settings.llm = llm
        
        # Set embedding model from oldmainGroq
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        
        # Create or load the index
        index = create_or_load_index(DATA_DIR, PERSIST_DIR)
        
        # Create query engine with specific configurations
        query_engine = index.as_query_engine(
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "3")),
            response_mode=os.getenv("RESPONSE_MODE", "compact")
        )
        
        return query_engine
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

# Initialize the query engine
query_engine = initialize_model()

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    links: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/conversation", response_class=HTMLResponse)
async def read_conversation():
    html_path = os.path.join(static_dir, "conversation.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/about", response_class=HTMLResponse)
async def read_about():
    html_path = os.path.join(static_dir, "about.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# API Endpoints
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Get main response from query engine
        response = query_engine.query(request.text)
        
        # Get related links/sources
        links = query_engine.query(
            "what is the link of the datasets where you can find this information: " + 
            request.text + " (provide only the link)"
        )
        
        # Convert response to string if it's not already
        response_text = str(response)
        links_text = str(links) if links else None
        
        return ChatResponse(response=response_text, links=links_text)
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-index")
async def update_index_endpoint():
    try:
        # Update the index and reinitialize the query engine
        update_index(DATA_DIR, PERSIST_DIR)
        global query_engine
        query_engine = initialize_model()
        return {"message": "Index updated successfully"}
    except Exception as e:
        logger.error(f"Error updating index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the new CSV file
        csv_handler = CSVHandler()
        documents = process_csv_files([file_path])
        
        # Update the index with the new document
        update_index(DATA_DIR, PERSIST_DIR)
        
        # Reinitialize query engine
        global query_engine
        query_engine = initialize_model()
        
        return {"message": f"File {file.filename} uploaded and processed successfully"}
    except Exception as e:
        logger.error(f"Error uploading CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Static file handler
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    full_path = os.path.join(static_dir, file_path)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)