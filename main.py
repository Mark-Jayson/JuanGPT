from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
import logging

from utils import QdrantManager, process_csv_files, CSVHandler
from db import create_conversation, get_conversations, get_messages, add_message, delete_conversation

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

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Data directory
DATA_DIR = "data"

# --- RAG Prompt (with conversation history) ---
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are JuanGPT, a helpful assistant that answers questions about Philippine labor, employment, and demographic statistics.
Use the following context from the datasets to answer the question. If you don't know the answer based on the context, say so.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# --- Initialize LLM, Vector Store, and RAG Chain ---

def format_docs(docs):
    return "\n\n".join(doc.page_content[:2000] for doc in docs)


def initialize_chain():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

        qdrant_manager = QdrantManager()
        vector_store = qdrant_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": int(os.getenv("SIMILARITY_TOP_K", "3"))}
        )

        rag_chain = (
            {
                "context": (lambda x: x["question"]) | retriever | format_docs,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )

        return rag_chain, retriever, qdrant_manager
    except Exception as e:
        logger.error(f"Error initializing chain: {str(e)}")
        raise


# Initialize on startup
rag_chain, retriever, qdrant_manager = initialize_chain()


# --- Models ---

class ChatRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    links: Optional[str] = None
    conversation_id: str


# --- Page Routes ---

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


# --- API Endpoints ---

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        conversation_id = request.conversation_id

        # Create a new conversation if none provided
        if not conversation_id:
            title = request.text[:60] + ("..." if len(request.text) > 60 else "")
            conv = create_conversation(title)
            conversation_id = conv["id"]

        # Load conversation history (last 10 messages for context)
        history = get_messages(conversation_id)
        chat_history = []
        for msg in history[-10:]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # Save user message
        add_message(conversation_id, "user", request.text)

        # Get response from RAG chain with history
        response_text = rag_chain.invoke({
            "question": request.text,
            "chat_history": chat_history,
        })

        # Get source document links
        source_docs = retriever.invoke(request.text)
        links = []
        for doc in source_docs:
            link = doc.metadata.get("link", "")
            if link and link not in links:
                links.append(link)
        links_text = ", ".join(links) if links else None

        # Save assistant message
        add_message(conversation_id, "assistant", response_text, links_text)

        return ChatResponse(response=response_text, links=links_text, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations")
async def list_conversations():
    try:
        return get_conversations()
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    try:
        return get_messages(conversation_id)
    except Exception as e:
        logger.error(f"Error fetching messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    try:
        delete_conversation(conversation_id)
        return {"message": "Conversation deleted"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex")
async def reindex():
    try:
        csv_files = [str(f) for f in Path(DATA_DIR).glob("*.csv")]
        if not csv_files:
            raise HTTPException(status_code=400, detail="No CSV files found in data/")

        logger.info(f"Reindexing {len(csv_files)} CSV files...")
        documents = process_csv_files(csv_files)

        if not documents:
            raise HTTPException(status_code=400, detail="No documents processed")

        qdrant_manager.reindex(documents)

        global rag_chain, retriever
        rag_chain, retriever, _ = initialize_chain()

        return {
            "message": f"Successfully indexed {len(documents)} documents into Qdrant",
            "files_processed": len(csv_files),
            "documents_created": len(documents),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        documents = process_csv_files([file_path])
        if documents:
            vector_store = qdrant_manager.get_vector_store()
            vector_store.add_documents(documents)

        global rag_chain, retriever
        rag_chain, retriever, _ = initialize_chain()

        return {"message": f"File {file.filename} uploaded and indexed successfully"}
    except Exception as e:
        logger.error(f"Error uploading CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Static file handler ---

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    full_path = os.path.join(static_dir, file_path)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
