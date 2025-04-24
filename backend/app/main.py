import os
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from app.database import get_db_session, async_session
from app.embeddings import process_pdf_document, search_similar_chunks
from app.rag import generate_response, Message

app = FastAPI(title="HIPAA RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response
class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    status: str
    message: str

@app.get("/")
async def root():
    return {"message": "HIPAA RAG API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# @app.post("/api/upload", response_model=StatusResponse)
# async def upload_document(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     db: AsyncSession = Depends(get_db_session)
# ):
#     """Upload and process a HIPAA document"""
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     # Save file temporarily
#     temp_file_path = f"/tmp/{file.filename}"
#     with open(temp_file_path, "wb") as f:
#         f.write(await file.read())
    
#     # Process document in background
#     background_tasks.add_task(process_pdf_document, temp_file_path, db)
    
#     return {
#         "status": "success",
#         "message": f"Document '{file.filename}' uploaded and being processed"
#     }

@app.post("/api/query", response_model=QueryResponse)
async def query_hipaa(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Query the HIPAA documentation using RAG"""
    # Pass history to generate_response
    response = await generate_response(
        query=request.query,
        chunks_fn=search_similar_chunks,
        db_session=db,
        history=request.history or []  # Default to empty list if None
    )
    return response

# Initialize the database and process initial documents
@app.on_event("startup")
async def startup_event():
    # Wait for DB to be ready
    await asyncio.sleep(5)
    print("Database connection ready, checking for documents...")
    
    # Start process in background so app can finish startup
    asyncio.create_task(_process_initial_documents())
    print("✅ API startup completed - document processing will continue in background")

async def _process_initial_documents():
    """Process initial documents in background to allow app to start up"""
    try:
        # Check if initial document exists and process it
        hipaa_pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "hipaa.pdf")
        if os.path.exists(hipaa_pdf_path):
            print(f"Found HIPAA PDF document at {hipaa_pdf_path}")
            # Create a new session directly instead of using the dependency
            async with async_session() as session:
                # Check if we already have documents in the database
                from sqlalchemy import func, select
                from app.models import DocumentChunk
                
                query = select(func.count()).select_from(DocumentChunk)
                result = await session.execute(query)
                count = result.scalar()
                
                if count == 0:
                    print("No existing documents found in database. Starting PDF processing...")
                    await process_pdf_document(hipaa_pdf_path, session)
                    print("✅ Initial document processing complete")
                else:
                    print(f"✅ Database already contains {count} document chunks, skipping processing")
        else:
            print("⚠️ No HIPAA PDF document found in data directory")
    except Exception as e:
        print(f"❌ Error during background processing: {str(e)}")