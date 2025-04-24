import os
import re
import asyncio
from typing import List, Dict, Any, Optional

import openai
import numpy as np
from pypdf import PdfReader
from sqlalchemy import select, func, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import Float
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk, Embedding
from app.config import OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

async def create_embedding(text: str) -> List[float]:
    """Create embedding for a text using OpenAI API"""
    response = await openai.Embedding.acreate(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response["data"][0]["embedding"]

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean text from PDF"""
    # Extract text using PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    
    # Apply aggressive cleaning
    
    # Fix section numbers with spaces
    text = re.sub(r'§\s+(\d+)\s+\.(\d+)', r'§ \1.\2', text)
    
    # Fix other numbers with spaces
    text = re.sub(r'(\d+)\s+\.(\d+)', r'\1.\2', text)
    text = re.sub(r'art\s+(\d+)\s+(\d+)', r'art \1\2', text)
    
    # Fix common split words ("H ealth" -> "Health")
    # This pattern looks for single uppercase letter followed by space then lowercase
    text = re.sub(r'([A-Z])\s+([a-z]{2,})', r'\1\2', text)
    
    # Fix split abbreviations ("C F R" -> "CFR")
    text = re.sub(r'([A-Z])\s+([A-Z])\s+([A-Z])', r'\1\2\3', text)
    text = re.sub(r'([A-Z])\s+([A-Z])', r'\1\2', text)
    
    # Fix joined words by adding spaces between lowercase-uppercase transitions
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Fix "HIPAAT ransaction" -> "HIPAA Transaction" 
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
    
    return text

def extract_metadata(chunk_text: str) -> Dict[str, Any]:
    """Extract HIPAA section metadata from text chunk"""
    metadata = {}
    
    # Clean the text for more reliable extraction
    clean_text = re.sub(r'§\s+(\d+)\s+\.(\d+)', r'§ \1.\2', chunk_text)
    clean_text = re.sub(r'(\d+)\s+\.(\d+)', r'\1.\2', clean_text)
    
    # Extract full section numbers with their part - this is critical
    section_match = re.search(r'§\s*(\d{3})\.(\d{3})', clean_text)
    if section_match:
        section = f"{section_match.group(1)}.{section_match.group(2)}"
        metadata["section"] = section
        
        # Always extract part from section number when available
        part = section_match.group(1)
        if part in ["160", "162", "164"]:
            metadata["part"] = part
    
    # If part not found from section, try direct part reference
    if "part" not in metadata:
        part_match = re.search(r'(?:45 CFR Part|PART)\s+(\d+)', clean_text, re.IGNORECASE)
        if part_match:
            part = part_match.group(1)
            # Fix truncated part numbers
            if part == "16":
                # Try to determine correct part from context
                if "160" in clean_text or "privacy" in clean_text.lower():
                    part = "160"
                elif "164" in clean_text or "security" in clean_text.lower():
                    part = "164"
                else:
                    part = "162"
            metadata["part"] = part
    
    # Extract subpart identifiers
    subpart_match = re.search(r'Subpart\s+([A-Z](?:\-[A-Z])?)\s*(?:\-|—|--)\s*([^\.]+)', clean_text)
    if subpart_match:
        metadata["subpart"] = subpart_match.group(1)
        metadata["subpart_title"] = subpart_match.group(2).strip()
        
    return metadata

def chunk_text(text: str, source: str) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks and extract metadata
    """
    chunks = []
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    current_chunk_size = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If this single paragraph is too long, split it further
        if len(paragraph) > CHUNK_SIZE:
            # Split at sentence boundaries or just force split if needed
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(sentence) > CHUNK_SIZE:
                    # For extremely long sentences, force split
                    words = sentence.split(' ')
                    temp_sentence = ""
                    for word in words:
                        if len(temp_sentence) + len(word) > CHUNK_SIZE - 100:  # Leave room for overlap
                            metadata = extract_metadata(temp_sentence)
                            chunks.append({
                                "content": temp_sentence,
                                "source": source,
                                "metadata": metadata
                            })
                            temp_sentence = word + " "
                        else:
                            temp_sentence += word + " "
                    
                    # Add the last part if not empty
                    if temp_sentence.strip():
                        metadata = extract_metadata(temp_sentence)
                        chunks.append({
                            "content": temp_sentence,
                            "source": source,
                            "metadata": metadata
                        })
                else:
                    # For normal sentences, add to current chunk if it fits
                    if current_chunk_size + len(sentence) <= CHUNK_SIZE:
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += sentence
                        current_chunk_size += len(sentence)
                    else:
                        # Save current chunk and start new one
                        if current_chunk:
                            metadata = extract_metadata(current_chunk)
                            chunks.append({
                                "content": current_chunk,
                                "source": source,
                                "metadata": metadata
                            })
                        current_chunk = sentence
                        current_chunk_size = len(sentence)
        else:
            # Normal paragraph handling (as before)
            if current_chunk_size + len(paragraph) <= CHUNK_SIZE:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
                current_chunk_size += len(paragraph)
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    metadata = extract_metadata(current_chunk)
                    chunks.append({
                        "content": current_chunk,
                        "source": source,
                        "metadata": metadata
                    })
                
                # Start a new chunk with the current paragraph
                current_chunk = paragraph
                current_chunk_size = len(paragraph)
    
    # Add the last chunk if not empty
    if current_chunk:
        metadata = extract_metadata(current_chunk)
        chunks.append({
            "content": current_chunk,
            "source": source,
            "metadata": metadata
        })
    
    return chunks

async def process_pdf_document(pdf_path: str, session: AsyncSession) -> None:
    """Process a PDF document and store it in the database"""
    try:
        filename = os.path.basename(pdf_path)
        print(f"Starting to process {filename}...")
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        
        # Split text into chunks
        print("Splitting text into chunks...")
        chunks = chunk_text(text, source=filename)
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Process chunks in batches
        batch_size = 50  # Adjust based on rate limits
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            end_idx = min(i + batch_size, total_chunks)
            
            # Log progress
            progress = (end_idx / total_chunks) * 100
            print(f"Progress: {progress:.1f}% ({end_idx}/{total_chunks} chunks)")
            
            # Create embeddings for the batch
            embedding_tasks = []
            for chunk_data in batch:
                # First save the chunk to get its ID
                chunk = DocumentChunk(
                    content=chunk_data["content"],
                    source=chunk_data["source"],
                    meta_data=chunk_data["metadata"]  # Make sure this is meta_data
                )
                session.add(chunk)
                await session.flush()  # This assigns an ID to the chunk
                
                # Create embeddings asynchronously
                task = create_embedding(chunk_data["content"])
                embedding_tasks.append((chunk.id, task))
            
            # Wait for all embedding tasks to complete
            for chunk_id, task in embedding_tasks:
                embedding_vector = await task
                
                # Store the embedding
                embedding = Embedding(
                    chunk_id=chunk_id,
                    embedding=embedding_vector
                )
                session.add(embedding)
            
            # Commit the batch
            await session.commit()
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        print(f"✅ Completed processing {filename} - {total_chunks} chunks processed")
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        raise

async def search_similar_chunks(query: str, session: AsyncSession, limit: int = 10) -> List[DocumentChunk]:
    """Search with hybrid retrieval (semantic + keyword)"""
    # Extract part number if present in query, handling all HIPAA parts
    part_match = re.search(r'part\s+(\d+)', query.lower())
    part_number = None
    
    if part_match:
        # Handle both full and truncated part numbers
        matched_part = part_match.group(1)
        if matched_part in ["160", "162", "164"]:
            part_number = matched_part
        elif matched_part == "16":
            # Try to determine which part from other query context
            if "privacy" in query.lower() or "privacy rule" in query.lower():
                part_number = "164"  # Privacy Rule is in Part 164
            elif "security" in query.lower() or "security rule" in query.lower():
                part_number = "164"  # Security Rule is also in Part 164
            elif "transactions" in query.lower() or "code sets" in query.lower():
                part_number = "162"  # Transactions and Code Sets are in Part 162
            elif "general" in query.lower() or "definitions" in query.lower():
                part_number = "160"  # General provisions are in Part 160
            else:
                # Default to 162 if we can't determine
                part_number = "162"
    
    # Get more chunks than requested to filter
    search_limit = limit * 2
    
    # Perform vector search
    query_embedding = await create_embedding(query)
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
    
    # Get base results from vector search
    sql = text(
        f"SELECT dc.id, dc.content, dc.source, dc.meta_data "
        f"FROM document_chunks dc "
        f"JOIN embeddings e ON dc.id = e.chunk_id "
        f"ORDER BY e.embedding <=> '{embedding_str}'::vector "
        f"LIMIT :limit"
    )
    
    result = await session.execute(sql, {"limit": search_limit})
    chunks = []
    for row in result:
        chunk = DocumentChunk(
            id=row[0],
            content=row[1],
            source=row[2],
            meta_data=row[3]
        )
        chunks.append(chunk)
    
    # Filter and prioritize the results
    if part_number:
        # First include chunks that mention this part
        matching_chunks = [c for c in chunks 
                          if c.meta_data and c.meta_data.get("part") == part_number]
        
        # Then add remaining chunks to fill the limit
        other_chunks = [c for c in chunks if c not in matching_chunks]
        return (matching_chunks + other_chunks)[:limit]
    
    return chunks[:limit]