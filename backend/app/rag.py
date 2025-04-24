import openai
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re

from app.models import DocumentChunk
from app.config import OPENAI_API_KEY, COMPLETION_MODEL, SYSTEM_PROMPT

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def format_context(chunks: List[DocumentChunk]) -> str:
    """Format document chunks into a context string for the LLM"""
    context = "HIPAA DOCUMENT CONTEXT:\n\n"
    
    for i, chunk in enumerate(chunks):
        # Extract metadata for citation
        metadata = chunk.meta_data or {}  # Make sure this is meta_data not metadata
        part = metadata.get("part", "")
        section = metadata.get("section", "")
        
        # Create citation string
        citation = ""
        if part:
            citation += f"Part {part}"
        if section:
            if citation:
                citation += f", § {section}"
            else:
                citation += f"§ {section}"
                
        # Add subparagraph info if available
        subpara_num = metadata.get("subparagraph_numeric", "")
        subpara_roman = metadata.get("subparagraph_roman", "")
        subpara_alpha = metadata.get("subparagraph_alpha", "")
        
        if subpara_num or subpara_roman or subpara_alpha:
            citation += " "
            if subpara_num:
                citation += f"({subpara_num})"
            if subpara_roman:
                citation += f"({subpara_roman})"
            if subpara_alpha:
                citation += f"({subpara_alpha})"
        
        # Add chunk to context
        context += f"[CHUNK {i+1}] "
        if citation:
            context += f"[{citation}] "
        context += f"{chunk.content}\n\n"
    
    return context

def format_source_reference(source: Dict[str, Any]) -> str:
    """Format a source reference showing only structured citation"""
    metadata = source.get("metadata", {}) or {}
    content = source.get("content", "").strip()
    
    # Build citation parts list
    citation_parts = []
    
    # Extract section number first (to infer part if needed)
    section = None
    if metadata.get("section"):
        section = metadata.get("section").strip()
        # Clean up section number
        section = re.sub(r'\s+', '', section)
    
    # Get part number or infer it from section
    part_num = None
    if metadata.get("part"):
        part_num = metadata.get("part").strip()
    elif section:
        # Infer part from section number
        if section.startswith("160."):
            part_num = "160"
        elif section.startswith("162."):
            part_num = "162"
        elif section.startswith("164."):
            part_num = "164"
    
    # Add part to citation if available
    if part_num:
        # Fix truncated part numbers
        if part_num == "16":
            if section and section.startswith("160."):
                part_num = "160"
            elif section and section.startswith("164."):
                part_num = "164"
            else:
                part_num = "162"  # default
                
        citation_parts.append(f"Part {part_num}")
    
    # Add section to citation if available
    if section:
        # Fix truncated sections based on part
        if section.startswith("16."):
            if part_num == "160":
                section = "160" + section[2:]
            elif part_num == "164":
                section = "164" + section[2:]
            else:
                section = "162" + section[2:]
                
        citation_parts.append(f"§ {section}")
    
    # Format the final source string
    if citation_parts:
        return " | ".join(citation_parts)
    
    # Last resort - try to find references in content
    if content:
        # Look for part references in content
        part_match = re.search(r'(?:45 CFR Part|PART)\s+(\d+)', content, re.IGNORECASE)
        if part_match and part_match.group(1) in ["160", "162", "164"]:
            return f"Part {part_match.group(1)}"
            
        # Look for section references in content
        section_match = re.search(r'§\s*(\d{3}\.\d{3})', content)
        if section_match:
            section = section_match.group(1)
            part = section[:3]  # Extract part from section (e.g., "164" from "164.506")
            return f"Part {part} | § {section}"
    
    return "HIPAA Reference"  # Simple, clean fallback

class Message(BaseModel):
    role: str
    content: str

async def generate_response(
    query: str, 
    chunks_fn, 
    db_session, 
    history: Optional[List[Message]] = None
) -> Dict[str, Any]:
    """
    Multi-step RAG with chat history support
    """
    if history is None:
        history = []
        
    # Format history for the interpreter
    history_text = ""
    if history:
        history_text = "Chat history:\n"
        for msg in history[-4:]:  # Only use last 4 messages to keep context reasonable
            # Access attributes with dot notation, not dictionary syntax
            role = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role}: {msg.content}\n\n"
    
    # Step 1: Interpret user query for retrieval
    interp_resp = await openai.ChatCompletion.acreate(
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant that reformulates user questions to maximize retrieval from HIPAA documents. When users ask about specific parts or sections (like 'Part 160'), EXPLICITLY include those exact terms and numbers in your reformulation. If the user refers to previous messages, use the chat history to understand the full context."},
            {"role": "user", "content": f"{history_text}\nOriginal question: {query}\n\nReformulate this for searching HIPAA documentation:"}
        ],
        temperature=0.3,
        max_tokens=256
    )
    rag_query = interp_resp.choices[0].message.content.strip()
    
    # Step 2: Retrieve relevant chunks using the interpreted query
    chunks = await chunks_fn(rag_query, db_session, limit=10)
    
    # Step 3: Generate the final answer using the original query, history, and retrieved context
    context = format_context(chunks)
    
    # Format history for the answer generation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history messages (limit to prevent context overflow)
    if history:
        for msg in history[-6:]:  # Use more history for the final answer
            # Access attributes with dot notation, not dictionary syntax
            messages.append({"role": msg.role, "content": msg.content})
    
    # Add current query with RAG context
    user_prompt = f"""Question: {query}

Please answer the question using the information provided in the context below.
If the exact answer is not stated explicitly, synthesize an answer based on the available information.
Only state "I cannot find information on this" if the context provides absolutely no relevant information.
Always cite specific HIPAA sections and provide exact quotes when appropriate.

{context}"""
    
    messages.append({"role": "user", "content": user_prompt})
    
    # Generate response with context and history
    response = await openai.ChatCompletion.acreate(
        model=COMPLETION_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048
    )

    # After getting the response, format the sources properly
    formatted_sources = []
    for chunk in chunks:
        formatted_sources.append({
            "content": chunk.content,
            "metadata": chunk.meta_data,
            "source": chunk.source,
            "formatted_citation": format_source_reference({
                "content": chunk.content,
                "metadata": chunk.meta_data,
                "source": chunk.source
            })
        })
    
    return {
        "answer": response.choices[0].message.content,
        "sources": formatted_sources
    }