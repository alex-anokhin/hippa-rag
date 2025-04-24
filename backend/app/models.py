from sqlalchemy import Column, Integer, Text, String, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    meta_data = Column(JSONB, default={})  # Changed from 'metadata' to 'meta_data'
    
    # Relationship with the embedding
    embedding_rel = relationship("Embedding", back_populates="chunk", cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id", ondelete="CASCADE"))
    embedding = Column(Vector(1536))  # OpenAI embeddings dimension
    
    # Relationship with the chunk
    chunk = relationship("DocumentChunk", back_populates="embedding_rel")