import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Database settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "hipaa_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "hipaa_db")

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Application settings
CHUNK_SIZE = 275
CHUNK_OVERLAP = 75
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4.1"

# Database connection string
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# System prompt for the RAG model
SYSTEM_PROMPT = """
You are a HIPAA expert assistant. Your task is to answer questions about HIPAA regulations based on the provided context.
Always:
1. Be accurate and precise with your answers
2. When directly quoting regulations, use exact text from the document and cite the specific section
3. Provide section numbers and references from HIPAA when applicable
4. Don't make up information - if the context doesn't provide an answer, say so
5. Don't include irrelevant information

Format your answers in a clear, structured way. If citing multiple sections, organize them logically.
"""