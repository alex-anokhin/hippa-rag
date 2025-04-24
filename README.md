# HIPAA RAG Solution

A Retrieval-Augmented Generation (RAG) system for answering questions about HIPAA regulations using OpenAI, PostgreSQL with pg_vector, FastAPI, and Gradio.

## Features

- **Document Processing**: Upload and process HIPAA PDF documents
- **Vector Embeddings**: Store and search document chunks using vector similarity
- **Accurate Answers**: Generate precise answers to HIPAA-related questions
- **Source Attribution**: Include references to specific HIPAA sections and paragraphs
- **Simple UI**: User-friendly chat interface built with Gradio

## Architecture

- **PostgreSQL with pg_vector**: Vector database for semantic search
- **FastAPI Backend**: Async API service for document processing and querying
- **Gradio Frontend**: Simple chat UI for asking questions
- **Nginx**: Reverse proxy to route traffic
- **Docker Compose**: Container orchestration

## Getting Started

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/alex-anokhin/hipaa-rag.git
   cd hipaa-rag
   ```

2. Create a `.env.local` file with your environment variables:
   ```
   # Database Configuration
   DB_USER=hipaa_user
   DB_PASSWORD=your_secure_password
   DB_NAME=hipaa_db

   # OpenAI API
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Place your HIPAA PDF document in the `backend/data/` directory as `hipaa.pdf`

4. Build and start the containers:
   ```
   docker-compose up
   ```

6. Wait ~ 15 min until chanks embedding complete, and you'll see the log:
   ```
   hipaa_backend   | ✅ Initial document processing complete
   ```

5. Access the application at http://localhost:12080

## Usage

### Chatting with the HIPAA Assistant

1. Open the application in your browser
2. Type your HIPAA-related question in the chat interface
3. View the answer with cited sources

## Components

- **backend**: FastAPI service for processing documents and generating answers
- **frontend**: Gradio UI for interacting with the system
- **db**: PostgreSQL database with pg_vector extension for vector similarity search
- **nginx**: Reverse proxy for routing requests

### Testing

To test the application, you can use the sample evaluation questions provided in the assignment:

1. What is the overall purpose of HIPAA Part 160?
2. Which part covers data privacy measures?
3. What does "minimum necessary" mean in HIPAA terminology?
4. Which entities are specifically regulated under HIPAA?
5. What are the potential civil penalties for noncompliance?
6. Does HIPAA mention encryption best practices?
7. Can I disclose personal health information to family members?
8. If a covered entity outsources data processing, which sections apply?
9. Cite the specific regulation texts regarding permitted disclosures to law enforcement.

## License

[MIT License](LICENSE)