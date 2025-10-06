#!/usr/bin/env python3
"""
FastAPI Server for Docling RAG Agent
=====================================
Provides RESTful API with streaming support for the RAG agent.
"""

import asyncio
import asyncpg
import json
import logging
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from utils import conversation_db

# Load environment variables
load_dotenv(".env", override=True)

logger = logging.getLogger(__name__)

# Global database pool
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await initialize_db()
    logger.info("FastAPI server started")
    yield
    # Shutdown
    await close_db()
    logger.info("FastAPI server shutdown")


# Create FastAPI app
app = FastAPI(
    title="Docling RAG API",
    description="AI-powered document search with streaming responses",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Database Functions =====

async def initialize_db():
    """Initialize database connection pool."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source citations
    """
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        # Generate embedding for query
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)

        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Search using match_chunks function
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        # Format results for response
        if not results:
            return "No relevant information found in the knowledge base for your query."

        # Build response with sources
        response_parts = []
        for i, row in enumerate(results, 1):
            similarity = row['similarity']
            content = row['content']
            doc_title = row['document_title']
            doc_source = row['document_source']

            response_parts.append(
                f"[Source: {doc_title}]\n{content}\n"
            )

        if not response_parts:
            return "Found some results but they may not be directly relevant to your query. Please try rephrasing your question."

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"I encountered an error searching the knowledge base: {str(e)}"


# Create the PydanticAI agent
agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an intelligent knowledge assistant with access to an organization's documentation and information.
Your role is to help users find accurate information from the knowledge base.
You have a professional yet friendly demeanor.

IMPORTANT: Always search the knowledge base before answering questions about specific information.
If information isn't in the knowledge base, clearly state that and offer general guidance.
Be concise but thorough in your responses.
Ask clarifying questions if the user's query is ambiguous.
When you find relevant information, synthesize it clearly and cite the source documents.""",
    tools=[search_knowledge_base]
)


# ===== Pydantic Models =====

class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    history: List[Message] = Field(default_factory=list, description="Conversation history")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Agent response")
    history: List[Message] = Field(..., description="Updated conversation history")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    documents: Optional[int] = None
    chunks: Optional[int] = None


class ConversationCreate(BaseModel):
    """Create conversation request."""
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Conversation title")


class ConversationResponse(BaseModel):
    """Conversation response."""
    id: str
    user_id: str
    title: str
    last_message: Optional[str] = None
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    """Message response."""
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


class ChatStreamRequest(BaseModel):
    """Chat stream request with optional conversation ID."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID (creates new if not provided)")
    user_id: str = Field(..., description="User ID")


# ===== API Endpoints =====

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Docling RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with database status."""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

            if result == 1:
                return HealthResponse(
                    status="healthy",
                    database="connected",
                    documents=doc_count,
                    chunks=chunk_count
                )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint.

    Send a message and receive the complete response.
    """
    try:
        # Convert history to PydanticAI format (simplified)
        # Just pass the message, agent handles history internally
        message_history = None

        # Run agent
        result = await agent.run(request.message, message_history=message_history)

        # Build updated history
        updated_history = request.history + [
            Message(role="user", content=request.message),
            Message(role="assistant", content=result.data)
        ]

        return ChatResponse(
            response=result.data,
            history=updated_history
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatStreamRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Send a message and receive streaming response tokens.
    Automatically saves messages to the conversation.
    """
    async def event_generator():
        conversation_id = request.conversation_id
        assistant_response = ""

        try:
            # Create new conversation if not provided
            if not conversation_id:
                # Generate title from first message (truncate if too long)
                title = request.message[:50] + "..." if len(request.message) > 50 else request.message
                conversation = await conversation_db.create_conversation(
                    db_pool,
                    request.user_id,
                    title
                )

                if not conversation:
                    raise Exception("Failed to create conversation")

                conversation_id = conversation["id"]

                # Send conversation ID to frontend
                yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"

            # Save user message
            await conversation_db.save_message(
                db_pool,
                conversation_id,
                "user",
                request.message
            )

            # Convert history to PydanticAI format (simplified)
            # Just pass the message, agent handles history internally
            message_history = None

            # Stream response
            async with agent.run_stream(
                request.message,
                message_history=message_history
            ) as result:
                # Stream each token
                async for text in result.stream_text(delta=True):
                    assistant_response += text
                    # Send token as SSE event
                    yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

                # Save assistant response
                await conversation_db.save_message(
                    db_pool,
                    conversation_id,
                    "assistant",
                    assistant_response
                )

                # Send completion event
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base."""
    try:
        async with db_pool.acquire() as conn:
            documents = await conn.fetch(
                """
                SELECT id, title, source, created_at, updated_at
                FROM documents
                ORDER BY created_at DESC
                """
            )

            return {
                "count": len(documents),
                "documents": [
                    {
                        "id": str(doc["id"]),
                        "title": doc["title"],
                        "source": doc["source"],
                        "created_at": doc["created_at"].isoformat() if doc["created_at"] else None,
                        "updated_at": doc["updated_at"].isoformat() if doc["updated_at"] else None,
                    }
                    for doc in documents
                ]
            }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")


# ===== Conversation Endpoints =====

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate):
    """Create a new conversation."""
    conversation = await conversation_db.create_conversation(
        db_pool,
        request.user_id,
        request.title
    )

    if not conversation:
        raise HTTPException(status_code=500, detail="Failed to create conversation")

    return ConversationResponse(**conversation)


@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(user_id: str):
    """List all conversations for a user."""
    conversations = await conversation_db.get_conversations(db_pool, user_id)
    return [ConversationResponse(**conv) for conv in conversations]


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(conversation_id: str):
    """Get all messages for a conversation."""
    messages = await conversation_db.get_conversation_messages(db_pool, conversation_id)
    return [MessageResponse(**msg) for msg in messages]


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str):
    """Delete a conversation."""
    success = await conversation_db.delete_conversation(db_pool, conversation_id, user_id)

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")

    return {"status": "deleted", "id": conversation_id}


@app.patch("/conversations/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, user_id: str, title: str):
    """Update conversation title."""
    success = await conversation_db.update_conversation_title(db_pool, conversation_id, user_id, title)

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")

    return {"status": "updated", "id": conversation_id, "title": title}


# ===== Main Entry Point =====

def main():
    """Run the FastAPI server with Uvicorn."""
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        raise ValueError("DATABASE_URL not set")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        raise ValueError("OPENAI_API_KEY not set")

    # Run server
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
