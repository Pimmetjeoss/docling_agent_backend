# Integration Guide: Connecting Docling RAG Agent to AIChat

This guide explains how to run the complete stack connecting your Python RAG agent to the Next.js chat interface.

## Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Next.js UI    │ ──────▶ │  FastAPI Backend │ ──────▶ │  PostgreSQL +   │
│  (Port 3000)    │ ◀────── │   (Port 8000)    │ ◀────── │    PGVector     │
└─────────────────┘         └──────────────────┘         └─────────────────┘
      (Frontend)                  (Python)                    (Database)
```

**Flow:**
1. User types message in Next.js chat UI
2. Next.js sends request to `/api/chat` (Next.js API route)
3. Next.js API route proxies to FastAPI backend at `localhost:8000/chat/stream`
4. FastAPI backend runs the RAG agent (PydanticAI)
5. RAG agent searches PostgreSQL/PGVector for relevant chunks
6. FastAPI streams response back via Server-Sent Events (SSE)
7. Next.js receives streaming tokens and displays them in real-time

## Prerequisites

- Python 3.9+ with `uv` installed
- Node.js 18+ with `npm` or `pnpm`
- PostgreSQL database with PGVector extension
- OpenAI API key

## Setup Instructions

### 1. Configure Python Backend

**Install Python dependencies:**
```bash
# From the root directory (docling-rag-agent/)
uv sync
```

**Configure environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials:
# - DATABASE_URL: Your PostgreSQL connection string
# - OPENAI_API_KEY: Your OpenAI API key
# - API_PORT: 8000 (default)
```

**Ensure your database is set up:**
```bash
# Run the schema file if you haven't already
psql $DATABASE_URL < sql/schema.sql
```

**Ingest documents into the knowledge base:**
```bash
# Add documents to the documents/ folder, then run:
uv run python -m ingestion.ingest --documents documents/
```

### 2. Configure Next.js Frontend

**Install Node.js dependencies:**
```bash
# Navigate to the aichat directory
cd aichat

# Install dependencies
npm install
# or
pnpm install
```

**Configure environment variables:**
```bash
# Copy the example file
cp .env.local.example .env.local

# Edit .env.local:
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Running the Application

You need to run **both** the Python backend and the Next.js frontend.

#### Terminal 1: Start Python FastAPI Backend

```bash
# From the root directory (docling-rag-agent/)
uv run python api_server.py
```

This starts the FastAPI server at `http://localhost:8000`

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 2: Start Next.js Frontend

```bash
# From the aichat directory
cd aichat
npm run dev
# or
pnpm dev
```

This starts the Next.js development server at `http://localhost:3000`

You should see:
```
▲ Next.js 15.5.4
- Local:        http://localhost:3000
```

### 4. Access the Application

1. Open your browser to **http://localhost:3000**
2. Click "Start Chatting" or navigate to **http://localhost:3000/chat**
3. Start asking questions about your knowledge base!

## Testing the Integration

### Health Check

Before using the chat, verify the backend is working:

```bash
# Test the backend directly
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "database": "connected",
  "documents": 20,
  "chunks": 156
}
```

### Test Non-Streaming Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What topics are in the knowledge base?",
    "history": []
  }'
```

### Test Streaming Chat

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What topics are in the knowledge base?",
    "history": []
  }'
```

You should see streaming Server-Sent Events (SSE) coming back:
```
data: {"type": "token", "content": "Based"}
data: {"type": "token", "content": " on"}
data: {"type": "token", "content": " the"}
...
data: {"type": "done"}
```

## Features

### Frontend (Next.js)
- ✅ Modern chat interface with Tailwind CSS
- ✅ Real-time streaming responses
- ✅ Message history persistence (client-side)
- ✅ Health status indicator
- ✅ Clear chat functionality
- ✅ Responsive design (mobile-friendly)
- ✅ Dark mode support

### Backend (FastAPI)
- ✅ RESTful API with streaming support
- ✅ Server-Sent Events (SSE) for real-time responses
- ✅ CORS configured for Next.js
- ✅ Health check endpoint with database stats
- ✅ Document listing endpoint
- ✅ Proper error handling

### RAG Agent (PydanticAI)
- ✅ Semantic search with vector embeddings
- ✅ PostgreSQL/PGVector for scalable storage
- ✅ OpenAI GPT-4o-mini for responses
- ✅ Source citations
- ✅ Conversation history

## API Endpoints

### Python FastAPI Backend (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with service info |
| `/health` | GET | Health check with database stats |
| `/chat` | POST | Non-streaming chat endpoint |
| `/chat/stream` | POST | Streaming chat with SSE |
| `/documents` | GET | List all documents in knowledge base |

### Next.js API Routes (Port 3000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Proxies to Python `/health` |
| `/api/chat` | POST | Proxies to Python `/chat/stream` |

## Troubleshooting

### Backend won't start
- Ensure PostgreSQL is running and accessible
- Check `DATABASE_URL` in `.env`
- Check `OPENAI_API_KEY` is set
- Run `uv sync` to install dependencies

### Frontend can't connect to backend
- Ensure FastAPI is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in `aichat/.env.local`
- Check CORS settings in `api_server.py`
- Check browser console for errors

### No documents found
- Run the ingestion script: `uv run python -m ingestion.ingest --documents documents/`
- Check the `/health` endpoint to see document count
- Verify documents are in the `documents/` folder

### Streaming not working
- Check browser console for SSE connection errors
- Verify `Content-Type: text/event-stream` in response headers
- Try the curl test command to verify backend streaming works

## Production Deployment

### Backend
- Use a production WSGI server (already using Uvicorn)
- Set up environment variables securely
- Configure database connection pooling
- Add rate limiting
- Set up logging and monitoring

### Frontend
- Build for production: `npm run build`
- Deploy to Vercel, Netlify, or your preferred hosting
- Update `NEXT_PUBLIC_API_URL` to point to production backend
- Enable API caching if needed

### Database
- Use a managed PostgreSQL service (Supabase, Neon, AWS RDS)
- Set up backups
- Monitor query performance
- Scale PGVector indices as needed

## Development Tips

### Watch Mode
Both servers support hot reloading:
- FastAPI: Auto-reloads when Python files change (via `--reload`)
- Next.js: Auto-reloads when TypeScript/React files change

### Logging
- Python backend logs go to console (INFO level)
- Next.js logs go to console
- Check browser DevTools Network tab for API requests

### Adding New Features
- New API endpoints: Add to `api_server.py` and create corresponding Next.js routes
- New UI components: Add to `aichat/app/components/`
- New RAG tools: Add to agent tools list in `api_server.py`

## Next Steps

1. **Customize the UI**: Modify `aichat/app/chat/page.tsx` to match your brand
2. **Add authentication**: Protect the chat endpoint with user authentication
3. **Improve RAG**: Tune the system prompt, add more tools, adjust chunk sizes
4. **Add features**: Document upload, chat history persistence, export conversations
5. **Deploy**: Put it in production!

## Resources

- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [PGVector Documentation](https://github.com/pgvector/pgvector)
