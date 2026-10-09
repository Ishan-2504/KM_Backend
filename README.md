# KM Search Portal

This repository contains a two-part KM Search application:

- A `FastAPI` backend that ingests KM data, stores embeddings in ChromaDB, and exposes search/filter APIs.
- A `React + TypeScript` frontend that provides the user interface and consumes the backend over HTTP and SSE.

This README is intended to be the project's global context for developers and AI agents.

## What The App Does

KM Search is an AI-assisted knowledge search tool for browsing and retrieving internal knowledge assets. It supports:

- Semantic search over ingested KM content.
- Filter-driven browsing across document metadata.
- Streaming search results from the backend to the frontend using server-sent events.
- Re-ingestion of source data into the vector store.

## High-Level Architecture

- `frontend/`
  React app built with Create React App and TypeScript.
- `backend/`
  FastAPI service responsible for ingestion, filtering, search orchestration, and Chroma persistence.
- `start.bat`
  Windows bootstrap script that installs dependencies and launches both apps.

Request flow:

1. The frontend sends filter and search requests to the backend.
2. The backend queries the Chroma collection and applies search logic.
3. Search results are streamed back to the frontend from `/api/search`.

## Project Structure

- `backend/app/main.py`
  FastAPI entry point and API route registration.
- `backend/app/config.py`
  Environment-driven configuration for Azure OpenAI, Chroma, and data paths.
- `backend/app/models.py`
  Shared backend schemas, metadata mappings, and filterable fields.
- `backend/app/services/`
  Search, ingestion, query decomposition, query expansion, filter extraction, embeddings, and logging logic.
- `backend/app/db/chroma_client.py`
  Chroma client and collection access.
- `backend/requirements.txt`
  Python dependencies.
- `frontend/src/App.tsx`
  Main UI shell.
- `frontend/src/components/`
  Search bar, filters, result cards, and results list.
- `frontend/src/hooks/useSSESearch.ts`
  Streaming search integration.
- `frontend/src/services/api.ts`
  Frontend API helpers.

## Tech Stack

Backend:

- Python
- FastAPI
- Uvicorn
- ChromaDB
- OpenAI SDK
- `sse-starlette`
- `openpyxl`

Frontend:

- React
- TypeScript
- `react-scripts`

## Requirements

- Python 3.10+
- Node.js 18+

## Configuration

### Backend Environment

The backend reads environment variables from `backend/.env`.

Expected variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDINGS_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `CHROMA_PERSIST_DIR`
- `DATA_FILE_PATH`

Current backend defaults in code:

- `CHROMA_PERSIST_DIR` defaults to `./app/db/chroma_data`
- `DATA_FILE_PATH` defaults to `../data/km_final_output.xlsx`

### Frontend Environment

The frontend reads `REACT_APP_API_URL` from `frontend/.env`.

Default local value:

- `REACT_APP_API_URL=http://localhost:8000`

`start.bat` creates `frontend/.env` automatically if it does not exist.

## Running The Project

### Windows Quick Start

From the repository root:

```bat
start.bat
```

What it does:

1. Checks for Python and Node.js.
2. Creates `backend/venv` if needed.
3. Installs Python dependencies from `backend/requirements.txt`.
4. Installs frontend dependencies with `npm install`.
5. Verifies `backend/.env` exists.
6. Starts backend on port `8000`.
7. Starts frontend on port `3000`.

### Manual Start

Backend:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```powershell
cd frontend
npm install
npm start
```

App URLs:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Core API Endpoints

- `GET /api/health`
  Returns service status and current document count.
- `GET /api/filters`
  Returns available values for filter dropdowns.
- `POST /api/search`
  Streaming search endpoint using SSE.
- `POST /api/search/sync`
  Non-streaming search endpoint.
- `POST /api/ingest`
  Triggers background data ingestion.

## Data Model And Filters

The backend supports metadata-based filtering across fields such as:

- `document_type`
- `designation`
- `knowledge_type`
- `knowledge_sub_type`
- `industry`
- `sub_industry`
- `solution`
- `client_name`
- `client_type`
- `year`

These fields are defined in `backend/app/models.py` and are used for both ingestion metadata and frontend filter options.

## Local Data And Persistence

- Source data is expected to come from an Excel file referenced by `DATA_FILE_PATH`.
- Vector data is persisted locally in Chroma under `CHROMA_PERSIST_DIR`.
- The root `.gitignore` excludes common local-only artifacts such as env files, virtual environments, `node_modules`, and Chroma data.

## Development Notes

- The frontend is currently configured for local backend development by default.
- The backend enables permissive CORS for development.
- Search results are streamed incrementally to improve perceived responsiveness in the UI.
- `frontend/README.md` is still the default Create React App README and is not the source of truth for overall project context.

## Suggested Team Conventions

- Keep secrets only in local `.env` files.
- Avoid committing generated Chroma data, local spreadsheets, or installed dependencies.
- Treat this `README.md` as the first-stop reference for project structure, setup, and intent.
