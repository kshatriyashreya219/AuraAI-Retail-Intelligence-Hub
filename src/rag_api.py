"""FastAPI application for the AURA OS retrieval-augmented chatbot.

This API accepts an uploaded Orders.csv file, indexes it in ChromaDB, and answers
user questions grounded in the stored order data. The response includes the answer,
its evidence sources, and chart-ready summary data.
"""

import asyncio
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.multimodal_api import router as multimodal_router

app = FastAPI(title="AURA OS RAG API", version="1.0.0")
app.include_router(multimodal_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
MAX_CSV_UPLOAD_BYTES = 25 * 1024 * 1024
rag_service = None


def get_rag_service() -> Any:
    """Load the vector service only when a RAG endpoint is used."""
    global rag_service
    if rag_service is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY must be configured on the API server.")
        from src.rag_chatbot import AURAOrdersRAG

        rag_service = AURAOrdersRAG(persist_directory="./.aura_chroma")
    return rag_service


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.post("/api/upload")
async def upload_orders(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload Orders.csv and ingest it into the vector store."""
    filename = (file.filename or "").replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    content = await file.read(MAX_CSV_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > MAX_CSV_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV uploads must be 25 MB or smaller.")

    temporary_path: str | None = None
    try:
        with NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name
        service = get_rag_service()
        document_count = await asyncio.to_thread(
            service.ingest_csv,
            temporary_path,
            source_name=filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EnvironmentError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - API safety layer
        raise HTTPException(status_code=502, detail=f"Failed to ingest data: {str(exc)}") from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
        await file.close()

    return {
        "status": "success",
        "file": filename,
        "documents_ingested": document_count,
        "message": "Orders data is now indexed for retrieval.",
    }


@app.post("/api/ask")
async def ask_question(payload: AskRequest) -> Dict[str, Any]:
    """Answer a natural-language business question from the indexed dataset."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        service = get_rag_service()
        result = await asyncio.to_thread(service.ask, question, k=5)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Install the declared RAG dependencies to enable chat.") from exc
    except Exception as exc:  # pragma: no cover - API safety layer
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {str(exc)}") from exc

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chart_data": result["chart_data"],
    }


@app.post("/api/run-agents")
async def run_agents(
    file: UploadFile = File(...),
    shaadi_code: str | None = Form(default=None),
) -> Dict[str, Any]:
    """Run the financial, inventory, and wedding agents against an uploaded CSV."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    temporary_path: str | None = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")

        with NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name

        from src.aura_agents import run_aura_agents

        agent_outputs = await asyncio.to_thread(run_aura_agents, temporary_path, shaadi_code)
        return {
            "profit_agent": agent_outputs["profit_agent"],
            "stock_agent": agent_outputs["stock_agent"],
            "wedding_agent": agent_outputs["wedding_agent"],
        }
    except HTTPException:
        raise
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent workflow failed: {str(exc)}") from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "service": "AURA OS RAG API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.rag_api:app", host="0.0.0.0", port=8000, reload=True)
