"""
FastAPI REST API for Agentic AI eBook RAG System.

Endpoints:
- POST /query: Submit query, returns final answer, retrieved context chunks, and confidence score.
- GET /health: Health check endpoint verifying service status.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from src.config import get_settings
from src.graph import build_rag_graph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Agentic AI eBook RAG API",
    description="Retrieval-Augmented Generation API powered by LangGraph, Pinecone, and Multi-Provider AI (Gemini/OpenAI)",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached graph instance
_cached_graph = None
_cached_provider = None


def get_cached_graph(provider: Optional[str] = None):
    """Lazily instantiate and cache compiled LangGraph instance."""
    global _cached_graph, _cached_provider
    target_provider = (provider or get_settings().llm_provider).strip().lower()

    if _cached_graph is None or _cached_provider != target_provider:
        _cached_graph = build_rag_graph(provider=target_provider)
        _cached_provider = target_provider
    return _cached_graph, target_provider


class ErrorResponse(BaseModel):
    """Standardized error response payload."""

    detail: str = Field(..., description="Description of the error condition")


class QueryRequest(BaseModel):
    """Request payload for RAG query and chat endpoints."""

    query: str = Field(
        ...,
        min_length=1,
        description="Question about the Agentic AI eBook (must contain non-whitespace text)",
        examples=["What is the core definition of Agentic AI as outlined in the eBook?"],
    )
    provider: Optional[Literal["gemini", "openai"]] = Field(
        default=None,
        description="Optional provider override ('gemini' or 'openai')",
        examples=["gemini"],
    )

    @field_validator("query")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace only.")
        return cleaned

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip().lower()
        if cleaned not in ("gemini", "openai"):
            raise ValueError("Provider must be either 'gemini' or 'openai'.")
        return cleaned


class QueryResponse(BaseModel):
    """Response payload strictly containing the 4 required output keys."""

    query: str = Field(..., description="The user query")
    final_answer: str = Field(..., description="Answer generated strictly from context")
    retrieved_context_chunks: List[str] = Field(
        default_factory=list, description="Top-k context chunks retrieved from Pinecone"
    )
    confidence_score: float = Field(
        ..., description="Confidence score reflecting context relevance (0.0 to 1.0)"
    )


RESPONSES_SPEC: Dict[int | str, Dict[str, Any]] = {
    200: {"description": "Successful retrieval-augmented response strictly grounded in context"},
    400: {"model": ErrorResponse, "description": "Invalid request parameter or bad input format"},
    422: {"description": "Schema validation error (e.g. empty query or unsupported provider value)"},
    500: {"model": ErrorResponse, "description": "Internal server or inference processing error"},
}


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    summary="Health Check",
    description="Verifies the API service status and configured settings.",
    responses={200: {"description": "Service is healthy and operational"}},
)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "configured_provider": settings.llm_provider,
        "index_name": settings.pinecone_index_name,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit RAG Query",
    description="Queries the Agentic AI eBook RAG pipeline and returns a grounded response with confidence score.",
    responses=RESPONSES_SPEC,
)
def query_endpoint(request: QueryRequest):
    """
    Query the Agentic AI eBook RAG pipeline.
    Invokes the stateful LangGraph workflow and returns strictly grounded response.
    Available on POST /query.
    Executed synchronously to offload blocking I/O to Starlette's threadpool.
    """
    try:
        rag_graph, active_provider = get_cached_graph(request.provider)
        initial_state = {
            "question": request.query,
            "context": [],
            "answer": "",
            "score": 0.0,
            "retry_count": 0,
        }
        result = rag_graph.invoke(initial_state)

        return QueryResponse(
            query=request.query,
            final_answer=result["answer"],
            retrieved_context_chunks=result["context"],
            confidence_score=result["score"],
        )
    except HTTPException:
        # Re-raise explicit HTTP exceptions without altering their status code
        raise
    except ValueError as ve:
        logger.warning(f"Bad request parameters in query endpoint: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error executing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the request. Please consult server logs.",
        )
