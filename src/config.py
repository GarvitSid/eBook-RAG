"""
Configuration and Multi-Provider AI Factory for Agentic AI eBook RAG.

Supports dual AI providers:
- 'gemini': Google Gemini (gemini-3.5-flash-lite / gemini-embedding-2, dimension 768)
- 'openai': OpenAI (gpt-4.1-mini / text-embedding-3-small, dimension 1536)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables from .env file
load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    # Provider configuration: 'gemini' or 'openai'
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    )

    # API Keys
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    pinecone_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY")
    )

    # Pinecone Vector DB Configuration
    pinecone_index_name: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index").strip()
    )
    pinecone_cloud: str = field(
        default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws").strip()
    )
    pinecone_region: str = field(
        default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1").strip()
    )

    # Ingestion & RAG Configuration
    pdf_path: str = field(
        default_factory=lambda: os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf").strip()
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "3"))
    )

    # Model specific settings (2026 Active Production Models)
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    )


@dataclass
class AIProviderConfig:
    """Encapsulates the LLM, Embeddings, and vector dimension for a provider."""

    provider_name: str
    llm: BaseChatModel
    embeddings: Embeddings
    dimension: int
    model_name: str
    embedding_model_name: str


def get_settings() -> Settings:
    """Retrieve application settings."""
    return Settings()


# In-memory index discovery cache with TTL to prevent redundant Pinecone network calls
_index_cache: Dict[str, Any] = {}


def get_index_name(provider: Optional[str] = None, settings: Optional[Settings] = None) -> str:
    """
    Get the configured Pinecone index name.
    
    Supports provider-specific indices ('agentic-ai-index-gemini' / 'agentic-ai-index-openai')
    as documented in README, while seamlessly maintaining backward compatibility with 
    a single shared index ('agentic-ai-index').
    """
    cfg = settings or get_settings()
    base_name = cfg.pinecone_index_name.strip()
    active_provider = (provider or cfg.llm_provider).strip().lower()

    # If base_name is already provider-suffixed, return it directly
    if base_name.endswith(f"-{active_provider}"):
        return base_name

    suffixed_name = f"{base_name}-{active_provider}"

    # Explicit provider-specific partitioning override
    if os.getenv("PINECONE_INDEX_PER_PROVIDER", "").strip().lower() in ("true", "1"):
        return suffixed_name

    # Check Pinecone for existing index names if API key is present
    if cfg.pinecone_api_key:
        try:
            now = time.time()
            if "indices" not in _index_cache or now - _index_cache.get("time", 0) > 30:
                from pinecone import Pinecone
                pc = Pinecone(api_key=cfg.pinecone_api_key)
                _index_cache["indices"] = [idx.name for idx in pc.list_indexes()]
                _index_cache["time"] = now

            existing = _index_cache.get("indices", [])
            # If provider-suffixed index exists (e.g. agentic-ai-index-gemini), prefer it
            if suffixed_name in existing:
                return suffixed_name
            # If base index exists (e.g. agentic-ai-index), use it
            if base_name in existing:
                return base_name
        except Exception:
            pass

    return base_name


def get_provider_config(
    provider: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> AIProviderConfig:
    """
    Factory function to instantiate LLM and Embeddings based on the active provider.
    Complies with 2026 Production Standards:
      - 'openai': ChatOpenAI(model="gpt-4.1-mini", temperature=0) + OpenAIEmbeddings(1536 dims)
      - 'gemini': ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0) +
                  GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", output_dimensionality=768)
    """
    cfg = settings or get_settings()
    active_provider = (provider or cfg.llm_provider).strip().lower()

    if active_provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in environment or .env file. "
                "Please configure OPENAI_API_KEY to use the 'openai' provider."
            )

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(
            model=cfg.openai_model,
            temperature=0,
            api_key=cfg.openai_api_key,
        )
        embeddings = OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.openai_api_key,
        )
        return AIProviderConfig(
            provider_name="openai",
            llm=llm,
            embeddings=embeddings,
            dimension=1536,
            model_name=cfg.openai_model,
            embedding_model_name=cfg.openai_embedding_model,
        )

    elif active_provider == "gemini":
        if not cfg.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set in environment or .env file. "
                "Please configure GOOGLE_API_KEY to use the 'gemini' provider."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(
            model=cfg.gemini_model,
            temperature=0,
            google_api_key=cfg.google_api_key,
        )
        # Note: Google's gemini-embedding-2 natively outputs 3072 dimensions.
        # We leverage Matryoshka Representation Learning (output_dimensionality=768)
        # to optimize vector storage and query latency in Pinecone Serverless while preserving high retrieval recall.
        embeddings = GoogleGenerativeAIEmbeddings(
            model=cfg.gemini_embedding_model,
            google_api_key=cfg.google_api_key,
            output_dimensionality=768,
        )
        return AIProviderConfig(
            provider_name="gemini",
            llm=llm,
            embeddings=embeddings,
            dimension=768,
            model_name=cfg.gemini_model,
            embedding_model_name=cfg.gemini_embedding_model,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{active_provider}'. Supported values are 'gemini' and 'openai'."
        )
