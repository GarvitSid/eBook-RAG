"""
Document Ingestion & Pinecone Vector Storage Pipeline for Agentic AI eBook.

ETL Process:
1. Load PDF with PyPDFLoader
2. Text Chunking with RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
3. Pinecone Index Setup with dimension safety check (1536 for OpenAI, 768 for Gemini)
4. Vector embedding and upsert to Pinecone
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    AIProviderConfig,
    Settings,
    get_index_name,
    get_provider_config,
    get_settings,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingestion")


def setup_pinecone_index(
    pinecone_client: Pinecone,
    index_name: str,
    dimension: int,
    cloud: str = "aws",
    region: str = "us-east-1",
    recreate: bool = False,
) -> None:
    """
    Ensure the Pinecone index exists with the correct dimension matching the embedding provider.
    Enforces the safety check required by SDD Section 5:
    - Verifies index dimension matches provider (1536 for OpenAI, 768 for Gemini).
    - If dimension mismatches and recreate is True, deletes and creates a fresh index.
    - If dimension mismatches and recreate is False, raises a descriptive error.
    """
    existing_indexes = [idx.name for idx in pinecone_client.list_indexes()]
    index_deleted = False

    if index_name in existing_indexes:
        desc = pinecone_client.describe_index(index_name)
        existing_dim = desc.dimension
        logger.info(
            f"Found existing Pinecone index '{index_name}' with dimension {existing_dim}."
        )

        if existing_dim != dimension:
            if recreate:
                logger.warning(
                    f"Dimension mismatch (existing={existing_dim}, required={dimension}). "
                    f"Deleting index '{index_name}' for recreation..."
                )
                pinecone_client.delete_index(index_name)
                # Brief pause for deletion propagation
                time.sleep(3)
                index_deleted = True
            else:
                raise ValueError(
                    f"Pinecone index '{index_name}' dimension mismatch: existing index has dimension "
                    f"{existing_dim}, but the active provider requires {dimension}. "
                    f"Run ingestion with `--recreate` to recreate the index or specify a different "
                    f"PINECONE_INDEX_NAME in your .env file."
                )

    # If index does not exist (or was deleted above), create it
    if index_name not in existing_indexes or index_deleted:
        logger.info(
            f"Creating Pinecone serverless index '{index_name}' "
            f"(dimension={dimension}, metric='cosine', cloud='{cloud}', region='{region}')..."
        )
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        # Wait until index is ready
        logger.info(f"Waiting for index '{index_name}' to become ready...")
        max_wait_secs = 60
        start_time = time.time()
        while True:
            desc = pinecone_client.describe_index(index_name)
            if desc.status.get("ready"):
                logger.info(f"Index '{index_name}' is ready!")
                break
            if time.time() - start_time > max_wait_secs:
                raise TimeoutError(f"Index '{index_name}' creation timed out after {max_wait_secs} seconds.")
            time.sleep(2)


def run_ingestion(
    pdf_path: Optional[str] = None,
    provider: Optional[str] = None,
    index_name: Optional[str] = None,
    recreate: bool = False,
    batch_size: int = 50,
    max_chunks: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> PineconeVectorStore:
    """
    Executes the complete ETL document ingestion pipeline:
    1. Loads PDF using PyPDFLoader
    2. Splits text into chunks using RecursiveCharacterTextSplitter
    3. Verifies / sets up Pinecone index
    4. Upserts chunk embeddings to Pinecone
    """
    cfg = settings or get_settings()
    active_pdf = pdf_path or cfg.pdf_path
    active_provider = (provider or cfg.llm_provider).strip().lower()
    target_index_name = index_name or get_index_name(active_provider, cfg)

    logger.info("=" * 60)
    logger.info("STARTING DOCUMENT INGESTION PIPELINE")
    logger.info(f"Provider:    {active_provider}")
    logger.info(f"PDF Source:  {active_pdf}")
    logger.info(f"Target Index:{target_index_name}")
    logger.info("=" * 60)

    # 1. Validate PDF file existence
    pdf_file = Path(active_pdf)
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"Source document not found at '{active_pdf}'. "
            f"Please ensure the Agentic AI eBook is placed in 'data/Ebook-Agentic-AI.pdf'."
        )

    # 2. Get provider configuration (validates keys & dimensions)
    provider_config: AIProviderConfig = get_provider_config(active_provider, cfg)
    logger.info(
        f"Active Embedding Model: {provider_config.embedding_model_name} "
        f"(Dimension: {provider_config.dimension})"
    )

    if not cfg.pinecone_api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. Please add it to your .env file or environment variables."
        )

    # 3. Load PDF
    logger.info(f"Loading PDF document from: {pdf_file.resolve()}...")
    loader = PyPDFLoader(str(pdf_file.resolve()))
    raw_pages = loader.load()

    # Filter out empty pages
    pages = [p for p in raw_pages if p.page_content.strip()]
    logger.info(f"Loaded {len(pages)} non-empty pages (total pages in PDF: {len(raw_pages)}).")

    # 4. Text Chunking
    logger.info(
        f"Splitting document into chunks (chunk_size={cfg.chunk_size}, chunk_overlap={cfg.chunk_overlap})..."
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    logger.info(f"Generated {len(chunks)} text chunks.")

    if max_chunks and max_chunks > 0:
        logger.info(f"Limiting ingestion to first {max_chunks} chunks for minimum API call testing.")
        chunks = chunks[:max_chunks]

    if not chunks:
        raise ValueError(f"No text chunks were generated from '{active_pdf}'. Check the PDF content.")

    # 5. Pinecone Index Setup
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    setup_pinecone_index(
        pinecone_client=pc,
        index_name=target_index_name,
        dimension=provider_config.dimension,
        cloud=cfg.pinecone_cloud,
        region=cfg.pinecone_region,
        recreate=recreate,
    )

    # 6. Upsert Embeddings to Pinecone in Batches with Rate-Limit Handling
    logger.info(
        f"Upserting {len(chunks)} chunks into Pinecone index '{target_index_name}' "
        f"in batches of {batch_size}..."
    )

    vector_store = PineconeVectorStore(
        index=pc.Index(target_index_name),
        embedding=provider_config.embeddings,
    )

    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    batch_num = 0

    for start_idx in range(0, total_chunks, batch_size):
        batch_num += 1
        end_idx = min(start_idx + batch_size, total_chunks)
        batch = chunks[start_idx:end_idx]
        logger.info(f"Uploading batch {batch_num}/{total_batches} (chunks {start_idx + 1} to {end_idx})...")

        batch_ids = [f"chunk_{i}" for i in range(start_idx, end_idx)]
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                vector_store.add_documents(documents=batch, ids=batch_ids)
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()) and attempt < max_retries:
                    wait_time = 15 * attempt
                    logger.warning(
                        f"Rate limit encountered on batch {batch_num} (attempt {attempt}/{max_retries}). "
                        f"Waiting {wait_time}s for Google API quota window to reset..."
                    )
                    time.sleep(wait_time)
                else:
                    raise

        # Respect Google Free Tier rate limits (100 RPM limit)
        if end_idx < total_chunks:
            time.sleep(6)

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE!")
    logger.info(f"Successfully indexed {len(chunks)} chunks in Pinecone index '{target_index_name}'.")
    logger.info("=" * 60)

    return vector_store


def main() -> None:
    """CLI entrypoint for standalone ingestion execution."""
    parser = argparse.ArgumentParser(description="Ingest Agentic AI eBook into Pinecone Vector Store")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["gemini", "openai"],
        help="Target AI provider (defaults to LLM_PROVIDER in .env)",
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to the PDF file (default: data/Ebook-Agentic-AI.pdf)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Pinecone index name (defaults to provider-aware index name)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Pinecone index if it exists with different dimensions or needs fresh data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of chunks to embed and upsert per batch (default: 25)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of chunks to ingest (for testing with minimum API calls)",
    )

    args = parser.parse_args()

    try:
        run_ingestion(
            pdf_path=args.pdf_path,
            provider=args.provider,
            index_name=args.index_name,
            recreate=args.recreate,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
