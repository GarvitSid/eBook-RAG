"""
LangGraph Corrective & Self-Reflective Agentic RAG Workflow.

Stateful Cyclic Graph Architecture:
                     +-------------------+
                     |  LangGraph START  |
                     +---------+---------+
                               |
                               v
                     +-------------------+
                     |   retrieve_node   | <──┐ (Cycle from query rewrite)
                     +---------+---------+    │
                               |              │
                               v              │
                     +-------------------+    │
                     |  grade_documents  |    │
                     +---------+---------+    │
                               |              │
                   [ should_generate? ]       │
                     /               \        │
        (Relevant)  /                 \  (Irrelevant / Out-of-scope)
                   v                   v      │
         +-------------------+  +-------------------+
         |   generate_node   |  |   refusal_node    |
         +---------+---------+  +---------+---------+
                   |                      |
                   v                      v
           [ is_grounded? ]              END
              /          \
         (Valid)      (Low confidence / Hallucinated)
            /              \
           v                v
          END       +--------------------+
                    | rewrite_query_node |────┘
                    +--------------------+ (max_retries=1)

Features:
- TypedDict AgentState with retry_count tracking
- Binary document relevance grading via structured output (GradeDocuments)
- Short-circuiting out-of-scope queries (saves tokens and reduces latency)
- Strict document grounding prompt & structured output (GroundedResponse)
- Hallucination / confidence verification conditional edge
- Cyclic query reformulation node with infinite-loop prevention
- Provider-agnostic factory integration (Gemini & OpenAI)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from pydantic import BaseModel, Field

from src.config import (
    AIProviderConfig,
    Settings,
    get_index_name,
    get_provider_config,
    get_settings,
)

logger = logging.getLogger("graph")

# Strict System Prompt Formulation per SDD Section 4
STRICT_SYSTEM_PROMPT = """You are a strict assistant. Answer the question relying ONLY on the context below. If the context does not contain enough info, state 'I cannot answer based on the provided document.' Do not use outside knowledge."""

REFUSAL_PHRASE = "I cannot answer based on the provided document."


class GradeDocuments(BaseModel):
    """Relevance check and dynamic scoring on retrieved documents."""

    binary_score: str = Field(
        ...,
        description="Relevance check: 'yes' if retrieved context contains information related to answering the question, or 'no' if the question is out of scope / unrelated to the eBook.",
    )
    relevance_score: float = Field(
        default=0.9,
        description="Score between 0.0 and 1.0 indicating how strongly the retrieved documents match the query topic.",
    )


class GroundedResponse(BaseModel):
    """Structured response schema combining synthesized answer and dynamic confidence score."""

    answer: str = Field(
        ...,
        description="The final synthesized answer addressing the user's specific query based strictly on the context. If the context does not contain enough info, state 'I cannot answer based on the provided document.'",
    )
    confidence: float = Field(
        ...,
        description="A score between 0.0 and 1.0 indicating how fully and accurately the provided context answers the question. Use 0.0 if the context lacks information or is out of scope.",
    )


class AgentState(TypedDict):
    """
    Schema for the LangGraph state machine.
      question: The user query (can be rewritten during cyclic correction)
      context: List of raw retrieved text chunks
      answer: Final generated response
      score: Confidence/grounding score (0.0 to 1.0)
      retry_count: Counter to track cyclic retries and prevent infinite loops
    """

    question: str
    context: List[str]
    answer: str
    score: float
    retry_count: int


def build_rag_graph(
    provider: Optional[str] = None,
    provider_config: Optional[AIProviderConfig] = None,
    index_name: Optional[str] = None,
    vectorstore: Optional[PineconeVectorStore] = None,
    settings: Optional[Settings] = None,
):
    """
    Builds and compiles the stateful Corrective LangGraph RAG workflow.
    Agnostic to AI provider (supports both Gemini and OpenAI seamlessly).
    """
    cfg = settings or get_settings()
    active_provider = (provider or cfg.llm_provider).strip().lower()
    target_index_name = index_name or get_index_name(active_provider, cfg)

    # 1. Resolve Provider Objects (LLM & Embeddings)
    resolved_provider_config = provider_config or get_provider_config(active_provider, cfg)
    llm = resolved_provider_config.llm

    # 2. Resolve VectorStore Retriever
    if vectorstore is not None:
        vs = vectorstore
    else:
        if not cfg.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set. Please add it to your .env file or environment variables."
            )
        pc = Pinecone(api_key=cfg.pinecone_api_key)
        vs = PineconeVectorStore(
            index=pc.Index(target_index_name),
            embedding=resolved_provider_config.embeddings,
        )

    retriever = vs.as_retriever(search_kwargs={"k": cfg.top_k})

    # 3. Helper for transient API retries
    def call_structured_with_retry(active_llm, schema_cls, messages, max_retries: int = 2) -> Optional[Any]:
        structured = active_llm.with_structured_output(schema_cls)
        for attempt in range(1, max_retries + 1):
            try:
                return structured.invoke(messages)
            except Exception as ex:
                err = str(ex).lower()
                is_transient = any(
                    s in err
                    for s in ("429", "503", "resource_exhausted", "unavailable", "overloaded", "quota", "rate", "deadline", "timeout")
                )
                if is_transient and attempt < max_retries:
                    delay = 1.5 * attempt
                    logger.warning(f"LLM call transient error (attempt {attempt}/{max_retries}): {ex}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise ex
        return None

    # 4. Define Graph Nodes
    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """
        Retrieves top-k relevant document chunks from Pinecone.
        Includes exponential backoff retry to handle transient embedding API limits.
        """
        query = state.get("question", "").strip()
        if not query:
            return {"context": [], "score": 0.0}

        max_retries = 3
        docs = []
        for attempt in range(1, max_retries + 1):
            try:
                docs = retriever.invoke(query)
                break
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(
                    sig in err_str
                    for sig in ("429", "503", "resource_exhausted", "unavailable", "overloaded", "quota", "rate", "deadline", "timeout")
                )
                if is_transient and attempt < max_retries:
                    wait_sec = 2 * attempt
                    logger.warning(
                        f"Transient error in retrieve_node embedding call (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {wait_sec}s..."
                    )
                    time.sleep(wait_sec)
                else:
                    logger.error(f"Failed to retrieve chunks from Pinecone after {attempt} attempts: {e}")
                    return {"context": [], "score": 0.0}

        context_texts = [d.page_content for d in docs if d.page_content.strip()]
        return {"context": context_texts}

    def grade_documents_node(state: AgentState) -> Dict[str, Any]:
        """
        Agentic Document Grader:
        Evaluates whether retrieved chunks contain information relevant to answering the question.
        Enables short-circuiting out-of-scope queries before calling generation.
        """
        context_list = state.get("context", [])
        question = state.get("question", "")

        if not context_list:
            logger.info("[Agentic Grader] No context chunks retrieved. Score = 0.0.")
            return {"score": 0.0}

        # Truncate preview of context for grading efficiency
        preview = "\n---\n".join(chunk[:400] for chunk in context_list[:3])
        grading_prompt = [
            SystemMessage(
                content="You are a strict relevance grader evaluating retrieved chunks from a technical eBook on Agentic AI. "
                "Determine if the chunks contain information relevant to answering the user question. "
                "Output binary_score as 'yes' if relevant, or 'no' if the question is off-topic / out-of-scope."
            ),
            HumanMessage(content=f"User Question: {question}\n\nRetrieved Chunks:\n{preview}"),
        ]

        try:
            grade_result = call_structured_with_retry(llm, GradeDocuments, grading_prompt, max_retries=2)
            if isinstance(grade_result, GradeDocuments) and grade_result.binary_score.strip().lower() == "yes":
                doc_score = float(grade_result.relevance_score) if grade_result.relevance_score > 0 else 0.90
                logger.info(f"[Agentic Grader] Query '{question[:40]}...' deemed RELEVANT (relevance={doc_score:.2f}).")
                return {"score": round(doc_score, 2)}
            else:
                logger.info(f"[Agentic Grader] Query '{question[:40]}...' deemed IRRELEVANT (no).")
                return {"score": 0.0}
        except Exception as e:
            logger.warning(f"[Agentic Grader] Grader encountered error ({e}); passing to generation conservatively.")
            return {"score": 0.5}

    def should_generate(state: AgentState) -> str:
        """
        Conditional Edge 1:
        Routes to 'generate' if documents are relevant, or 'refusal' if out of scope.
        """
        score = state.get("score", 0.0)
        context = state.get("context", [])
        if score == 0.0 or not context:
            logger.info("[Agentic Router] Routing to 'refusal' (out-of-scope / zero relevance).")
            return "refusal"
        logger.info("[Agentic Router] Routing to 'generate' (context verified).")
        return "generate"

    def refusal_node(state: AgentState) -> Dict[str, Any]:
        """
        Refusal node:
        Immediately outputs refusal without wasting generation tokens.
        """
        return {
            "answer": REFUSAL_PHRASE,
            "score": 0.0,
        }

    def generate_node(state: AgentState) -> Dict[str, Any]:
        """
        Strict generation node:
        Prompts LLM to rely strictly on retrieved context chunks.
        Includes robust retry, exponential backoff, and cascading fallback models.
        """
        context_list = state.get("context", [])
        question = state.get("question", "")

        if not context_list:
            return {
                "answer": REFUSAL_PHRASE,
                "score": 0.0,
            }

        context_str = "\n\n---\n\n".join(context_list)
        messages = [
            SystemMessage(
                content=f"{STRICT_SYSTEM_PROMPT}\nProvide a direct, synthesized answer addressing the user's specific query based on the context. Do not output or recite entire raw chunks verbatim."
            ),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {question}"),
        ]

        response = None
        # Attempt 1: Try primary configured LLM with retry
        try:
            response = call_structured_with_retry(llm, GroundedResponse, messages, max_retries=2)
        except Exception as e:
            err_str = str(e).lower()
            is_quota_or_overload = any(
                s in err_str
                for s in ("429", "503", "resource_exhausted", "unavailable", "overloaded", "quota", "rate", "deadline", "timeout")
            )
            if active_provider == "gemini" and is_quota_or_overload:
                fallback_model = "gemini-2.5-flash" if cfg.gemini_model == "gemini-3.5-flash-lite" else "gemini-3.5-flash-lite"
                logger.warning(
                    f"Primary Gemini model '{cfg.gemini_model}' hit quota/overload limit ({e}). "
                    f"Cascading to fallback '{fallback_model}'..."
                )
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    fallback_llm = ChatGoogleGenerativeAI(
                        model=fallback_model,
                        temperature=0,
                        google_api_key=cfg.google_api_key,
                    )
                    response = call_structured_with_retry(fallback_llm, GroundedResponse, messages, max_retries=2)
                except Exception as fallback_err:
                    logger.error(f"Fallback model also encountered an error: {fallback_err}")
                    return {
                        "answer": "The AI service is temporarily experiencing high global traffic or quota limits. Please retry in a few moments.",
                        "score": 0.0,
                    }
            else:
                logger.error(f"Error during LLM generation: {e}")
                return {
                    "answer": "An unexpected error occurred while generating the response. Please check system logs.",
                    "score": 0.0,
                }

        # Unpack structured response
        if isinstance(response, GroundedResponse):
            answer_text = response.answer
            confidence_score = float(response.confidence)
        else:
            answer_text = REFUSAL_PHRASE
            confidence_score = 0.0

        # Enforce refusal if low confidence or refusal indicators found
        lower_ans = answer_text.strip().lower()
        if confidence_score < 0.3 or "cannot answer" in lower_ans:
            return {
                "answer": REFUSAL_PHRASE,
                "score": 0.0,
            }

        return {
            "answer": answer_text,
            "score": round(confidence_score, 2),
        }

    def grade_hallucination_and_grounding(state: AgentState) -> str:
        """
        Conditional Edge 2:
        Evaluates grounding of the answer. If grounded, finishes at END.
        If confidence is low or refusal occurred on attempt 0, triggers query rewrite cycle.
        """
        score = state.get("score", 0.0)
        retry_count = state.get("retry_count", 0)
        ans = state.get("answer", "").strip().lower()

        is_refusal = REFUSAL_PHRASE.lower() in ans or "cannot answer" in ans
        if score >= 0.3 and not is_refusal:
            logger.info("[Agentic Grounding] Response is well-grounded. Ending workflow.")
            return "grounded"

        # If low confidence or refusal, attempt self-correction via query rewrite (max 1 retry)
        if retry_count < 1:
            logger.info(f"[Agentic Grounding] Low confidence / refusal detected (score={score}). Triggering rewrite cycle...")
            return "rewrite"

        logger.info("[Agentic Grounding] Max retry limit reached. Routing to refusal.")
        return "refusal"

    def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
        """
        Agentic Self-Correction Node:
        Reformulates the question to improve semantic retrieval against the eBook,
        then cycles back to retrieve_node.
        """
        old_q = state.get("question", "")
        retry_count = state.get("retry_count", 0) + 1
        prompt = (
            f"You are an expert search query optimizer for a technical eBook on Agentic AI.\n"
            f"The original question was: '{old_q}'.\n"
            f"Formulate a clarified, targeted keyword-rich search query to retrieve relevant paragraphs from the eBook.\n"
            f"Return ONLY the reformulated query text without quotes or explanations."
        )
        try:
            rewritten_msg = llm.invoke(prompt)
            new_q = rewritten_msg.content if hasattr(rewritten_msg, "content") else str(rewritten_msg)
            new_q = new_q.strip().strip('"').strip("'")
            logger.info(f"[Agentic Rewriter] Reformulated query: '{old_q}' -> '{new_q}' (cycle {retry_count}/1)")
        except Exception as e:
            logger.warning(f"[Agentic Rewriter] Failed to rewrite query ({e}); keeping original.")
            new_q = old_q

        return {
            "question": new_q,
            "retry_count": retry_count,
        }

    # 5. Assemble StateGraph with Conditional Edges & Cycles
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("refusal", refusal_node)

    # Execution flow:
    # START -> retrieve -> grade_documents
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional Branch 1: grade_documents -> generate OR refusal
    workflow.add_conditional_edges(
        "grade_documents",
        should_generate,
        {
            "generate": "generate",
            "refusal": "refusal",
        },
    )

    # Conditional Branch 2: generate -> END (grounded) OR rewrite_query OR refusal
    workflow.add_conditional_edges(
        "generate",
        grade_hallucination_and_grounding,
        {
            "grounded": END,
            "rewrite": "rewrite_query",
            "refusal": "refusal",
        },
    )

    # Cyclic Edge: rewrite_query cycles back to retrieve!
    workflow.add_edge("rewrite_query", "retrieve")

    # Terminal Edge for refusal
    workflow.add_edge("refusal", END)

    # 6. Compile Workflow
    compiled_app = workflow.compile()
    return compiled_app


def query_rag(
    question: str,
    graph: Optional[Any] = None,
    provider: Optional[str] = None,
) -> AgentState:
    """
    Convenience function to run a single query through the RAG graph.
    Returns the final AgentState dictionary.
    """
    rag_app = graph or build_rag_graph(provider=provider)
    initial_state: AgentState = {
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0,
        "retry_count": 0,
    }
    result = rag_app.invoke(initial_state)
    return result
