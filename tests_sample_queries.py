"""
Automated Benchmark Testing Script for Agentic AI eBook RAG System.

Runs the 6 benchmark queries defined in Appening AI Assignment:
1. Definition & Scope: "What is the core definition of Agentic AI as outlined in the eBook?"
2. Architecture & Paradigms: "What are the main architectural components required to build agentic systems?"
3. Use Cases: "What real-world industry use cases for Agentic AI are discussed in the eBook?"
4. Comparison: "How does Agentic AI differ from traditional generative AI chatbots according to the text?"
5. Challenges & Considerations: "What key challenges or limitations of Agentic AI are mentioned in the document?"
6. Out-of-Scope Test: "What is the capital of France?" (Validation test: system must refuse).

Supports:
- Live mode (using configured Pinecone index and active LLM provider)
- Offline mock mode (--mock, for pipeline and contract verification without live API keys)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.config import AIProviderConfig, Settings, get_provider_config, get_settings
from src.graph import REFUSAL_PHRASE, build_rag_graph, query_rag

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "category": "Definition & Scope",
        "query": "What is the core definition of Agentic AI as outlined in the eBook?",
        "expected_type": "grounded",
        "description": "Core definition of Agentic AI as outlined in eBook",
    },
    {
        "id": 2,
        "category": "Architecture & Paradigms",
        "query": "What are the main architectural components required to build agentic systems?",
        "expected_type": "grounded",
        "description": "Architectural breakdown (perception, planning, execution, memory, tools)",
    },
    {
        "id": 3,
        "category": "Use Cases",
        "query": "What real-world industry use cases for Agentic AI are discussed in the eBook?",
        "expected_type": "grounded",
        "description": "Real-world industry applications and use cases",
    },
    {
        "id": 4,
        "category": "Comparison",
        "query": "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "expected_type": "grounded",
        "description": "Comparison between Agentic AI and traditional GenAI/LLM chatbots",
    },
    {
        "id": 5,
        "category": "Challenges & Considerations",
        "query": "What key challenges or limitations of Agentic AI are mentioned in the document?",
        "expected_type": "grounded",
        "description": "Key challenges, governance, and limitations",
    },
    {
        "id": 6,
        "category": "Out-of-Scope Test (Groundedness Check)",
        "query": "What is the capital of France?",
        "expected_type": "refusal",
        "description": "Out-of-domain query (system must refuse to answer or state lack of info)",
    },
]


def create_mock_rag_graph():
    """
    Creates a simulated RAG graph for offline contract and routing verification.
    Supports both GradeDocuments and GroundedResponse structured outputs.
    """
    from src.graph import GradeDocuments, GroundedResponse

    mock_llm = MagicMock()

    def mock_structured_factory(schema):
        mock_handler = MagicMock()
        if schema == GradeDocuments:
            def grade_invoke(messages):
                prompt_text = str(messages)
                if "France" in prompt_text or "FIFA" in prompt_text:
                    return GradeDocuments(binary_score="no", relevance_score=0.0)
                return GradeDocuments(binary_score="yes", relevance_score=0.95)
            mock_handler.invoke.side_effect = grade_invoke
        elif schema == GroundedResponse:
            def generate_invoke(messages):
                prompt_text = str(messages)
                if "France" in prompt_text or "FIFA" in prompt_text:
                    return GroundedResponse(answer=REFUSAL_PHRASE, confidence=0.0)
                lower_p = prompt_text.lower()
                if "core definition" in lower_p or "definition" in lower_p:
                    mock_conf = 0.98
                elif "architectural" in lower_p:
                    mock_conf = 0.94
                elif "use cases" in lower_p:
                    mock_conf = 0.91
                elif "differ" in lower_p:
                    mock_conf = 0.96
                elif "challenges" in lower_p:
                    mock_conf = 0.88
                else:
                    mock_conf = 0.92
                return GroundedResponse(
                    answer="Based on the provided eBook, Agentic AI represents autonomous systems capable of reasoning, dynamic planning, tool utilization, and goal achievement through iterative workflows.",
                    confidence=mock_conf,
                )
            mock_handler.invoke.side_effect = generate_invoke
        else:
            mock_handler.invoke.return_value = MagicMock()
        return mock_handler

    mock_llm.with_structured_output.side_effect = mock_structured_factory
    mock_llm.invoke.return_value = MagicMock(content="agentic ai architecture components")

    mock_vs = MagicMock()
    mock_retriever = MagicMock()

    def mock_retrieve(query):
        if "France" in query or "FIFA" in query:
            # Irrelevant or no matching chunks
            return []
        return [
            Document(
                page_content="Agentic AI eBook Chapter 1: Agentic AI systems are autonomous agents "
                "combining reasoning, memory, and tool invocation to solve complex multi-step objectives."
            ),
            Document(
                page_content="Traditional automation follows rigid, pre-defined decision trees, whereas "
                "Agentic AI perceives environment feedback and dynamically adapts its execution plan."
            ),
            Document(
                page_content="Core architectural components of Agentic systems include Planning, Memory, "
                "Tool Use, and Reflection mechanisms."
            ),
        ]

    mock_retriever.invoke.side_effect = mock_retrieve
    mock_vs.as_retriever.return_value = mock_retriever

    mock_provider_cfg = AIProviderConfig(
        provider_name="mock-gemini",
        llm=mock_llm,
        embeddings=MagicMock(),
        dimension=768,
        model_name="mock-model",
        embedding_model_name="mock-embed",
    )

    return build_rag_graph(provider_config=mock_provider_cfg, vectorstore=mock_vs)


def run_benchmark(mock: bool = False, provider: str | None = None) -> bool:
    """
    Executes benchmark queries and prints evaluation table.
    """
    settings = get_settings()
    active_provider = provider or settings.llm_provider

    print("=" * 80)
    print(f"AGENTIC AI RAG BENCHMARK EVALUATION")
    print(f"Mode:     {'OFFLINE MOCK TEST' if mock else 'LIVE VECTOR DATABASE'}")
    print(f"Provider: {active_provider}")
    print("=" * 80)

    if mock:
        graph = create_mock_rag_graph()
    else:
        try:
            graph = build_rag_graph(provider=active_provider)
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize live RAG graph: {e}")
            print("\nHint: To test without live API keys, run with `--mock`:")
            print("      python tests_sample_queries.py --mock")
            return False

    results: List[Dict[str, Any]] = []
    all_passed = True

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        query = item["query"]
        expected = item["expected_type"]

        total_q = len(BENCHMARK_QUERIES)
        print(f"\n[{q_id}/{total_q}] Testing: '{query}'")
        start_time = time.time()

        initial_state = {
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0,
            "retry_count": 0,
        }

        try:
            res = graph.invoke(initial_state)
            elapsed = time.time() - start_time
            answer = res.get("answer", "")
            score = res.get("score", 0.0)
            context_count = len(res.get("context", []))

            # Grounding verification logic
            if expected == "refusal":
                refused = (
                    score == 0.0
                    or "cannot answer based on the provided document" in answer.lower()
                    or "not contain enough info" in answer.lower()
                )
                passed = refused
            else:
                passed = score > 0.0 and len(answer) > 20

            if not passed:
                all_passed = False

            status_str = "PASS [OK]" if passed else "FAIL [X]"

            print(f"   Status:    {status_str}")
            print(f"   Score:     {score:.2f}")
            print(f"   Chunks:    {context_count}")
            print(f"   Latency:   {elapsed:.2f}s")
            print(f"   Answer:    {answer[:160]}...")

            results.append(
                {
                    "id": q_id,
                    "query": query,
                    "expected": expected,
                    "score": score,
                    "chunks": context_count,
                    "passed": passed,
                    "latency": elapsed,
                    "answer_preview": answer[:120],
                }
            )

        except Exception as e:
            print(f"   Status:    ERROR ({e})")
            all_passed = False
            results.append(
                {
                    "id": q_id,
                    "query": query,
                    "expected": expected,
                    "score": 0.0,
                    "chunks": 0,
                    "passed": False,
                    "latency": 0.0,
                    "answer_preview": f"Error: {e}",
                }
            )

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"{'ID':<3} | {'Query':<38} | {'Type':<8} | {'Score':<5} | {'Chunks':<6} | {'Result':<6}")
    print("-" * 80)
    for r in results:
        res_label = "PASS" if r["passed"] else "FAIL"
        short_q = (r["query"][:35] + "...") if len(r["query"]) > 38 else r["query"]
        print(
            f"{r['id']:<3} | {short_q:<38} | {r['expected']:<8} | {r['score']:<5.2f} | {r['chunks']:<6} | {res_label:<6}"
        )
    print("=" * 80)

    if all_passed:
        print(f"[SUCCESS] All {len(BENCHMARK_QUERIES)} benchmark test queries passed verification!")
    else:
        print("[WARNING] Some queries did not meet the grounding criteria.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark tests")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline mock test verifying pipeline without live API keys",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["gemini", "openai"],
        help="Select AI provider for testing",
    )

    args = parser.parse_args()
    success = run_benchmark(mock=args.mock, provider=args.provider)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
