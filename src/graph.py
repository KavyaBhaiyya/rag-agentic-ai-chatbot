"""LangGraph state machine for the RAG pipeline.

Flow:
    validate_input --(invalid)--> reject --> END
                    --(valid)--> retrieve --> relevance_gate
                                                  --(pass)--> generate --> postprocess --> END
                                                  --(fail)--> fallback --> postprocess --> END
"""
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END

from src.guardrails import (
    validate_query,
    compute_retrieval_confidence,
    passes_relevance_gate,
    passes_groundedness_gate,
    looks_like_refusal,
    sanitize_llm_output,
)
from src.logger import get_logger

log = get_logger(__name__)

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided document to answer that. "
    "Try rephrasing, or ask something more directly covered by the Agentic AI ebook."
)


class RAGState(TypedDict, total=False):
    raw_query: str
    query: str
    is_valid: bool
    error: Optional[str]
    retrieved: List[Dict]
    relevance_passed: bool
    answer: str
    confidence: float
    groundedness: Optional[float]
    used_fallback: bool


def build_graph(embedder, vectorstore, llm_client):
    """Wire the pipeline. Dependencies are injected so nodes are unit-testable
    with fakes/mocks, and the graph itself has no network calls of its own."""

    def validate_input(state: RAGState) -> RAGState:
        result = validate_query(state["raw_query"])
        if not result.is_valid:
            log.info(f"Rejected query: {result.reason}")
            return {
                **state,
                "is_valid": False,
                "error": result.reason,
                "query": result.sanitized_query or "",
            }
        return {**state, "is_valid": True, "query": result.sanitized_query}

    def retrieve(state: RAGState) -> RAGState:
        try:
            vec = embedder.embed_query(state["query"])
            matches = vectorstore.query(vec, top_k=None)
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return {**state, "retrieved": [], "error": f"Retrieval failed: {e}"}
        return {**state, "retrieved": matches}

    def relevance_gate(state: RAGState) -> RAGState:
        passed = passes_relevance_gate(state.get("retrieved", []))
        return {**state, "relevance_passed": passed}

    def generate(state: RAGState) -> RAGState:
        chunks = state["retrieved"]
        try:
            answer = llm_client.generate_answer(state["query"], chunks)
            answer = sanitize_llm_output(answer)
        except Exception as e:
            log.error(f"Generation failed: {e}")
            return {
                **state,
                "answer": "The answer service is temporarily unavailable. Please try again shortly.",
                "used_fallback": True,
                "error": str(e),
                "confidence": 0.0,
                "groundedness": None,
            }

        retrieval_conf = compute_retrieval_confidence(chunks)

        if looks_like_refusal(answer):
            log.info("Generated answer reads as a refusal; marking used_fallback=True.")
            return {
                **state,
                "answer": answer,
                "used_fallback": True,
                "confidence": retrieval_conf,
                "groundedness": None,
            }

        groundedness = llm_client.grade_groundedness(state["query"], answer, chunks)

        if groundedness is None:
            log.warning("Groundedness unmeasured for this answer; skipping the groundedness gate.")
            return {
                **state,
                "answer": answer,
                "used_fallback": False,
                "confidence": retrieval_conf,
                "groundedness": None,
            }

        overall_confidence = round(0.5 * retrieval_conf + 0.5 * groundedness, 3)

        if not passes_groundedness_gate(groundedness):
            log.info(f"Answer failed groundedness gate ({groundedness}); showing fallback instead.")
            return {
                **state,
                "answer": (
                    "I found related content in the document, but couldn't produce an "
                    "answer confidently grounded in it. Try rephrasing your question to "
                    "be more specific."
                ),
                "used_fallback": True,
                "confidence": overall_confidence,
                "groundedness": groundedness,
            }

        return {
            **state,
            "answer": answer,
            "used_fallback": False,
            "confidence": overall_confidence,
            "groundedness": groundedness,
        }

    def fallback(state: RAGState) -> RAGState:
        return {
            **state,
            "answer": FALLBACK_MESSAGE,
            "used_fallback": True,
            "confidence": compute_retrieval_confidence(state.get("retrieved", [])),
            "groundedness": 0.0,
        }

    def reject(state: RAGState) -> RAGState:
        return {
            **state,
            "answer": state.get("error") or "Invalid query.",
            "used_fallback": True,
            "retrieved": [],
            "confidence": 0.0,
            "groundedness": 0.0,
        }

    def postprocess(state: RAGState) -> RAGState:
        # final structured shape is assembled by the caller (api.py); this
        # node exists so logging/metrics hooks have one clear exit point.
        log.info(
            f"query='{state.get('query', '')[:60]}' "
            f"fallback={state.get('used_fallback')} "
            f"confidence={state.get('confidence')}"
        )
        return state

    graph = StateGraph(RAGState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("retrieve", retrieve)
    graph.add_node("relevance_gate", relevance_gate)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_node("reject", reject)
    graph.add_node("postprocess", postprocess)

    graph.set_entry_point("validate_input")

    graph.add_conditional_edges(
        "validate_input",
        lambda s: "retrieve" if s.get("is_valid") else "reject",
        {"retrieve": "retrieve", "reject": "reject"},
    )
    graph.add_edge("retrieve", "relevance_gate")
    graph.add_conditional_edges(
        "relevance_gate",
        lambda s: "generate" if s.get("relevance_passed") else "fallback",
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", "postprocess")
    graph.add_edge("fallback", "postprocess")
    graph.add_edge("reject", "postprocess")
    graph.add_edge("postprocess", END)

    return graph.compile()


def run_query(app, raw_query: str) -> Dict:
    """Run the compiled graph and return the API-shaped result dict."""
    final_state = app.invoke({"raw_query": raw_query})
    return {
        "answer": final_state.get("answer", ""),
        "sources": [
            {
                "chunk_id": m["id"],
                "page_number": m["page_number"],
                "score": round(m["score"], 4),
                "text": m["text"],
            }
            for m in final_state.get("retrieved", [])
        ],
        "confidence": final_state.get("confidence", 0.0),
        "groundedness": final_state.get("groundedness", 0.0),
        "used_fallback": final_state.get("used_fallback", False),
        "error": final_state.get("error"),
    }
