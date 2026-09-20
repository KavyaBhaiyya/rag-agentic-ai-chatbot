"""Simple Streamlit UI for the RAG chatbot.

Run with:  streamlit run streamlit_app.py
"""
import streamlit as st

from src.vectorstore import get_pinecone_client, VectorStore, VectorStoreError
from src.embeddings import Embedder, EmbeddingError
from src.llm import LLMClient, LLMError
from src.graph import build_graph, run_query

st.set_page_config(page_title="Agentic AI Ebook Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI Ebook Chatbot")
st.caption("Answers are grounded strictly in Konverge.ai's Agentic AI ebook.")


@st.cache_resource
def load_pipeline():
    pc = get_pinecone_client()
    vectorstore = VectorStore(pc)
    vectorstore.ensure_index()
    embedder = Embedder(pc)
    llm_client = LLMClient()
    return build_graph(embedder, vectorstore, llm_client)


try:
    pipeline = load_pipeline()
    init_error = None
except (VectorStoreError, LLMError, EmbeddingError) as e:
    pipeline = None
    init_error = str(e)

if init_error:
    st.error(
        f"Setup problem: {init_error}\n\n"
        "Check your .env file (PINECONE_API_KEY, GROQ_API_KEY) and make sure "
        "you've run `python -m src.ingest` at least once."
    )
    st.stop()

question = st.text_input("Ask a question about the ebook:", placeholder="What is agentic AI?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Thinking..."):
        try:
            result = run_query(pipeline, question)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    if result["error"] and result["used_fallback"] and not result["sources"]:
        st.warning(result["answer"])
    else:
        st.markdown("### Answer")
        st.write(result["answer"])

        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{result['confidence']:.2f}")
        groundedness_display = (
            f"{result['groundedness']:.2f}" if result["groundedness"] is not None else "N/A"
        )
        col2.metric("Groundedness", groundedness_display)
        
        if result["used_fallback"]:
            st.info("The system did not find strong enough support in the document for a direct answer.")

        if result["sources"]:
            with st.expander(f"Retrieved source chunks ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(f"**Page {s['page_number']} · score {s['score']:.3f}**")
                    st.write(s["text"])
                    st.divider()
