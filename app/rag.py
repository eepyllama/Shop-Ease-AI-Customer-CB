"""
RAG pipeline — adapted from ComplaintBot's setup_rag.py.

Original used: OllamaEmbeddings("nomic-embed-text") + Chroma + PyPDFLoader.
This version uses:
  - HuggingFace sentence-transformers (no API key, no local server needed)
  - OR OpenAI embeddings when OPENAI_API_KEY is set (better quality)
  - Chroma vector store (persistent, same as ComplaintBot)
  - Markdown/TXT policy file (or PDF if available)
"""
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

from app.config import CHROMA_DIR, POLICY_PATH, USE_OPENAI_EMBEDDINGS

_retriever = None  # cached after first build


def _build_embeddings():
    if USE_OPENAI_EMBEDDINGS:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def _load_policy_docs():
    path = Path(POLICY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    raw = loader.load()
    # Same chunking as ComplaintBot: 150 chars, 30 overlap
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(raw)


def build_retriever(force_rebuild: bool = False):
    """Build (or load cached) Chroma retriever over the policy document."""
    global _retriever
    if _retriever is not None and not force_rebuild:
        return _retriever

    embeddings = _build_embeddings()
    chroma_path = Path(CHROMA_DIR)

    if chroma_path.exists() and any(chroma_path.iterdir()) and not force_rebuild:
        # Load existing collection
        vectorstore = Chroma(
            collection_name="policy-rag",
            embedding_function=embeddings,
            persist_directory=str(chroma_path),
        )
    else:
        # Build from policy document
        docs = _load_policy_docs()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="policy-rag",
            persist_directory=str(chroma_path),
        )

    # Return top-2 chunks (same concept as ComplaintBot's k=1, but slightly wider)
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return _retriever


def get_rag_context(query: str) -> str:
    """
    Retrieve the most relevant policy sections for a complaint query.
    Returns a formatted string ready to be injected into the prompt.
    """
    try:
        retriever = build_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No specific policy found. Apply general fairness principles."
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        return f"Policy retrieval unavailable: {e}. Apply general fairness principles."
