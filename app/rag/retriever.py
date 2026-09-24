from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import FAISS_INDEX_DIR, EMBEDDING_MODEL

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vector_store = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,  # safe: this is our own locally-built index
        )
    return _vector_store


def retrieve_policy_chunks(query: str, k: int = 3):
    """Returns the top-k most relevant policy chunks for a query, with scores."""
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=k)
    return [
        {"content": doc.page_content, "source": doc.metadata.get("source", "unknown"), "score": float(score)}
        for doc, score in results
    ]