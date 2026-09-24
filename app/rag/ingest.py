"""
Loads HR policy documents from disk, splits them into chunks, embeds them,
and saves them into a local FAISS index.

Run standalone with: python -m app.rag.ingest
"""
import os
import glob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import FAISS_INDEX_DIR, POLICY_DOCS_DIR, EMBEDDING_MODEL


def load_documents(docs_dir: str = POLICY_DOCS_DIR):
    docs = []
    for path in glob.glob(os.path.join(docs_dir, "*.md")):
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
    return docs


def build_vector_store():
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No policy documents found in {POLICY_DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"Ingested {len(chunks)} chunks from {len(documents)} documents "
          f"into '{FAISS_INDEX_DIR}'.")
    return vector_store


if __name__ == "__main__":
    build_vector_store()