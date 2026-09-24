import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
POLICY_DOCS_DIR = os.getenv("POLICY_DOCS_DIR", "./data/sample_policies")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # free on Groq
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # runs locally, free