import os
import sys
from dotenv import load_dotenv

# Load .env
load_dotenv()

def print_status(service: str, status: bool, msg: str = ""):
    symbol = "✅" if status else "❌"
    detail = f" — {msg}" if msg else ""
    print(f"{symbol} {service}{detail}")

def check_keys():
    print("=" * 60)
    print(" Enterprise Agentic RAG - Credential & Connectivity Check")
    print("=" * 60)
    
    # 1. Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key.startswith("gsk_"):
        print_status("Groq Primary Key", True, f"Found ({groq_key[:8]}...)")
    else:
        print_status("Groq Primary Key", False, "Missing or invalid format in .env (expected 'gsk_...')")

    # 2. Portkey
    portkey_key = os.getenv("PORTKEY_API_KEY")
    if portkey_key and len(portkey_key.strip()) > 5:
        print_status("Portkey Gateway Key", True, f"Found ({portkey_key[:6]}...)")
    else:
        print_status("Portkey Gateway Key", False, "Missing PORTKEY_API_KEY in .env")

    # 3. Gemini Embeddings
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print_status("Gemini API Key", True, f"Found ({gemini_key[:6]}...)")
    else:
        print_status("Gemini API Key", False, "Missing GEMINI_API_KEY in .env")

    # 4. Qdrant Cloud
    qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url and qdrant_key:
        print_status("Qdrant Credentials", True, f"URL: {qdrant_url}")
    else:
        missing = []
        if not qdrant_url: missing.append("QDRANT_CLUSTER_ENDPOINT")
        if not qdrant_key: missing.append("QDRANT_API_KEY")
        print_status("Qdrant Credentials", False, f"Missing: {', '.join(missing)}")

    # 5. Logfire
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        print_status("Pydantic Logfire Token", True, "Found")
    else:
        print_status("Pydantic Logfire Token", False, "Missing LOGFIRE_TOKEN in .env")

    # 6. LangSmith
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        print_status("LangSmith API Key", True, "Found")
    else:
        print_status("LangSmith API Key", False, "Missing LANGSMITH_API_KEY in .env (optional for local running, but recommended)")

    print("=" * 60)
    print("To test live connections after filling your keys, run:")
    print("  ./venv/bin/python -m scripts.verify_credentials --live")
    print("=" * 60)

if __name__ == "__main__":
    check_keys()
