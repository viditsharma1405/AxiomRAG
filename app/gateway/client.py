import os
import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from groq import Groq

from app.config import settings

# If a saved config ID (e.g. pc-...) is provided in .env, use it; otherwise route directly via virtual key slug
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG_ID if PORTKEY_CONFIG_ID else None
)

groq_client = Groq(api_key=settings.GROQ_API_KEY)


def get_langchain_llm(feature: str = "rag"):
    """
    Returns a Portkey-backed ChatOpenAI with an automatic fallback to ChatGroq
    if the Portkey gateway encounters errors or outages.
    """
    headers_kwargs = {
        "api_key": settings.PORTKEY_API_KEY,
        "metadata": {
            "feature": feature,
            "_user": "rag-system",
            "environment": "production"
        }
    }
    if PORTKEY_CONFIG_ID:
        headers_kwargs["config"] = PORTKEY_CONFIG_ID

    portkey_llm = ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,
        default_headers=createHeaders(**headers_kwargs)
    )
    fallback_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )
    return portkey_llm.with_fallbacks([fallback_llm])

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"