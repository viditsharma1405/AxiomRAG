import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "OFF_TOPIC":
        logfire.info("Refusing off-topic request directly in responder.")
        content = "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I cannot assist with off-topic subjects like movies, entertainment, or pop culture — but feel free to ask me anything technical about your infrastructure!"
        return {
            "final_answer": content,
            "status": "Refused off-topic query.",
            "plan": state["plan"] + ["Refused Off-Topic Query"],
            "messages": [{"role": "assistant", "content": content}]
        }

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a professional Enterprise AI Assistant focused exclusively on Kubernetes, Intel hardware, and enterprise networking.
        Answer the user's latest greeting or conversation history question using the CONVERSATION HISTORY below.
        CRITICAL RULE: If the user asks about entertainment, movies, pop culture, personal tastes, or non-IT topics, decline politely and steer back to enterprise technical topics.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    from app.config import settings
    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Gateway: Portkey 🌐", "Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM (Portkey Gateway).")
                plan_update = state["plan"] + ["Gateway: Portkey 🌐"]
                status = "Response generated (via Portkey Gateway)."

        except Exception as e:
            logfire.warning(f"Portkey gateway unavailable ({e}), falling back to direct Groq.")
            from app.gateway.client import groq_client
            from app.config import settings
            fallback_resp = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = fallback_resp.choices[0].message.content
            plan_update = state["plan"] + ["Gateway: Direct Groq Fallback"]
            status = "Response generated (via direct Groq)."

        return {
            "final_answer": content,
            "status": status,
            "plan": plan_update,
            "messages": [{"role": "assistant", "content": content}]
        }
