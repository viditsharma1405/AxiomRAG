from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    prompt = f"""
    You are an intelligent Assistant Planner for an Enterprise IT Assistant focused strictly on:
    - Kubernetes
    - Intel hardware
    - Enterprise networking
    
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the message is a greeting (e.g. hi, hello) or a question about previous conversation context (e.g. "what is my name"), respond with 'CONVERSATIONAL'.
    2. If the message is off-topic (e.g. movies, Netflix, entertainment, jokes, sports, food, personal preferences, non-IT chit-chat), respond with 'OFF_TOPIC'.
    3. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL', 'OFF_TOPIC', or the search query.
    """
    
    with logfire.span("🧠 Planner Decision"):
        raw_decision = llm.invoke(prompt).content.strip()
        # Clean reasoning tokens if any
        import re
        decision = re.sub(r"<think>.*?</think>", "", raw_decision, flags=re.DOTALL).strip()
        logfire.info(f"Intent identified: {decision}")
    
    if "OFF_TOPIC" in decision.upper():
        return {
            "current_query": "OFF_TOPIC",
            "status": "Identified as off-topic query. Bypassing retrieval...",
            "plan": ["Intent: Off-Topic (Enterprise Scope Enforced)", "Retrieval: Skipped"]
        }

    if "CONVERSATIONAL" in decision.upper():
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
