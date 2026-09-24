from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.config import LLM_MODEL, GROQ_API_KEY
from app.agents.tools import TOOLS

SYSTEM_PROMPT = (
    "You are an HR Policy Assistant. You answer employee questions about "
    "company HR policy (leave, WFH, maternity/paternity, equipment, etc.) "
    "using the search_hr_policy tool to ground every factual claim in the "
    "actual policy documents — never invent policy details. Use the "
    "calculate_leave_balance tool when the user asks about their remaining "
    "leave. If a question is outside HR policy scope, say so politely."
)

_checkpointer = MemorySaver()

_agent = create_react_agent(
    model=ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY),
    tools=TOOLS,
    messages_modifier=SYSTEM_PROMPT,
    checkpointer=_checkpointer,
)


def run_agent(user_message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    final_message = result["messages"][-1]
    return final_message.content