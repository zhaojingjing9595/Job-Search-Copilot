"""
graph.py

description: Phase 5 - the LangGraph agent loop. Replaces a manual "call LLM
-> check for tool call -> execute -> loop" script with an explicit graph: an
agent node (calls Gemini with the running message history), a tools node
(executes whatever tool Gemini asked for), and a conditional edge that keeps
looping until Gemini responds without a tool call.

No hardcoded call order - Gemini decides which of agent.tools' four tools to
call, when, and how many times, up to `recursion_limit` (a Phase 6 guardrail,
applied here since LangGraph requires it at invoke time regardless).

usage: python -m agent.graph "find backend jobs in Tel Aviv and evaluate matches"
"""
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console

from agent.tools import AGENT_TOOLS
from core.constants import GEMINI_MODEL
from core.logger import get_logger

load_dotenv()
console = Console()
logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a job-search copilot. You have tools to search live postings, \
read the candidate's profile, draft grounded customized CV profiles, and log evaluated postings \
to a tracking sheet.

For each posting you consider, reason about fit against the candidate's actual profile \
(skills, experience, preferences) before deciding anything. Only draft a customized CV profile \
for postings worth applying to - this saves a submission-ready file and gives you back its path. \
When you log such a posting, pass that path as cv_profile_path so the tracking sheet points \
straight at the artifact ready to submit. Only log postings you have actually evaluated, and \
always give a real match_reasoning that explains your judgment - never a restatement of a score. \
When you've searched, evaluated, tailored, and logged what's worth logging, summarize what you \
found and stop."""

_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.environ["GEMINI_API_KEY"])
_llm_with_tools = _llm.bind_tools(AGENT_TOOLS)


def agent_node(state: MessagesState) -> dict:
    """Call Gemini with the running message history; it decides whether to
    answer directly or request a tool call."""
    response = _llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    """Assemble the agent loop: agent -> (tools -> agent)* -> end.

    Returns:
        a compiled LangGraph app, ready for `.invoke()` / `.stream()`.
    """
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(AGENT_TOOLS))
    graph.set_entry_point("agent")
    # tools_condition reads the last message: routes to "tools" if it has a
    # tool call, otherwise to END - this is the loop's only branch point.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run(goal: str, recursion_limit: int = 25) -> str:
    """Invoke the agent loop on a goal and return its final answer.

    Args:
        goal: e.g. "find backend jobs in Tel Aviv and evaluate matches".
        recursion_limit: max graph steps before LangGraph aborts the run.

    Returns:
        str: the agent's final (non-tool-call) message content.
    """
    app = build_graph()
    messages = [SystemMessage(content=_SYSTEM_PROMPT), ("user", goal)]
    logger.info("Running agent for goal: %r", goal)
    final_state = await app.ainvoke({"messages": messages}, config={"recursion_limit": recursion_limit})
    return final_state["messages"][-1].content


async def stream(goal: str, recursion_limit: int = 25):
    """Invoke the agent loop on a goal, yielding one (node_name, update) pair
    per graph step as it happens - Phase 7 tracing, for a caller (main.py) to
    render node transitions live instead of blocking until the final answer.

    Uses astream/ToolNode's async execution path since AGENT_TOOLS includes
    async tools (search_jobs) that a sync app.stream() can't invoke.

    Args:
        goal: e.g. "find backend jobs in Tel Aviv and evaluate matches".
        recursion_limit: max graph steps before LangGraph aborts the run.

    Yields:
        tuple[str, dict]: node_name ("agent" or "tools") and that node's
        MessagesState update ({"messages": [...]}) for the step just completed.
    """
    app = build_graph()
    messages = [SystemMessage(content=_SYSTEM_PROMPT), ("user", goal)]
    logger.info("Streaming agent for goal: %r", goal)
    async for update in app.astream(
        {"messages": messages}, config={"recursion_limit": recursion_limit}, stream_mode="updates"
    ):
        for node_name, node_update in update.items():
            yield node_name, node_update


if __name__ == "__main__":
    import asyncio

    goal = " ".join(sys.argv[1:]) or "find backend jobs in Tel Aviv and evaluate matches"
    result = asyncio.run(run(goal))
    console.print(f"[bold green]Final answer:[/bold green]\n{result}")
