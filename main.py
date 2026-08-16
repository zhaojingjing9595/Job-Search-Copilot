"""
main.py

description: Phase 7 CLI entrypoint. Streams the LangGraph agent loop
(agent/graph.py) node by node via rich.Panel - each agent reasoning step,
each tool call, and each tool result - as it happens, ending with the
agent's final summary. This is the "readable trace scrolling by" milestone
from the build guide's Phase 7.

usage: python main.py "find backend jobs in Tel Aviv and evaluate matches"
       python main.py                          # uses the default goal
       python main.py --recursion-limit 10 "..."
"""
import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from agent.graph import stream
from core.logger import get_logger

load_dotenv()
console = Console()
logger = get_logger(__name__)

_DEFAULT_GOAL = "find backend jobs in Tel Aviv and evaluate matches"


def _render_agent_update(messages: list) -> None:
    """An 'agent' node update: one AIMessage, possibly carrying tool calls
    (request to act) and/or content (reasoning or the final answer)."""
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            console.print(Panel(
                f"[bold]{call['name']}[/bold]({call['args']})",
                title="agent -> tool call", border_style="cyan",
            ))
        if msg.content:
            console.print(Panel(str(msg.content), title="agent", border_style="green"))


def _render_tools_update(messages: list) -> None:
    """A 'tools' node update: one ToolMessage per tool call just executed."""
    for msg in messages:
        name = getattr(msg, "name", "tool")
        console.print(Panel(str(msg.content), title=f"tool result: {name}", border_style="yellow"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Job-search copilot: an LLM-driven agent that searches, "
                     "evaluates, drafts cover letters for, and logs job postings."
    )
    parser.add_argument("goal", nargs="*", help=f"what to ask the agent to do (default: {_DEFAULT_GOAL!r})")
    parser.add_argument("--recursion-limit", type=int, default=25,
                         help="max graph steps before LangGraph aborts the run (default: 25)")
    args = parser.parse_args()

    goal = " ".join(args.goal) or _DEFAULT_GOAL
    console.print(Panel(goal, title="goal", border_style="bold blue"))

    final_content = ""
    for node_name, update in stream(goal, recursion_limit=args.recursion_limit):
        messages = update.get("messages", [])
        if node_name == "agent":
            _render_agent_update(messages)
        elif node_name == "tools":
            _render_tools_update(messages)
        if messages and messages[-1].content:
            final_content = messages[-1].content

    console.print(Panel(final_content, title="final answer", border_style="bold green"))


if __name__ == "__main__":
    main()
