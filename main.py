import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from core.logger import get_logger

load_dotenv()
console=Console()
logger = get_logger(__name__)
gemini_api_key = os.environ["GEMINI_API_KEY"]
llm_model = "gemini-3.5-flash"

logger.info("Initializing Gemini LLM (%s)", llm_model)
llm = GoogleGenerativeAI(model=llm_model, google_api_key=gemini_api_key)
# response = llm.invoke(
#     "What are some of the pros and cons of Python as a programming language?"
# )
# console.print(Panel(Markdown(response), title="Gemini", border_style="cyan"))

