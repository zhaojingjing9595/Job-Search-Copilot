"""
profile.py

description: one-time setup script that builds profile.yaml from user cv file(pdf, word, txt) plus a few clarifying questions. Run once during the project setup; edit the resulting YAML by hand afterward if anything changes.

usage: python -m tools.profile
"""
import json
import os
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from constants import GEMINI_MODEL
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
load_dotenv()
gemini_api_key = os.environ["GEMINI_API_KEY"]
console = Console()

def load_text(path: str) -> str:
    """load user resume file, return file content

    Args:
        path (str): user resume file path

    Raises:
        ValueError: unsupported file type
        RuntimeError: _description_

    Returns:
        str: file content
    """
    try:
        if path.endswith(".pdf"):
            reader = PdfReader(path)
            return "\n".join(page.extract_text() for page in reader.pages)
        elif path.endswith(".docx"):
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        elif path.endswith(".txt"):
            return Path(path).read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {path}")
    except (ValueError, Exception) as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Failed to load {path}: {e}") from e

def extract_profile(text: str) -> dict:
    """
    """
    
    llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=gemini_api_key)
    schema_example = """{
        "title": "string",
        "summary": "string",
        "professional_experiences": [
            {"company": "string", "role": "string", "years": "string", "highlights": ["string"]}
        ],
        "education": [
            {"school": "string", "degree": "string", "year": "string"}
        ],
        "skills": {
            "languages": ["string"],
            "frontend": ["string"],
            "backend": ["string"],...
        },
        "other": {
            "languages": ["string"],
            "hobby": ["string"]
        }
    }"""
    
    raw = llm.invoke(
        f"according to given resume content : {text}, summary information about title, summary, skills, profession experiences, education, languages, and extract them into a JSON object, the json schema should follow this example: {schema_example}, Return only valid JSON, no extra commentary."
    )
    return json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())

def asking_clarifying_questions() -> dict:
    console.print("[bold cyan]Please answer the following questions[/bold cyan], so I can understand better what positions you are looking for.")
    target_roles = Prompt.ask("[bold]Your target roles[/bold] (comma separated)")
    must_haves = Prompt.ask("[bold]Must-haves for a role[/bold] (comma separated)")
    deal_breakers = Prompt.ask("[bold]Deal-breakers[/bold] (comma separated)")
    preferred_locations = Prompt.ask("[bold]Preferred locations[/bold] (comma separated, or 'remote')")
    salary_floor = Prompt.ask("[bold]Minimum acceptable salary[/bold] (leave blank if not relevant)", default="")

    return {
        "target_roles": [r.strip() for r in target_roles.split(",") if r.strip()],
        "must_haves": [m.strip() for m in must_haves.split(",") if m.strip()],
        "deal_breakers": [d.strip() for d in deal_breakers.split(",") if d.strip()],
        "preferred_locations": [l.strip() for l in preferred_locations.split(",") if l.strip()],
        "salary_floor": salary_floor.strip() or None,
    }

def main():
    # 1. load cv file , return cv content
    cv_path = Path(__file__).resolve().parent.parent / "my_cv.pdf"
    try:
        text = load_text(str(cv_path))
        console.print("[green]CV loaded successfully.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error loading text from {cv_path}:[/bold red] {e}")
        return

    # 2. use llm to extract details into structured json
    profile = extract_profile(text)

    # 3. ask user clarifying questions and merge into profile
    preferences = asking_clarifying_questions()
    profile["preferences"] = preferences

    # 4. save merged profile to json file
    profile_path = Path(__file__).resolve().parent.parent / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved profile to {profile_path}[/bold green]")

if __name__ == "__main__":
    main()