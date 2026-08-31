import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
S2_API_KEY = os.environ.get("S2_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

# Directories
PDF_DIR = "downloaded_pdfs"
LATEX_DIR = "downloaded_latex"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(LATEX_DIR, exist_ok=True)

# Common System Prompt for LLM Paper Analysis
SYSTEM_PROMPT = """
You are an expert AI research assistant. Your task is to analyze the provided text extracted from a scientific paper (which may be raw PDF text or LaTeX source code) and extract specific structured information.

You MUST return the output strictly as a valid JSON object with the following keys, and nothing else. All values must be strings or arrays of strings in English.

JSON Schema:
{
  "problem_addressed": "Brief description of the main problem the paper tries to solve",
  "main_contributions": ["Contribution 1", "Contribution 2"],
  "methodology": "Summary of the proposed method or architecture",
  "datasets_used": ["Dataset 1", "Dataset 2"],
  "experiments": "Summary of how the models were evaluated",
  "results": "Key quantitative and qualitative results",
  "limitations": "Any limits acknowledged by the authors",
  "future_work": "Future directions mentioned in the paper",
  "relevant_elements": "List of crucial formulas, algorithms, or tables mentioned (e.g., 'Algorithm 1: PPO Update', 'Table 3: Baseline Comparison')"
}

If any information is completely missing from the text, use "Not specified" or an empty list []. Do not include markdown code block syntax (like ```json) in your response if possible, just return the raw JSON object string.
"""
