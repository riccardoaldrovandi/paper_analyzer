# 📚 Academic Paper Analyzer & Literature Review Assistant

A modular Python framework designed to streamline academic literature reviews for research and thesis writing. It automates paper discovery via APIs (Semantic Scholar & INSPIRE HEP), downloads open-access PDFs and LaTeX source archives, and leverages multi-LLM backends (Google Gemini, Groq, OpenRouter) to extract structured JSON summaries.

---

## 🚀 Features

- **API Integration:** Search and query literature via **Semantic Scholar** and **INSPIRE HEP**.
- **Automated Downloads:** Fetch open-access PDFs and raw LaTeX `.tar.gz` packages straight from arXiv.
- **Robust Text Extraction:** Parse text cleanly page-by-page using `PyMuPDF` or traverse raw `.tex` source files in memory.
- **Multi-LLM Structural Analysis:** Extract core contributions, methodology, datasets, and limitations into standardized JSON format using:
  - Google Gemini (Flash models)
  - Groq (Llama models)
  - OpenRouter (Free tier models)
- **Modular Architecture:** Clean separation of concerns (API clients, downloaders, extractors, LLM wrappers, and CLI).

---

## 📁 Project Structure

```text
paper_analyzer/
├── src/
│   ├── api/              # Semantic Scholar and INSPIRE HEP connectors
│   ├── download/         # PDF and LaTeX downloader logic
│   ├── processing/       # PDF and LaTeX text extraction engines
│   ├── llm/              # LLM clients (Gemini, Groq, OpenRouter)
│   ├── utils/            # Helpers for sanitization and export
│   └── config.py         # Global configuration and system prompts
├── main.py               # Unified CLI entry point
├── requirements.txt      # Project dependencies
└── .env                  # Environment variables template
```
--- 

## 🛠️ Installation & Setup

**Clone the repository:**
```
git clone [https://github.com/riccardoaldro/paper_analyzer.git](https://github.com/riccardoaldro/paper_analyzer.git)
cd paper_analyzer
```

**Install dependencies**

```
pip install -r requirements.txt
```

**Configure Environment Variables:**
Create a .env file in the root directory and add your API keys:
```
S2_API_KEY=your_semantic_scholar_api_key_optional
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

---

## ⚙️ Complete CLI Usage & Configuration
The framework is operated via ***main.py*** using two main subcommands: ***fetch*** and ***analyze***.

### 1. The fetch Subcommand (API Querying & Downloading)

Used to search academic databases, save metadata to JSON/CSV, and download local copies of PDFs and LaTeX source files.

- Basic Semantic Scholar search (default, 3 papers):
```
python main.py fetch
```
- Query a specific topic on Semantic Scholar with custom limit:
```
python main.py fetch --api semantic --query "Transformer models" --limit 5
```
- Query INSPIRE HEP (Physics focus):
```
python main.py fetch --api inspire --query "Quantum Chromodynamics" --limit 3
```
- Query both APIs sequentially:
```
python main.py fetch --api both --query "Reinforcement Learning" --limit 4
```
- Fetch papers AND download arXiv LaTeX source archives (.tar.gz):
```
python main.py fetch --api semantic --query "Multi-Agent Reinforcement Learning" --limit 3 --latex
```

### The analyze Subcommand (LLM Structured Extraction)
Used to extract text from a local PDF or LaTeX archive and analyze it using LLM backends to output a standardized JSON summary.

- Analyze a local PDF using Google Gemini (default):
```
python main.py analyze --pdf downloaded_pdfs/paper_id_title.pdf --model gemini
```
- Analyze a local PDF using Groq:
```
python main.py analyze --pdf downloaded_pdfs/paper_id_title.pdf --model groq
```
- Analyze a local PDF using OpenRouter:
```
python main.py analyze --pdf downloaded_pdfs/paper_id_title.pdf --model openrouter
```
- Analyze a LaTeX source archive (.tar.gz) across all providers simultaneously::
```
python main.py analyze --latex downloaded_latex/paper_id_source.tar.gz --model all
```

---

## 📜 License

Distributed under the MIT License. See **LICENSE** for more information.
