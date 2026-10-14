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
- **Smart Figure Extraction:** Scan a PDF or LaTeX archive and keep only the pages that actually contain a data plot or chart, using Gemini Vision to discard text-only pages and architecture diagrams, then crop multi-figure pages into individual, clean chart images.
- **Visuo-Mathematical Analysis:** Digitize an extracted chart into structured (X, Y) coordinates — separating overlapping curves into distinct series instead of merging them — and run symbolic regression (PySR) to recover the underlying mathematical formula behind a continuous plot. Every discovered formula is validated on a held-out slice of the extracted points, so equations that merely memorize noisy data get flagged instead of reported as fact.
- **Grounded Attribution:** AI-derived curve-fit formulas from chart analysis are kept in a dedicated `visual_analysis_findings` field of the output JSON, clearly separated from `results`/`methodology` — so the final summary never misattributes an algorithm's own approximation to the paper's authors.
- **End-to-End Pipeline:** The `full-analyze` command chains text extraction, figure extraction, and visuo-mathematical analysis into one call, feeding the LLM a paper enriched with everything the pipeline discovered about its own charts.
- **Fine-Tuning Dataset Export:** Batch-run the full pipeline over every downloaded paper and export (input, teacher-model output) pairs as a resumable JSONL dataset, ready to fine-tune a local open-weight LLM as a cheaper, offline replacement for the cloud API calls.
- **Resilient Vision Pipeline:** Automatic retry with backoff on transient Gemini errors (HTTP 503), and automatic fallback to a free OpenRouter model when the Gemini free-tier quota runs out (HTTP 429), so long extraction runs don't have to be restarted from scratch.
- **Modular Architecture:** Clean separation of concerns (API clients, downloaders, extractors, LLM wrappers, and CLI).

---

## 📁 Project Structure

```text
paper_analyzer/
├── src/
│   ├── api/              # Semantic Scholar and INSPIRE HEP connectors
│   ├── download/         # PDF and LaTeX downloader logic
│   ├── processing/       # PDF/LaTeX text extraction + figure extraction (image_extractor.py)
│   ├── llm/              # LLM clients (Gemini, Groq, OpenRouter, local vLLM) + JSON schemas
│   ├── advanced/         # Plot digitization (Gemini Vision) and symbolic regression (PySR)
│   ├── utils/            # Helpers for sanitization, export, and API retry/fallback logic
│   ├── evaluation/       # Offline dataset quality metrics (Level 1 structure, Level 2 grounding)
│   └── config.py         # Global configuration and system prompts
├── scripts/              # SLURM/vLLM scripts for CINECA Leonardo, probes and A/B tests
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
The framework is operated via ***main.py*** using six subcommands: ***fetch***, ***analyze***, ***images***, ***advanced***, ***full-analyze***, and ***export-dataset***.

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
- Fetch only papers with a minimum citation count, to filter out noise on a broad query:
```
python main.py fetch --api semantic --query "Quantum Chromodynamics" --limit 10 --min-citations 50
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

### The images Subcommand (Figure Extraction)
Scans a local PDF or LaTeX archive and pulls out only the pages/figures that actually contain a data plot or chart. Gemini Vision filters out pages that are just text, tables, or architecture diagrams, and for pages holding several figures side by side it automatically crops each chart into its own clean image. Extracted images are saved under `extracted_images/<paper_name>/`.

- Extract figures from a local PDF:
```
python main.py images --pdf downloaded_pdfs/paper_id_title.pdf
```
- Extract figures from a local LaTeX archive:
```
python main.py images --latex downloaded_latex/paper_id_source.tar.gz
```

### The advanced Subcommand (Visuo-Mathematical Analysis)
Takes a single chart image (ideally one already produced by the `images` subcommand) and runs it through the full visuo-mathematical pipeline. Gemini Vision first digitizes the chart into (X, Y) coordinates, classifying it as **continuous** (a curve or trend line) or **discrete** (a bar chart or scatter comparison), and — crucially — separates multiple overlapping curves on the same plot into distinct series instead of merging them into one confused point cloud. For each continuous curve, PySR then performs symbolic regression to recover the underlying formula. Every candidate formula is validated on a held-out slice of the extracted points it never saw during the search, so an equation that just memorizes noisy data instead of capturing the real trend gets flagged rather than reported as gospel.

- Discover the formula behind a continuous chart (default: 20 PySR iterations):
```
python main.py advanced --image extracted_images/paper_id_title/paper_id_title_page4_crop_fig5.png
```
- Give PySR a bigger search budget for a harder curve:
```
python main.py advanced --image extracted_images/paper_id_title/paper_id_title_page4_crop_fig5.png --iterations 60
```

### The full-analyze Subcommand (End-to-End Enriched Analysis)
Chains the entire pipeline into a single command: extracts the paper's text, extracts and analyzes every figure it can find (digitizing continuous curves into formulas or summarizing discrete charts), and finally sends the enriched content to an LLM for structured extraction. The result is the same JSON schema as `analyze`, but with an extra `visual_analysis_findings` field carrying whatever the visuo-mathematical pipeline discovered — kept clearly separate from the paper's own stated `results` (see Grounded Attribution above). A figure or curve that fails to analyze is skipped with a log message rather than aborting the whole run.

- Run the full pipeline on a local PDF with Gemini:
```
python main.py full-analyze --pdf downloaded_pdfs/paper_id_title.pdf --model gemini
```
- Run it on a LaTeX source archive with a bigger PySR search budget per curve:
```
python main.py full-analyze --latex downloaded_latex/paper_id_source.tar.gz --model gemini --iterations 60
```

### The export-dataset Subcommand (Fine-Tuning Dataset Generation)
Batch-runs the same pipeline as `full-analyze` over every PDF/LaTeX file already downloaded, and appends each (paper text, LLM JSON output) pair to a JSONL file in a standard chat-SFT format (`system`/`user`/`assistant` messages) — ready to fine-tune a local open-weight model (e.g. Qwen, Llama) as a distilled, offline replacement for the cloud LLM calls. The command is resumable: papers already present in the output file are skipped on the next run, so you can keep extending the dataset as you download more papers without re-spending API calls on the ones already processed. Any paper whose LLM output is empty or fails to parse as JSON is skipped rather than polluting the dataset.

- Export a dataset from everything already downloaded, using Gemini as the teacher model:
```
python main.py export-dataset --model gemini --output finetune_dataset.jsonl
```
- Process only the next 5 new papers (useful to stay under a free-tier daily quota):
```
python main.py export-dataset --model gemini --output finetune_dataset.jsonl --limit 5
```
- Point at custom PDF/LaTeX directories instead of the default `downloaded_pdfs/`/`downloaded_latex/`:
```
python main.py export-dataset --pdf-dir my_pdfs --latex-dir my_latex --output finetune_dataset.jsonl
```

---

`images`, `advanced`, `full-analyze`, and `export-dataset` all automatically retry on a transient Gemini overload (HTTP 503) and fall back to a free OpenRouter model if the Gemini free-tier quota is exhausted (HTTP 429), so a long batch of extractions doesn't need to be restarted by hand.

---

## 🖥️ Local Open-Weight Pipeline (branch `cineca-local-pipeline`)

This branch adds a **fully local backend** to the pipeline: the cloud LLM calls (Gemini / Groq / OpenRouter) can be replaced by open-weight models served with [vLLM](https://github.com/vllm-project/vllm) on GPUs. It was built and tested on the **CINECA Leonardo** supercomputer (A100 64 GB nodes) to generate the fine-tuning dataset without API rate limits or quotas.

### How it works

The local run uses **two models on two dedicated GPUs**:

| Role | Where it runs | Used for |
|---|---|---|
| **Text model** (e.g. Qwen2.5-14B-Instruct) | loaded in-process by `src/llm/local_client.py` (GPU 0) | structured JSON extraction from the paper text |
| **Vision model** (e.g. Qwen2.5-VL-7B-Instruct) | OpenAI-compatible `vllm serve` server (GPU 1), called over HTTP by `src/advanced/digitizer.py` | digitizing chart images into (X, Y) points |

The pipeline itself is unchanged: text extraction → figure digitization → PySR symbolic regression → structured extraction. Select it with `--model local` on `analyze`, `full-analyze` and `export-dataset`.

### What the local backend adds

- **Model families:** `scripts/run_dual_model.sbatch` takes a `FAMILY` that picks the text + vision pair:
  - `qwen`: Qwen2.5-14B + Qwen2.5-VL-7B
  - `llama`: Llama-3.1-8B + LLaVA-NeXT-Llama3-8B
  - `mistral`: Mistral-Nemo-12B + Pixtral-12B
  - `velvet`: Velvet-14B (Almawave, text only) + Qwen2.5-VL-7B
- **Guided JSON decoding:** the text model's output is constrained to the answer JSON schema (`src/llm/schemas.py`), so it always parses. If guided decoding fails, the call is retried unconstrained. The same option is available for the vision model, enabled per family.
- **Context-aware truncation:** papers longer than the model's context are trimmed token-aware (head 70% / tail 30% of the body). The ADVANCED PLOT ANALYSIS block is always kept whole, and the record notes when the input was truncated.
- **Deterministic `visual_analysis_findings`:** this field is rebuilt from the PySR results instead of relying on the model copying formulas correctly. The model's own version is kept as `llm_visual_analysis_findings`.
- **Robust answer handling:** JSON fence stripping, backslash repair, salvaging answers whose only invalid part is the findings field, and normalizing placeholder values ("None mentioned", empty strings) to `[]` / `"Not specified"`.
- **Data-quality gates before PySR:** synthetic-looking, identity or degenerate point sets are skipped, and curves with fewer than 4 points are reported as raw points instead of fitted.
- **Pre-cropped figures:** figure filtering and cropping still uses Gemini Vision. Offline, pass `--figures-dir <dir>` to use already-cropped figures from `<dir>/<paper name>/`. `comparison_figures/` holds the figures used for the model comparison, so every model receives the same images.
- **Resumable, reproducible runs:** per-paper checkpoints (`CHECKPOINT_DIR`, `KEEP_CHECKPOINTS`), per-paper error isolation (failures go to `<output>.rejected.jsonl` and the batch continues), a fixed sampling seed, and a `generation` metadata block in each record (prompt hash, code version, settings).

### Evaluating the generated dataset

The new `evaluate` subcommand scores an `export-dataset` JSONL offline. It needs no GPU and no API calls.

- **Level 1 (structural gate):** is the answer a well-formed, non-degenerate instance of the schema?
- **Level 2 (grounding score, 0-1):** are the numbers, references and datasets it cites actually in the paper, and are the PySR formulas kept out of the paper's own fields?

```
python main.py evaluate --input training_data_qwen.jsonl --report eval_qwen.jsonl
python main.py evaluate --input training_data_qwen.jsonl --filtered-output clean.jsonl --min-l2 0.8
```

### Running on CINECA Leonardo

Requirements: a Python 3.11 venv at `$WORK/venv_vllm` with `torch==2.5.1+cu121`, `vllm==0.7.3` and `transformers==4.49.0` (the node driver supports up to CUDA 12.2), plus the models downloaded under `$WORK/models/` with `hf download`. Compute nodes have no internet access, so install packages and download models from a login node.

The SLURM scripts don't hardcode a project account: set yours once on the login node (e.g. in `~/.bashrc`), and `sbatch` picks it up from the environment:

```bash
export SBATCH_ACCOUNT=<your_project_account>
```

```bash
# one family on the comparison set, using pre-cropped figures
sbatch --export=ALL,FAMILY=qwen,PDF_DIR=comparison_pdfs,LATEX_DIR=none,\
OUTPUT=training_data_qwen.jsonl,EXTRA_ARGS="--figures-dir comparison_figures" \
    scripts/run_dual_model.sbatch
```

Each job requests 2 GPUs, 16 CPUs and 240 GB of RAM. The script starts the vision server, waits until it is ready, runs `export-dataset --model local`, and always stops the server at the end. Logs are written to `logs/`.

**Main environment variables:**

| Variable | Default | Meaning |
|---|---|---|
| `SBATCH_ACCOUNT` | — | your SLURM project account (required by `sbatch`) |
| `MODELS_DIR` | `$WORK/models` | directory holding the downloaded models |
| `LOCAL_MODEL_PATH` | `$WORK/models/qwen2.5-14b` | text model directory |
| `LOCAL_VISION_MODEL_PATH` / `LOCAL_VISION_SERVER_URL` | Qwen2.5-VL-7B / `http://localhost:8001/v1` | vision model and its server |
| `LOCAL_TEMPERATURE`, `LOCAL_MAX_MODEL_LEN`, `LOCAL_SEED` | `0.2`, `32768`, `0` | text sampling settings |
| `LOCAL_GUIDED_JSON` / `LOCAL_VISION_GUIDED_JSON` | `1` / `0` | guided JSON decoding for text / vision |
| `CHECKPOINT_DIR`, `KEEP_CHECKPOINTS` | `full_analyze_checkpoints`, `0` | resume checkpoints |
| `SEED_CHECKPOINT_DIR` (sbatch only) | — | reuse another run's figure analysis, so only the text model changes |

### Other scripts

- `scripts/vllm_server.sh`: helpers to start, wait for and stop the vision server, with per-model `vllm serve` flags.
- `scripts/probe_vision_models.sbatch` + `vision_probe.py`: compare vision models on `scripts/probe_figures/` (`chat_templates/` holds a fixed Idefics3 template).
- `scripts/test_guided_json.*`: A/B test of free vs guided JSON decoding.
- `scripts/diagnose_text_model.*`: single-paper diagnostics for a text model.

### Known limitations

- Figure filtering and cropping (`images`) still requires Gemini. A fully local cropper is future work.
- No vision model tested returns the requested 15-30 points per curve; they typically read 4-9 points at the tick marks.
- The Llama vision models (LLaVA-NeXT, Idefics3) mostly fail to digitize charts, and Velvet-14B was not suitable as a text teacher in our tests.

---

## 📜 License

Distributed under the MIT License. See [**LICENSE**](https://github.com/riccardoaldrovandi/paper_analyzer/blob/main/LICENSE) for more information.
