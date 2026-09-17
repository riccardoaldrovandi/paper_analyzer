import argparse
import os
import json
import glob
from src.api.semantic_scholar import fetch_semantic_scholar
from src.api.inspire_hep import fetch_inspire_hep
from src.utils.helpers import sanitize_filename, export_results, process_and_print_result
from src.processing.pdf_extractor import extract_text_from_pdf
from src.processing.latex_extractor import extract_text_from_latex_tarball
from src.llm.gemini_client import analyze_with_gemini
from src.llm.groq_client import analyze_with_groq
from src.llm.openrouter_client import analyze_with_openrouter
from src.processing.image_extractor import extract_images_from_pdf, extract_images_from_latex
from src.advanced.digitizer import digitize_plot
from src.advanced.pysr import discover_formula
from src.config import SYSTEM_PROMPT, PDF_DIR, LATEX_DIR


def run_llm_providers(text_content: str, model_arg: str) -> None:
    """
    Runs the requested LLM provider(s) (or all of them) over the given text content
    and prints each resulting structured JSON. Shared between `analyze` and
    `full-analyze` so both commands report results the exact same way.
    """
    providers = ["gemini", "groq", "openrouter"] if model_arg.lower() == "all" else [model_arg.lower()]

    for provider in providers:
        print(f"\n--- Running analysis with: {provider.upper()} ---")
        if provider == "gemini":
            m_name, raw_res = analyze_with_gemini(text_content)
        elif provider == "groq":
            m_name, raw_res = analyze_with_groq(text_content)
        elif provider == "openrouter":
            m_name, raw_res = analyze_with_openrouter(text_content)
        else:
            continue

        print(f"\n------------ {provider.capitalize()} ------------")
        print(f"Model used: {m_name}")
        print("Output JSON:")
        process_and_print_result(m_name, raw_res)
        print("-" * 40)


def analyze_figure(image_path: str, iterations: int) -> str:
    """
    Runs the visuo-mathematical pipeline (digitize + symbolic regression) on a
    single extracted figure and returns a plain-text summary block describing it,
    or None if no usable data could be extracted from that image. Any failure on
    an individual figure or curve is logged and skipped rather than aborting the
    whole batch, since one bad chart shouldn't sink the analysis of the rest.
    """
    print(f"\n[~] Analyzing figure: {os.path.basename(image_path)}")
    try:
        plot_data = digitize_plot(image_path)
    except Exception as e:
        print(f"[!] Skipping {os.path.basename(image_path)}: unexpected error during digitization: {e}")
        return None

    if plot_data is None:
        print(f"[!] Skipping {os.path.basename(image_path)}: could not extract data from this image.")
        return None

    plot_type = plot_data.get("plot_type", "continuous")
    lines = [f"[Figure: {os.path.basename(image_path)}] ({plot_type} plot)"]

    if plot_type == "continuous":
        for series in plot_data.get("series", []):
            label = series.get("label", "curve")
            try:
                result = discover_formula(series["X"], series["y"], iterations=iterations)
            except Exception as e:
                print(f"[!] Skipping curve '{label}': unexpected error during symbolic regression: {e}")
                continue

            if result.get("r2_holdout") is not None:
                confidence = f", R^2 fit={result['r2']:.3f}, holdout={result['r2_holdout']:.3f}"
                if result["r2_holdout"] < 0.85 or (result["r2"] - result["r2_holdout"]) > 0.15:
                    confidence += " (low confidence, possible overfitting)"
            elif result.get("r2") is not None:
                confidence = f", R^2={result['r2']:.3f} (not cross-validated, too few points)"
            else:
                confidence = ""
            lines.append(f"  - Curve '{label}': y = {result['equation']}{confidence}")
    else:
        summary = plot_data.get("summary", "")
        if summary:
            lines.append(f"  Summary: {summary}")
        labels = plot_data.get("labels", [])
        X, y = plot_data.get("X"), plot_data.get("y")
        if X is not None and y is not None:
            for i, (px, py) in enumerate(zip(X, y)):
                label = labels[i] if i < len(labels) else f"Item {i + 1}"
                lines.append(f"  - {label}: ({px}, {py})")

    if len(lines) == 1:
        # Nothing usable was extracted beyond the header line.
        return None

    return "\n".join(lines)


def build_enriched_paper_text(pdf: str = None, latex: str = None, iterations: int = 20, max_figures: int = 0) -> str:
    """
    Extracts a paper's text and figures, runs the visuo-mathematical analysis on
    each figure, and returns the text enriched with the ADVANCED PLOT ANALYSIS
    block. This is the exact input both `full-analyze` and `export-dataset` feed
    to the LLM, kept in one place so the two commands can never drift apart.
    Returns None if the source file doesn't exist.

    max_figures caps how many extracted figures actually get analyzed (0 = no
    cap): some papers (e.g. LaTeX sources with dozens of repetitive ablation
    plots) can dump 50+ images, which would otherwise burn through a whole day's
    API quota analyzing a single paper.
    """
    if pdf:
        if not os.path.exists(pdf):
            print(f"[!] PDF not found: {pdf}")
            return None
        print(f"\n[1/3] Extracting text from PDF: {pdf}")
        text_content = extract_text_from_pdf(pdf)
        print(f"\n[2/3] Extracting figures...")
        image_paths = extract_images_from_pdf(pdf)
    elif latex:
        if not os.path.exists(latex):
            print(f"[!] LaTeX tarball not found: {latex}")
            return None
        print(f"\n[1/3] Extracting text from LaTeX tarball: {latex}")
        text_content = extract_text_from_latex_tarball(latex)
        print(f"\n[2/3] Extracting figures...")
        image_paths = extract_images_from_latex(latex)
    else:
        return None

    print(f"[+] Found {len(image_paths)} figure(s) to analyze.")
    if max_figures > 0 and len(image_paths) > max_figures:
        print(f"[!] Capping analysis to the first {max_figures} figure(s) out of {len(image_paths)} (--max-figures).")
        image_paths = image_paths[:max_figures]

    print(f"\n[3/3] Running visuo-mathematical analysis on each figure...")
    advanced_blocks = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n--- Figure {i}/{len(image_paths)} ---")
        block = analyze_figure(image_path, iterations)
        if block:
            advanced_blocks.append(block)

    if advanced_blocks:
        disclaimer = (
            "(The formulas below were derived by an independent AI curve-fitting/symbolic-regression "
            "tool applied to the pixel data of each figure. They are NOT part of the original paper and "
            "were NOT derived or stated by its authors -- report them separately, never as the paper's own results.)"
        )
        text_content += "\n\n--- ADVANCED PLOT ANALYSIS ---\n" + disclaimer + "\n\n" + "\n\n".join(advanced_blocks)
        print(f"\n[+] Enriched paper content with analysis of {len(advanced_blocks)}/{len(image_paths)} figure(s).")
    else:
        print(f"\n[!] No usable figure analysis produced; proceeding with text-only content.")

    return text_content


def already_exported_sources(output_path: str) -> set:
    """
    Reads an existing dataset JSONL file (if any) and returns the set of source
    files already exported, so re-running `export-dataset` only processes papers
    that aren't in the dataset yet instead of re-spending API calls on them.
    """
    exported = set()
    if not os.path.exists(output_path):
        return exported

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                exported.add(record.get("source_file"))
            except json.JSONDecodeError:
                continue
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📚 Academic Paper Analyzer Framework & Literature Review Assistant",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Subcommand: fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch papers from academic APIs and download PDF/LaTeX.")
    fetch_parser.add_argument("--api", type=str, choices=["semantic", "inspire", "both"], default="semantic")
    fetch_parser.add_argument("--query", type=str, default="Multi-Agent Reinforcement Learning")
    fetch_parser.add_argument("--limit", type=int, default=3)
    fetch_parser.add_argument("--latex", action="store_true")
    fetch_parser.add_argument("--min-citations", type=int, default=0, help="Minimum number of citations")
    fetch_parser.add_argument("--sort", type=str, default=None, choices=["mostrecent", "mostcited"], help="INSPIRE-only: sort results by date or by citation count instead of relevance")

    # Subcommand: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a local PDF or LaTeX tarball using LLMs.")
    analyze_parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    analyze_parser.add_argument("--latex", type=str, help="Path to local LaTeX .tar.gz file")
    analyze_parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "groq", "openrouter", "all"])

    # Subcommand: images
    images_parser = subparsers.add_parser("images", help="Extract images from a local PDF or LaTeX tarball.")
    images_parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    images_parser.add_argument("--latex", type=str, help="Path to local LaTeX .tar.gz file")

    # Subcommand: advanced
    advanced_parser = subparsers.add_parser("advanced", help="Esegue l'estrazione dati (Vision) e la regressione simbolica (PySR) su un grafico.")
    advanced_parser.add_argument("--image", type=str, required=True, help="Percorso dell'immagine del grafico estratto")
    advanced_parser.add_argument("--iterations", type=int, default=20, help="Numero di iterazioni per PySR")

    # Subcommand: full-analyze
    full_analyze_parser = subparsers.add_parser(
        "full-analyze",
        help="Full pipeline: extract text + figures, discover formulas/describe each chart, then run LLM structured analysis enriched with the visual findings."
    )
    full_analyze_parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    full_analyze_parser.add_argument("--latex", type=str, help="Path to local LaTeX .tar.gz file")
    full_analyze_parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "groq", "openrouter", "all"])
    full_analyze_parser.add_argument("--iterations", type=int, default=20, help="Number of PySR iterations per continuous curve")
    full_analyze_parser.add_argument("--max-figures", type=int, default=0, help="Max number of figures to analyze for this paper (0 = no limit)")

    # Subcommand: export-dataset
    export_dataset_parser = subparsers.add_parser(
        "export-dataset",
        help="Batch-run full-analyze over downloaded papers and export (input, teacher JSON) pairs as a JSONL fine-tuning dataset."
    )
    export_dataset_parser.add_argument("--pdf-dir", type=str, default=PDF_DIR, help="Directory of PDFs to process")
    export_dataset_parser.add_argument("--latex-dir", type=str, default=LATEX_DIR, help="Directory of LaTeX .tar.gz archives to process")
    export_dataset_parser.add_argument("--output", type=str, default="finetune_dataset.jsonl", help="Output JSONL file (appended to, resumable)")
    export_dataset_parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "groq", "openrouter", "all"], help="Teacher model(s) used to generate the training labels")
    export_dataset_parser.add_argument("--iterations", type=int, default=20, help="Number of PySR iterations per continuous curve")
    export_dataset_parser.add_argument("--limit", type=int, default=0, help="Max number of new papers to process this run (0 = no limit)")
    export_dataset_parser.add_argument("--max-figures", type=int, default=0, help="Max number of figures to analyze per paper (0 = no limit)")

    args = parser.parse_args()

    if args.command == "fetch":
        safe_query_name = sanitize_filename(args.query).replace(" ", "_")
        if args.api in ["semantic", "both"]:
            dataset_s2 = fetch_semantic_scholar(
                query=args.query, 
                limit=args.limit, 
                fetch_latex=args.latex, 
                min_citations=args.min_citations
            )
            export_results(dataset_s2, base_filename=f"{safe_query_name}_semantic")
        if args.api in ["inspire", "both"]:
            dataset_inspire = fetch_inspire_hep(
                query=args.query,
                limit=args.limit,
                fetch_latex=args.latex,
                min_citations=args.min_citations,
                sort_by=args.sort
            )
            export_results(dataset_inspire, base_filename=f"{safe_query_name}_inspire")

    elif args.command == "analyze":
        if not args.pdf and not args.latex:
            print("[!] Please specify either --pdf or --latex file to analyze.")
            exit()

        text_content = ""
        if args.pdf:
            if os.path.exists(args.pdf):
                print(f"\n[~] Extracting text from PDF: {args.pdf}")
                text_content = extract_text_from_pdf(args.pdf)
            else:
                print(f"[!] PDF not found: {args.pdf}")
                exit()
        elif args.latex:
            if os.path.exists(args.latex):
                print(f"\n[~] Extracting text from LaTeX tarball: {args.latex}")
                text_content = extract_text_from_latex_tarball(args.latex)
            else:
                print(f"[!] LaTeX tarball not found: {args.latex}")
                exit()

        run_llm_providers(text_content, args.model)
    elif args.command == "images":
        if not args.pdf and not args.latex:
            print("[!] Specify either --pdf or --latex to extract images.")
            exit()

        extracted = []
        if args.pdf:
            extracted = extract_images_from_pdf(args.pdf)
        elif args.latex:
            extracted = extract_images_from_latex(args.latex)
            
        print(f"\n[+] Operation completed. Found {len(extracted)} images:")
        for img in extracted:
            print(f" -> {img}")
    elif args.command == "advanced":
        if not os.path.exists(args.image):
            print(f"[!] Immagine non trovata: {args.image}")
            exit()
            
        print(f"\n--- AVVIO PIPELINE VISUO-MATEMATICA ---")
        print(f"[1] Estrazione dati strutturati con Gemini Vision...")
        plot_data = digitize_plot(args.image)
        
        # Controlliamo che plot_data non sia None prima di procedere
        if plot_data is not None:
            plot_type = plot_data.get("plot_type", "continuous")

            print("\n======================================")
            print(" RISULTATO FINALE ANALISI AVANZATA")
            print("======================================")
            print(f" Immagine   : {os.path.basename(args.image)}")
            print(f" Tipo Plot  : {plot_type.upper()}")

            if plot_type == "continuous":
                series_list = plot_data.get("series", [])
                print(f"\n[2] Grafico continuo rilevato: {len(series_list)} curva/e distinta/e. Scoperta delle formule con PySR...")
                for series in series_list:
                    print(f"\n--- Curva: {series['label']} ({len(series['X'])} punti) ---")
                    result = discover_formula(series["X"], series["y"], iterations=args.iterations)
                    print(f" Equazione       : {result['equation']}")
                    if result.get("r2_holdout") is not None:
                        gap = result["r2"] - result["r2_holdout"]
                        overfit_warn = " (possibile overfitting: fidati poco di questa formula)" if (result["r2_holdout"] < 0.85 or gap > 0.15) else ""
                        print(f" R^2 (fit)       : {result['r2']:.4f}")
                        print(f" R^2 (holdout)   : {result['r2_holdout']:.4f}{overfit_warn}")
                    elif result.get("r2") is not None:
                        print(f" R^2             : {result['r2']:.4f} (non validato su holdout: pochi punti)")

            else:
                # Caso Discreto
                X = plot_data.get("X")
                y = plot_data.get("y")
                print(f"\n[2] Grafico discreto rilevato: PySR disabilitato.")
                labels = plot_data.get("labels", [])
                summary = plot_data.get("summary", "Nessun riassunto disponibile.")

                print(f"\n Descrizione: {summary}")
                print("\n Mapping dei dati estratti (X, Y):")
                for i, (px, py) in enumerate(zip(X, y)):
                    label = labels[i] if i < len(labels) else f"Elemento {i+1}"
                    print(f"  -> {label}: ({px}, {py})")

            print("======================================")
        else:
            print("\n[-] Pipeline interrotta: impossibile estrarre i dati dall'immagine (errore API o timeout).")
    elif args.command == "full-analyze":
        if not args.pdf and not args.latex:
            print("[!] Please specify either --pdf or --latex file to analyze.")
            exit()

        text_content = build_enriched_paper_text(pdf=args.pdf, latex=args.latex, iterations=args.iterations, max_figures=args.max_figures)
        if text_content is None:
            exit()

        run_llm_providers(text_content, args.model)
    elif args.command == "export-dataset":
        pdf_files = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
        latex_files = sorted(glob.glob(os.path.join(args.latex_dir, "*.tar.gz")))
        all_sources = [(p, "pdf") for p in pdf_files] + [(p, "latex") for p in latex_files]

        already_done = already_exported_sources(args.output)
        pending = [(p, kind) for p, kind in all_sources if p not in already_done]
        print(f"[+] {len(all_sources)} paper(s) found, {len(already_done)} already in '{args.output}', {len(pending)} pending.")

        if args.limit > 0:
            pending = pending[:args.limit]
            print(f"[+] Limiting this run to {len(pending)} paper(s) (--limit {args.limit}).")

        providers = ["gemini", "groq", "openrouter"] if args.model.lower() == "all" else [args.model.lower()]
        written = 0

        with open(args.output, "a", encoding="utf-8") as out_f:
            for i, (source_path, kind) in enumerate(pending, 1):
                print(f"\n========== Paper {i}/{len(pending)}: {os.path.basename(source_path)} ==========")
                if kind == "pdf":
                    text_content = build_enriched_paper_text(pdf=source_path, iterations=args.iterations, max_figures=args.max_figures)
                else:
                    text_content = build_enriched_paper_text(latex=source_path, iterations=args.iterations, max_figures=args.max_figures)

                if not text_content:
                    print(f"[!] Skipping {source_path}: no text could be extracted.")
                    continue

                for provider in providers:
                    print(f"\n[~] Generating training label with {provider.upper()}...")
                    if provider == "gemini":
                        m_name, raw_res = analyze_with_gemini(text_content)
                    elif provider == "groq":
                        m_name, raw_res = analyze_with_groq(text_content)
                    elif provider == "openrouter":
                        m_name, raw_res = analyze_with_openrouter(text_content)
                    else:
                        continue

                    try:
                        parsed = json.loads(raw_res)
                    except json.JSONDecodeError:
                        parsed = None

                    if not parsed:
                        print(f"[!] Skipping {provider}: empty or invalid JSON output.")
                        continue

                    record = {
                        "source_file": source_path,
                        "provider": m_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text_content},
                            {"role": "assistant", "content": raw_res},
                        ],
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    written += 1
                    print(f"[+] Wrote training example ({provider}, {len(text_content)} chars input).")

        print(f"\n[+] Done. {written} training example(s) appended to '{args.output}'.")
    else:
        parser.print_help()
