import argparse
import os
from src.api.semantic_scholar import fetch_semantic_scholar
from src.api.inspire_hep import fetch_inspire_hep
from src.utils.helpers import sanitize_filename, export_results, process_and_print_result
from src.processing.pdf_extractor import extract_text_from_pdf
from src.processing.latex_extractor import extract_text_from_latex_tarball
from src.llm.gemini_client import analyze_with_gemini
from src.llm.groq_client import analyze_with_groq
from src.llm.openrouter_client import analyze_with_openrouter

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

    # Subcommand: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a local PDF or LaTeX tarball using LLMs.")
    analyze_parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    analyze_parser.add_argument("--latex", type=str, help="Path to local LaTeX .tar.gz file")
    analyze_parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "groq", "openrouter", "all"])

    args = parser.parse_args()

    if args.command == "fetch":
        safe_query_name = sanitize_filename(args.query).replace(" ", "_")
        if args.api in ["semantic", "both"]:
            dataset_s2 = fetch_semantic_scholar(query=args.query, limit=args.limit, fetch_latex=args.latex)
            export_results(dataset_s2, base_filename=f"{safe_query_name}_semantic")
        if args.api in ["inspire", "both"]:
            dataset_inspire = fetch_inspire_hep(query=args.query, limit=args.limit, fetch_latex=args.latex)
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

        providers = ["gemini", "groq", "openrouter"] if args.model.lower() == "all" else [args.model.lower()]

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
    else:
        parser.print_help()
