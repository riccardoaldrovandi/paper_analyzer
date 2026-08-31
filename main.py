import argparse
from src.api.semantic_scholar import fetch_semantic_scholar
from src.api.inspire_hep import fetch_inspire_hep
from src.utils.helpers import sanitize_filename, export_results
from src.processing.pdf_extractor import extract_text_from_pdf
from src.config import PDF_DIR
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "👋 Welcome to the Academic Paper Analyzer Framework!\n\n"
            "Modulized architecture to fetch, download, and analyze literature papers."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--api", type=str, choices=["semantic", "inspire", "both"], default="semantic",
                        help="Database to query ('semantic', 'inspire', 'both')")
    parser.add_argument("--query", type=str, default="Multi-Agent Reinforcement Learning",
                        help="Search query string")
    parser.add_argument("--limit", type=int, default=3,
                        help="Number of papers to fetch per API")
    parser.add_argument("--latex", action="store_true",
                        help="Attempt downloading LaTeX source files (.tar.gz) from arXiv")
    parser.add_argument("--extract-sample", action="store_true",
                        help="Test text extraction on the first available downloaded PDF")
                        
    args = parser.parse_args()

    safe_query_name = sanitize_filename(args.query).replace(" ", "_")

    if args.api in ["semantic", "both"]:
        dataset_s2 = fetch_semantic_scholar(query=args.query, limit=args.limit, fetch_latex=args.latex) 
        export_results(dataset_s2, base_filename=f"{safe_query_name}_semantic")
        
    if args.api in ["inspire", "both"]:
        dataset_inspire = fetch_inspire_hep(query=args.query, limit=args.limit, fetch_latex=args.latex)
        export_results(dataset_inspire, base_filename=f"{safe_query_name}_inspire")

    if args.extract_sample:
        print("\n--- Running Sample PDF Extraction ---")
        if os.path.exists(PDF_DIR):
            files = os.listdir(PDF_DIR)
            pdf_files = [f for f in files if f.endswith(".pdf")]
            if pdf_files:
                sample_pdf = os.path.join(PDF_DIR, pdf_files[0])
                print(f"[~] Extracting text from: {sample_pdf}\n")
                extracted_text = extract_text_from_pdf(sample_pdf)
                print(f"[+] Total characters extracted: {len(extracted_text)}")
                print("\n--- Preview (First 500 characters) ---\n")
                print(extracted_text[:500])
            else:
                print("[!] No PDF files found in downloaded_pdfs/")
