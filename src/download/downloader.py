import os
import requests
from src.config import PDF_DIR, LATEX_DIR
from src.utils.helpers import sanitize_filename

def download_pdf(pdf_url: str, paper_id: str, title: str) -> str:
    if not pdf_url or pdf_url == "N/A":
        return "N/A"

    safe_title = sanitize_filename(title)
    filename = f"{paper_id[:8]}_{safe_title}.pdf"
    filepath = os.path.join(PDF_DIR, filename)

    if os.path.exists(filepath):
        print(f"[-] PDF already exists locally, skipping download: {filename}")
        return filepath

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        print(f"[~] Attempting to download PDF: {pdf_url}")
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=15)
        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[+] Success! Saved PDF to: {filename}")
            return filepath
        else:
            print(f"[!] Hit a snag: The link didn't point directly to a PDF (Status: {response.status_code})")
            return "N/A"
    except Exception as e:
        print(f"[!] Connection dropped or failed during PDF download: {e}")
        return "N/A"

def download_latex(arxiv_id: str, paper_id: str, title: str) -> str:
    if not arxiv_id or arxiv_id == "N/A":
        return "N/A"

    safe_title = sanitize_filename(title)
    filename = f"{paper_id[:8]}_{safe_title}_source.tar.gz"
    filepath = os.path.join(LATEX_DIR, filename)

    if os.path.exists(filepath):
        print(f"[-] LaTeX source already exists locally, skipping download: {filename}")
        return filepath

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    url = f"[https://arxiv.org/e-print/](https://arxiv.org/e-print/){arxiv_id}"

    try:
        print(f"[~] Attempting to download LaTeX source: {url}")
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[+] Success! Saved LaTeX source to: {filename}")
            return filepath
        else:
            print(f"[!] Failed to get LaTeX source (Status: {response.status_code})")
            return "N/A"
    except Exception as e:
        print(f"[!] Connection dropped during LaTeX download: {e}")
        return "N/A"
