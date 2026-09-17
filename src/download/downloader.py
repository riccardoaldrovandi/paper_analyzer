import os
import requests
import urllib3
from src.config import PDF_DIR, LATEX_DIR
from src.utils.helpers import sanitize_filename

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        print(f"[~] Attempting to download PDF: {pdf_url}")
        # Aggiunto verify=False per bypassare server accademici con certificati scaduti
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=15, verify=False)
        
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
    url = f"https://arxiv.org/e-print/{arxiv_id}"

    try:
        print(f"[~] Attempting to download LaTeX source: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            # arXiv's /e-print/ endpoint serves a plain PDF instead of a gzip
            # tarball when the author submitted no separate LaTeX source (common
            # for older or non-standard submissions). Saving that under a
            # .tar.gz name would report a false success -- it looks downloaded
            # but tarfile.open() will silently fail on it later, and there'd be
            # no way to tell "genuinely no LaTeX source" from "we broke it".
            content_type = response.headers.get("Content-Type", "")
            is_gzip = "gzip" in content_type or response.content[:2] == b"\x1f\x8b"
            if not is_gzip:
                print(f"[!] No real LaTeX source for {arxiv_id}: arXiv served a '{content_type or 'non-gzip'}' file (likely a PDF-only submission).")
                return "N/A"

            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"[+] Success! Saved LaTeX source to: {filename}")
            return filepath
        else:
            print(f"[!] Failed to get LaTeX source (Status: {response.status_code})")
            return "N/A"
    except Exception as e:
        print(f"[!] Connection dropped during LaTeX download: {e}")
        return "N/A"
