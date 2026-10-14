import time
import requests
from src.download.downloader import download_pdf, download_latex

def fetch_inspire_hep(query: str, limit: int = 5, fetch_latex: bool = False, min_citations: int = 0, sort_by: str = None) -> list:
    """
    Queries the INSPIRE HEP API (specialized in High Energy Physics and related fields),
    extracts the metadata, applies filters (e.g. minimum citations) and downloads PDF/LaTeX via arXiv.
    sort_by: None (relevance, INSPIRE's default), "mostrecent", or "mostcited".
    """
    url = "https://inspirehep.net/api/literature"
    params = {"q": query, "size": limit}
    if sort_by:
        params["sort"] = sort_by
    
    print(f"\n======================================")
    print(f" [INSPIRE HEP] Searching for: '{query}'")
    print(f"======================================\n")
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"[!] INSPIRE API Error: {response.status_code}")
        return []

    raw_data = response.json().get("hits", {}).get("hits", [])
    processed_papers = []

    for item in raw_data:
        metadata = item.get("metadata", {})
        paper_id = item.get("id", "unknown")
        
        titles = metadata.get("titles", [{}])
        title = titles[0].get("title", "Unknown Title") if titles else "Unknown Title"
        
        dois = metadata.get("dois", [{}])
        doi = dois[0].get("value", "N/A") if dois else "N/A"
        
        authors_list = metadata.get("authors", [])
        authors = ", ".join([a.get("full_name", "") for a in authors_list])
        
        pub_info = metadata.get("publication_info", [{}])
        year = pub_info[0].get("year", "N/A") if pub_info else "N/A"
        
        abstracts = metadata.get("abstracts", [{}])
        abstract = abstracts[0].get("value", "N/A") if abstracts else "N/A"
        
        citations = metadata.get("citation_count", 0)
        
        # --- CITATION FILTER ---
        # Skip straight to the next paper if it doesn't have enough citations
        if citations < min_citations:
            print(f"[-] Discarded: '{title}' (Citations: {citations} < {min_citations})")
            continue
            
        categories = ", ".join([c.get("term", "") for c in metadata.get("inspire_categories", [])])
        
        # Extract the ArXiv ID to build the download links
        arxiv_eprints = metadata.get("arxiv_eprints", [{}])
        arxiv_id = arxiv_eprints[0].get("value", "N/A") if arxiv_eprints else "N/A"
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id != "N/A" else "N/A"
        
        # 1. Download the PDF
        local_pdf_path = download_pdf(pdf_url, paper_id, title)
        
        # 2. Download the LaTeX source
        local_latex_path = "N/A"
        if fetch_latex and arxiv_id != "N/A":
            time.sleep(2)
            local_latex_path = download_latex(arxiv_id, paper_id, title)
        
        if pdf_url != "N/A" or local_latex_path != "N/A":
            time.sleep(2)
            
        processed_papers.append({
            "Paper_ID": paper_id,
            "Title": title,
            "Authors": authors,
            "Year": year,
            "DOI": doi,
            "ArXiv_ID": arxiv_id,
            "Categories": categories,
            "Citations": citations,
            "PDF_URL": pdf_url,
            "Local_PDF_Path": local_pdf_path,
            "Local_LaTeX_Path": local_latex_path,
            "Abstract": abstract
        })
        
    return processed_papers