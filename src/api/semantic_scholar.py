import time
import requests
from src.config import S2_API_KEY
from src.download.downloader import download_pdf, download_latex

def fetch_semantic_scholar(query: str, limit: int = 5, fetch_latex: bool = False) -> list:
    """
    Interroga le API di Semantic Scholar, estrae i metadati e gestisce il download
    di PDF e sorgenti LaTeX correlati.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,authors,abstract,externalIds,year,s2FieldsOfStudy,citationCount,references,openAccessPdf"
    
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"query": query, "limit": limit, "fields": fields}
    
    print(f"\n======================================")
    print(f" [SEMANTIC SCHOLAR] Searching for: '{query}'")
    print(f"======================================\n")
    
    session = requests.Session()
    time.sleep(1.5)  # Pausa iniziale per cortesia verso il server
    
    max_retries = 5
    response = None
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                wait_time = 3 ** attempt
                print(f"[!] Rate limit (429) hit. Pausing for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"[!] Semantic Scholar API Error: {response.status_code} - {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"[!] Network connection error: {e}")
            time.sleep(2)
    else:
        print("[!] Max retries reached. Semantic Scholar is currently blocking requests.")
        return []

    raw_data = response.json().get("data", [])
    processed_papers = []

    for item in raw_data:
        paper_id = item.get("paperId", "unknown")
        title = item.get("title", "Unknown Title")
        
        # Estrazione ID esterni (qui si trova l'ArXiv ID)
        external_ids = item.get("externalIds", {})
        doi = external_ids.get("DOI", "N/A")
        arxiv_id = external_ids.get("ArXiv", "N/A")
        
        authors = ", ".join([a["name"] for a in item.get("authors", [])])
        year = item.get("year", "N/A")
        abstract = item.get("abstract", "N/A")
        citations = item.get("citationCount", 0)
        categories = ", ".join([c["category"] for c in item.get("s2FieldsOfStudy", []) if "category" in c])
        
        pdf_url = item.get("openAccessPdf", {}).get("url", "N/A") if item.get("openAccessPdf") else "N/A"
        
        # 1. Download del PDF
        local_pdf_path = download_pdf(pdf_url, paper_id, title)
        
        # 2. Download del LaTeX (se richiesto e se l'ArXiv ID è disponibile)
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
