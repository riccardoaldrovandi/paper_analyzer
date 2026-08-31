import os
import pymupdf

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        print(f"[!] File not found: {pdf_path}")
        return ""

    doc = pymupdf.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        full_text.append(text)

    doc.close()
    return "\n".join(full_text)
