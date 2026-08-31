import re
import json
import csv
import os

def sanitize_filename(title: str) -> str:
    """
    Pulisce le stringhe per evitare problemi con i filesystem.
    """
    return re.sub(r'[\\/*?:"<>|]', "", title)[:50].strip()

def export_results(data: list, base_filename: str):
    """
    Esporta la lista di dizionari in formati JSON e CSV.
    """
    if not data:
        print(f"No data to save for {base_filename}. Skipping export.")
        return

    json_file = f"{base_filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    csv_file = f"{base_filename}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
    print(f"\n[OK] Export completed successfully!")
    print(f" -> {json_file}")
    print(f" -> {csv_file}\n")

def process_and_print_result(model_name: str, raw_json_str: str) -> None:
    """
    Formatta e stampa l'output JSON dell'LLM.
    """
    try:
        parsed_json = json.loads(raw_json_str)
        print(json.dumps(parsed_json, indent=4))
    except json.JSONDecodeError:
        print("[!] Failed to decode JSON. Raw response:")
        print(raw_json_str)
