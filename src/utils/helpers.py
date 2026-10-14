import re
import json
import csv
import os

def sanitize_filename(title: str) -> str:
    """
    Cleans up strings to avoid filesystem problems.
    """
    return re.sub(r'[\\/*?:"<>|]', "", title)[:50].strip()

def export_results(data: list, base_filename: str):
    """
    Exports the list of dictionaries in JSON and CSV formats.
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

def strip_json_fences(text: str) -> str:
    """
    Free-tier chat models (used as an OpenRouter fallback for Gemini Vision calls)
    sometimes ignore the "no markdown" instruction and wrap their JSON in ```
    fences anyway.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()

def extract_json_object(text: str) -> str:
    """
    Some local models (e.g. Mistral-Nemo, 2026-10-08 run) put a sentence such
    as "Here's the extracted information:" before the ```json block, which
    strip_json_fences can't handle. If the text isn't valid JSON as-is, keep
    only the span from the first "{" to the last "}". Applied the same way to
    every model, so the comparison measures content, not chattiness.
    """
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return text
    return text[start:end + 1]


# The findings key as the schema defines it, plus the variants weaker models
# write (e.g. Mistral-Nemo's "visual_analysis Findings", 2026-10-09 run).
_FINDINGS_KEY_RE = re.compile(r',\s*"visual[ _]?analysis[ _]?findings"\s*:', re.IGNORECASE)


def salvage_without_findings(text: str, required_keys: list) -> tuple:
    """
    Recovers an answer whose ONLY broken part is "visual_analysis_findings":
    weaker local models often write that field as invalid JSON (junk keys,
    "..." placeholders, // comments) while the rest of the answer is fine, and
    the field gets rebuilt deterministically anyway. Cuts the text right
    before the findings key and closes the object. Returns (parsed dict,
    removed findings text), or (None, None) if the rest still doesn't parse or
    misses any of required_keys -- a record with missing fields is not kept.
    """
    match = _FINDINGS_KEY_RE.search(text or "")
    if not match:
        return None, None
    try:
        parsed = json.loads(repair_json_backslashes(text[:match.start()] + "\n}"))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(parsed, dict) or any(k not in parsed for k in required_keys):
        return None, None
    return parsed, text[match.end():].strip()


def process_and_print_result(model_name: str, raw_json_str: str) -> None:
    """
    Formats and prints the LLM's JSON output.
    """
    try:
        parsed_json = json.loads(raw_json_str)
        print(json.dumps(parsed_json, indent=4))
    except json.JSONDecodeError:
        print("[!] Failed to decode JSON. Raw response:")
        print(raw_json_str)

# Matches a backslash that already forms a JSON escape safe to keep (\" \\ \/ \uXXXX),
# or any other lone backslash (group 1 empty) that needs doubling.
_JSON_BACKSLASH_RE = re.compile(r'\\(["\\/]|u[0-9a-fA-F]{4})|\\')

def repair_json_backslashes(text: str) -> str:
    """
    Local models often write LaTeX (e.g. \\(\\tilde{\\varphi}\\)) inside JSON strings
    without escaping the backslashes, which makes json.loads fail on "\\(".
    Only kicks in when the text doesn't already parse: it doubles every backslash
    except \\" \\\\ \\/ \\uXXXX, so LaTeX like \\tilde or \\frac is preserved
    instead of being read as a tab / form feed.
    """
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return _JSON_BACKSLASH_RE.sub(lambda m: m.group(0) if m.group(1) else r'\\', text)
