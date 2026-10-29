"""Manual review of the fine-tuning answers (every selected answer is reviewed by hand, decided 2026-10-28).

    streamlit run scripts/finetune/review_app.py            (reads/writes ft_review/queue.jsonl)

Left: the paper (rendered PDF pages). Right: the teacher's answer, one field at
a time, editable; above it what the automatic checks flagged (numbers,
references and datasets not found in the paper, cross-seed agreement).
Each paper ends as:
  accepted  the answer is correct as it is
  edited    accepted after your corrections (the corrected JSON is stored)
  rejected  not usable (with a note on why)
The edited answer must still pass the v2.1 Level 1 checks (valid JSON, all
keys, right types) before it can be saved. Every save rewrites the queue file,
so the review can be stopped and resumed at any point.
build_training_set.py turns the accepted/edited answers into the training file.
"""
import copy
import json
import os
import sys
from datetime import datetime

import pymupdf
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.evaluation.metrics import level1  # noqa: E402
from src.prompt_v2 import FIELDS_V2, FIELDS_V21, record_version  # noqa: E402

QUEUE = os.environ.get("FT_QUEUE", os.path.join("ft_review", "queue.jsonl"))
STATUS_ICON = {"pending": "⏳", "accepted": "✅", "edited": "✏️", "rejected": "❌"}


def load_queue() -> list:
    with open(QUEUE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_queue(items: list) -> None:
    tmp = QUEUE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, QUEUE)


@st.cache_data(show_spinner=False)
def page_png(pdf_path: str, page: int, zoom: float) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        return doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


@st.cache_data(show_spinner=False)
def page_count(pdf_path: str) -> int:
    with pymupdf.open(pdf_path) as doc:
        return len(doc)


def field_editor(key: str, spec, value, wkey: str):
    """One answer field as a widget; returns the edited value (or raises ValueError on bad JSON)."""
    if spec == "string":
        return st.text_area(key, value or "", key=wkey, height=110)
    if spec == "list":
        text = st.text_area(f"{key}  (one item per line)", "\n".join(value or []), key=wkey, height=110)
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(spec, dict):
        st.markdown(f"**{key}**")
        return {sub: field_editor(f"{key}.{sub}", sub_spec, (value or {}).get(sub), f"{wkey}.{sub}") for sub, sub_spec in spec.items()}
    # A list of objects: edited as JSON (keeps nested lists such as figures[].curves).
    text = st.text_area(f"{key}  (JSON list)", json.dumps(value or [], ensure_ascii=False, indent=1), key=wkey,
                        height=min(80 + 18 * len(json.dumps(value or [], indent=1).splitlines()), 520))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{key}: invalid JSON ({e})")
    if not isinstance(parsed, list):
        raise ValueError(f"{key}: must be a JSON list")
    return parsed


def main():
    st.set_page_config(page_title="Fine-tuning review", layout="wide")
    if not os.path.exists(QUEUE):
        st.error(f"{QUEUE} not found: run scripts/finetune/select_answers.py first.")
        return
    items = load_queue()
    counts = {s: sum(1 for i in items if i["status"] == s) for s in STATUS_ICON}

    with st.sidebar:
        st.header("Review queue")
        st.write(" · ".join(f"{STATUS_ICON[s]} {n}" for s, n in counts.items()))
        st.progress((len(items) - counts["pending"]) / max(len(items), 1))
        show = st.multiselect("Show", list(STATUS_ICON), default=list(STATUS_ICON))
        visible = [i for i, it in enumerate(items) if it["status"] in show]
        if not visible:
            st.info("Nothing to show.")
            return
        labels = {i: f"{STATUS_ICON[items[i]['status']]} {(items[i].get('title') or items[i]['paper'])[:60]}" for i in visible}
        # "goto" is set after a save (a widget's own state can't be changed once it is drawn).
        if "goto" in st.session_state:
            st.session_state.current = st.session_state.pop("goto")
        if st.session_state.get("current") not in visible:
            st.session_state.current = next((i for i in visible if items[i]["status"] == "pending"), visible[0])
        current = st.radio("Paper", visible, format_func=labels.get, key="current")

    item = items[current]
    answer = item["reviewed_answer"] or item["answer"]
    # The prompt the answer was generated with (v2.1 for the fine-tuning set; v2 in tests on older runs).
    version = record_version({"messages": [{"role": "system", "content": item["system"]}]})
    fields = FIELDS_V21 if version == "v2.1" else FIELDS_V2
    st.subheader(item.get("title") or item["paper"])
    st.caption(f"{item.get('category') or ''} · arXiv {item.get('arxiv_id') or '?'} · seed {item.get('seed')} · "
               f"{item['valid']}/{item['candidates']} valid answers · status {STATUS_ICON[item['status']]} {item['status']}")

    m = item["metrics"]
    l2 = m["level2"]
    flags = []
    for part in ("numbers", "references", "datasets"):
        if l2.get(part, {}).get("ungrounded"):
            flags.append(f"**{part} not found in the paper:** {', '.join(map(str, l2[part]['ungrounded']))}")
    st.info(f"L2 (no visual) {m['l2_no_visual']} · numbers agreeing across seeds {m['seed_number_agreement']} · "
            f"empty fields: {', '.join(m['empty_fields']) or 'none'}" + ("\n\n" + "\n\n".join(flags) if flags else ""))

    left, right = st.columns([1, 1])
    with left:
        pdf = item.get("local_pdf")
        if pdf and os.path.exists(pdf):
            n = page_count(pdf)
            page = st.number_input(f"Page (1-{n})", 1, n, 1, key=f"page{current}") - 1
            zoom = st.slider("Zoom", 1.0, 3.0, 1.6, 0.2, key="zoom")
            st.image(page_png(pdf, page, zoom), width="stretch")
        else:
            st.warning(f"PDF not found locally: {pdf}")
        with st.expander("Text the model received"):
            st.text(item["user"][:200000])

    with right:
        edited, errors = {}, []
        for key, spec in fields.items():
            if key == "visual_analysis_findings":
                with st.expander(f"visual_analysis_findings (filled by the code, {len(answer.get(key) or [])} entries)"):
                    st.json(answer.get(key) or [])
                edited[key] = answer.get(key) or []
                continue
            try:
                edited[key] = field_editor(key, spec, copy.deepcopy(answer.get(key)), f"f{current}.{key}")
            except ValueError as e:
                errors.append(str(e))
        note = st.text_input("Note (required to reject)", item.get("note", ""), key=f"note{current}")

        c1, c2, c3 = st.columns(3)
        action = None
        if c1.button("✅ Accept as is", width="stretch"):
            action = "accepted"
        if c2.button("✏️ Save corrections", width="stretch"):
            action = "edited"
        if c3.button("❌ Reject", width="stretch"):
            action = "rejected"

        if action:
            if action == "rejected" and not note.strip():
                st.error("Write in the note why the answer is rejected.")
            elif action == "edited" and errors:
                st.error("Fix these first: " + "; ".join(errors))
            else:
                if action == "edited":
                    check = level1(json.dumps(edited, ensure_ascii=False), version)
                    if not check["passed"]:
                        st.error(f"The corrected answer fails the {version} checks: {check['reasons'] or check}")
                        return
                    if edited == item["answer"]:
                        action = "accepted"
                item["status"] = action
                item["reviewed_answer"] = edited if action == "edited" else None
                item["note"] = note.strip()
                item["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
                save_queue(items)
                nxt = next((i for i in visible if items[i]["status"] == "pending" and i != current), None)
                if nxt is not None:
                    st.session_state.goto = nxt
                st.rerun()


if __name__ == "__main__":
    main()
