"""
Diagnosis for a text model whose answers stop after a few tokens (Velvet-14B,
smoke test job 60294735).

Job 60300834 (A-D variants) found: the repetition/frequency penalties make
Velvet stop at once (they also count the ~9k prompt tokens, so </s> -- never
in the prompt -- wins); vLLM's llm.chat doubles the BOS but that changes
nothing; without penalties Velvet ignores the task and just summarizes the
ADVANCED PLOT ANALYSIS block, the last thing it read.

Job 60302490 (E-G variants, all penalties off): Velvet copies the schema's
field descriptions instead of filling them in, then loops until max_tokens
(e.g. "AI curve-fit approximation of AI curve-fit approximation of ...");
the task reminder after the paper (F) gave the only real content.

This run loads the model once and generates the answer for one paper, with
repetition_penalty off but frequency_penalty kept, in 2 variants:
  H  task reminder after the paper, free
  I  task reminder after the paper, guided JSON
plus a tiny sanity prompt. Prints the prompt token ids vLLM actually used, the
output length, the start/end of each answer and whether it is valid JSON.

Run from the repo root (see scripts/diagnose_text_model.sbatch).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="export-dataset JSONL whose first record's user message is reused")
    args = parser.parse_args()

    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from src.config import SYSTEM_PROMPT
    from src.llm.schemas import ANSWER_SCHEMA
    import src.llm.local_client as lc

    record = json.loads(open(args.inputs, encoding="utf-8").readline())
    paper = f"{lc.USER_PREFIX}{record['messages'][1]['content']}"
    # Restates the task AFTER the paper: with the instructions only at the start
    # of a ~9k-token prompt, Velvet answered by summarizing the last block it read.
    reminder = (
        "\n\n--- END OF PAPER CONTENT ---\n"
        "Now return ONLY the JSON object described in the instructions above, with exactly these keys: "
        + ", ".join(ANSWER_SCHEMA["required"])
        + ". Formulas from the ADVANCED PLOT ANALYSIS section go only in visual_analysis_findings."
    )
    variants = {
        "H: reminder, free": ([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": paper + reminder}], False),
        "I: reminder, guided": ([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": paper + reminder}], True),
    }
    sanity = [{"role": "user", "content": 'Return only this JSON object, filled in: {"capital_of_italy": "..."}'}]

    llm = lc._get_llm()
    tokenizer = llm.get_tokenizer()
    settings = lc.GENERATION_SETTINGS
    # repetition_penalty off (it also counts the prompt tokens -> Velvet stops at once,
    # job 60300834) but frequency_penalty kept: it only counts generated tokens, and
    # without it Velvet loops until max_tokens (job 60302490).
    base = dict(temperature=settings["temperature"], max_tokens=lc.MAX_NEW_TOKENS, seed=settings["seed"],
                frequency_penalty=settings["frequency_penalty"])
    free = SamplingParams(**base)
    guided = SamplingParams(**base, guided_decoding=GuidedDecodingParams(json=ANSWER_SCHEMA))

    print(f"[~] Model {lc.MODEL_NAME} | paper: {record['source_file']}")
    print(f"[~] bos_token_id={tokenizer.bos_token_id} eos_token_id={tokenizer.eos_token_id}")

    def show(label, out):
        ids = out.prompt_token_ids
        o = out.outputs[0]
        print(f"\n===== {label} =====")
        print(f"prompt: {len(ids)} tokens, first 4 ids {ids[:4]} -> double BOS: {ids[:2] == [tokenizer.bos_token_id] * 2}")
        print(f"output: {len(o.token_ids)} tokens, finish={o.finish_reason}, first ids {list(o.token_ids[:8])}")
        print(f"text start: {o.text[:400]!r}")
        print(f"text end:   {o.text[-300:]!r}")
        try:
            parsed = json.loads(o.text)
            print(f"VALID JSON, keys: {sorted(parsed) if isinstance(parsed, dict) else type(parsed).__name__}")
        except json.JSONDecodeError as e:
            print(f"NOT valid JSON ({e})")

    show("SANITY, penalties off", llm.chat(sanity, free)[0])
    for label, (messages, use_guided) in variants.items():
        show(label, llm.chat(messages, guided if use_guided else free)[0])


if __name__ == "__main__":
    main()
