"""
Groundedness judges: does the paper support a claim?

Two kinds, both local (no API):

- MiniCheckJudge: MiniCheck models (Tang et al., EMNLP 2024), trained to check a
  claim against a long grounding document. The package chunks the document
  itself. Binary output: SUPPORTED / NOT_SUPPORTED.
- NLIJudge: a plain 3-class NLI model (entailment / neutral / contradiction),
  SciFact-style: it only reads ~512 tokens, so BM25 first retrieves the paper
  sentences closest to the claim and the model judges the claim against each
  retrieved passage. Output: SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO.

    SUPPORTED     if max_p(entail) >= t and max_p(entail) >= max_p(contra)
    CONTRADICTED  if max_p(contra) >= t and max_p(contra) >  max_p(entail)
    NOT_ENOUGH_INFO otherwise                    (max over the retrieved passages)

The threshold t is fixed (0.5), not tuned on the gold claims it is evaluated on.

Heavy imports (torch, transformers, minicheck) are inside the classes, so the
retrieval helpers can be used and tested without a GPU.
"""
import json
import math
import os
import re
import unicodedata
from collections import Counter

from src.config import ADVANCED_HEADER


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()


def load_papers(records_path: str) -> dict:
    """{basename of source_file: the paper text the model saw} from an export-dataset JSONL."""
    papers = {}
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            user = next(m["content"] for m in record["messages"] if m["role"] == "user")
            idx = user.find(ADVANCED_HEADER)
            papers[os.path.basename(record["source_file"])] = normalize_text(user if idx == -1 else user[:idx])
    return papers


# ---------------------------------------------------------------- retrieval --

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\\$])")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+")
MIN_SENTENCE_CHARS = 40


def split_sentences(text: str) -> list:
    """Regex sentence split; fragments shorter than MIN_SENTENCE_CHARS are glued to the next one."""
    out, carry = [], ""
    for piece in _SENT_SPLIT.split(text):
        piece = (carry + " " + piece).strip() if carry else piece.strip()
        if len(piece) < MIN_SENTENCE_CHARS:
            carry = piece
            continue
        out.append(piece)
        carry = ""
    if carry:
        out.append(carry)
    return out


def tokenize(text: str) -> list:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in documents]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list:
        terms = set(tokenize(query))
        out = []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out


def retrieve_passages(sentences: list, bm25: BM25, claim: str, top_k: int = 5, window: int = 1) -> list:
    """The top_k best-matching sentences, each with `window` neighbours on both sides (overlaps merged)."""
    scores = bm25.scores(claim)
    best = sorted(range(len(sentences)), key=lambda i: -scores[i])[:top_k]
    passages, seen = [], set()
    for i in best:
        lo, hi = max(0, i - window), min(len(sentences), i + window + 1)
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        passages.append(" ".join(sentences[lo:hi]))
    return passages


# ------------------------------------------------------------------- judges --

class MiniCheckJudge:
    """model_name: 'flan-t5-large', 'deberta-v3-large', 'roberta-large' or 'Bespoke-MiniCheck-7B'."""

    def __init__(self, model_name: str, cache_dir: str = None):
        from minicheck.minicheck import MiniCheck  # noqa: PLC0415
        # MiniCheck loads with device_map="auto"; transformers 4.49 refuses it for
        # DeBERTa-v2 ("does not support device_map='auto'") because the classes
        # declare no _no_split_modules. One layer class is the unit to keep whole.
        try:
            from transformers.models.deberta_v2 import modeling_deberta_v2  # noqa: PLC0415
            cls = modeling_deberta_v2.DebertaV2PreTrainedModel  # base class of every DeBERTa-v2 head
            if not getattr(cls, "_no_split_modules", None):
                cls._no_split_modules = ["DebertaV2Layer"]
        except ImportError:
            pass
        self.scorer = MiniCheck(model_name=model_name, cache_dir=cache_dir)

    def judge(self, docs: list, claims: list, batch_size: int = 8) -> list:
        out = []
        for i in range(0, len(claims), batch_size):
            labels, probs, _, _ = self.scorer.score(docs=docs[i:i + batch_size], claims=claims[i:i + batch_size])
            for label, prob in zip(labels, probs):
                out.append({"label": "SUPPORTED" if int(label) == 1 else "NOT_SUPPORTED", "scores": {"supported": float(prob)}})
        return out


class NLIJudge:
    def __init__(self, model_id: str, cache_dir: str = None, top_k: int = 5, window: int = 1, threshold: float = 0.5):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
        self.torch = torch
        self.top_k, self.window, self.threshold = top_k, window, threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir).to(self.device).eval()
        names = {int(i): n.lower() for i, n in self.model.config.id2label.items()}
        try:
            self.idx = {k: next(i for i, n in names.items() if n.startswith(k)) for k in ("entail", "neutral", "contra")}
        except StopIteration:
            raise ValueError(f"{model_id}: id2label {names} has no entailment/neutral/contradiction") from None
        self._index = {}  # id(doc) -> (sentences, BM25)

    def _probs(self, premises: list, claim: str) -> list:
        enc = self.tokenizer(premises, [claim] * len(premises), truncation="only_first", max_length=512,
                             padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model(**enc).logits.softmax(-1).cpu().tolist()

    def judge(self, docs: list, claims: list) -> list:
        out = []
        for doc, claim in zip(docs, claims):
            if id(doc) not in self._index:
                sentences = split_sentences(doc)
                self._index[id(doc)] = (sentences, BM25(sentences))
            sentences, bm25 = self._index[id(doc)]
            passages = retrieve_passages(sentences, bm25, claim, self.top_k, self.window)
            probs = self._probs(passages, claim)
            ent = max(p[self.idx["entail"]] for p in probs)
            con = max(p[self.idx["contra"]] for p in probs)
            if ent >= self.threshold and ent >= con:
                label = "SUPPORTED"
            elif con >= self.threshold and con > ent:
                label = "CONTRADICTED"
            else:
                label = "NOT_ENOUGH_INFO"
            out.append({"label": label, "scores": {"entailment": ent, "contradiction": con}})
        return out


# Name -> factory. cache_dir is the Hugging Face cache the models were downloaded to.
JUDGES = {
    "minicheck-flan-t5": lambda cache: MiniCheckJudge("flan-t5-large", cache),
    "minicheck-deberta": lambda cache: MiniCheckJudge("deberta-v3-large", cache),
    "minicheck-roberta": lambda cache: MiniCheckJudge("roberta-large", cache),
    "bespoke-minicheck-7b": lambda cache: MiniCheckJudge("Bespoke-MiniCheck-7B", cache),
    "deberta-mnli": lambda cache: NLIJudge("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", cache),
}
