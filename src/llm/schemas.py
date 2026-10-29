"""
JSON schemas for vLLM guided decoding, shared by the pipeline and by
scripts/test_guided_json.py so production and test always use the same ones.

Only features supported by xgrammar (vLLM 0.7.3's default guided-decoding
backend) are used: "enum", "pattern", numeric ranges or minItems/maxItems make
vLLM 0.7.3 fall back to outlines (see has_xgrammar_unsupported_json_features
in vllm/model_executor/guided_decoding/utils.py), which failed with HTTP 500
on Pixtral's Mistral tokenizer (jobs 59806387 and 59809276, 2026-10-09).
"""
from src.config import PROMPT_VERSION
from src.evaluation.metrics import STRING_FIELDS, LIST_FIELDS, FLEXIBLE_FIELDS
from src.prompt_v2 import ANSWER_SCHEMA_V2, ANSWER_SCHEMA_V21

# The teacher answer, built from the same field lists `evaluate` checks.
_STRING_OR_LIST = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}
ANSWER_SCHEMA_V1 = {
    "type": "object",
    "properties": {
        **{k: {"type": "string"} for k in STRING_FIELDS},
        **{k: {"type": "array", "items": {"type": "string"}} for k in LIST_FIELDS},
        **{k: _STRING_OR_LIST for k in FLEXIBLE_FIELDS},
    },
    "required": STRING_FIELDS + LIST_FIELDS + FLEXIBLE_FIELDS,
    "additionalProperties": False,
}
# The schema matching the prompt the pipeline sends (PROMPT_VERSION).
ANSWER_SCHEMA = {"v1": ANSWER_SCHEMA_V1, "v2": ANSWER_SCHEMA_V2, "v2.1": ANSWER_SCHEMA_V21}[PROMPT_VERSION]

# The VISION_PROMPT output format. Points are plain arrays of numbers: the
# "exactly two coordinates" rule is enforced afterwards by _valid_points().
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        # A plain string, not an enum (see above): _parse_vision_output() treats
        # anything other than "continuous" as a discrete plot anyway.
        "plot_type": {"type": "string"},
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                },
                "required": ["label", "points"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["plot_type", "series", "summary"],
}
