# Shared settings for the groundedness-judge scripts (scripts/nli). Sourced.
#
# The judges run on a SEPARATE venv from $WORK/venv_vllm and venv_vllm_qwen38,
# which stay untouched. It reuses the verified torch/vLLM/transformers pins of
# venv_vllm (CUDA 12.2 driver ceiling): torch 2.5.1+cu121, vllm 0.7.3 (only for
# Bespoke-MiniCheck-7B), transformers 4.49.0, numpy 1.26.4.

WORK_DIR="${WORK:?WORK is not set (it is on Leonardo)}"
NLI_VENV="${NLI_VENV:-$WORK_DIR/venv_nli}"
# Hugging Face cache holding every judge model (compute nodes are offline).
export NLI_HF_CACHE="${NLI_HF_CACHE:-$WORK_DIR/hf_nli}"
export HF_HOME="$NLI_HF_CACHE"
export HF_HUB_CACHE="$NLI_HF_CACHE"
# sentence tokenizer data used by MiniCheck, fetched on a login node.
export NLTK_DATA="${NLTK_DATA:-$WORK_DIR/nltk_data}"
export VLLM_CACHE_ROOT="$WORK_DIR/.cache/vllm"
export TRITON_CACHE_DIR="$WORK_DIR/.cache/triton"

# Hugging Face repos: judge models, and the base-model tokenizers MiniCheck loads.
NLI_MODEL_REPOS=(
    lytang/MiniCheck-Flan-T5-Large
    lytang/MiniCheck-DeBERTa-v3-Large
    lytang/MiniCheck-RoBERTa-Large
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
)
BESPOKE_REPO=bespokelabs/Bespoke-MiniCheck-7B
NLI_TOKENIZER_REPOS=(google/flan-t5-large roberta-large microsoft/deberta-v3-large)
