"""Writes a patched copy of vLLM 0.19.1's MLA attention module for A100 + FP8.

The bug (mistral4_server_62431278.log, 2026-10-27): when a prefill attends to
tokens already in the KV cache (a prefix-cache hit, or any prompt longer than
--max-num-batched-tokens, which is prefilled in chunks), _compute_prefill_context
casts the cached latent to "the dtype of kv_b_proj's weight". On a GPU without
FP8 (A100), FP8 weights are repacked for the Marlin kernel as int32, so the
input is cast to int32 and marlin_gemm fails with "unsupported `a` scalar_type",
killing the engine. The fix keeps the model dtype whenever that weight dtype is
not a floating type: Marlin wants bf16/fp16 input and dequantizes internally.

Run inside the container; prints "<patched copy>:<original>", the Apptainer bind
that puts the copy over the original (scripts/mistral4/common.sh does this).
    python3 patch_vllm_mla.py <output_dir>
"""
import importlib.util
import os
import sys

# Located without importing vLLM (that would load torch and CUDA for nothing).
TARGET = os.path.join(importlib.util.find_spec("vllm").submodule_search_locations[0],
                      "model_executor", "layers", "attention", "mla_attention.py")

OLD = "            if use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype():\n"
NEW = ("            if use_fp8_prefill or (_kv_b_proj_w_dtype.is_floating_point\n"
       "                                   and _kv_b_proj_w_dtype != current_platform.fp8_dtype()):\n")

src = open(TARGET, encoding="utf-8").read()
if src.count(OLD) != 1:
    sys.exit(f"[!] {TARGET}: expected the line to patch exactly once, found {src.count(OLD)} "
             "(different vLLM version?). Not patched.")
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "mla_attention.py")
with open(out, "w", encoding="utf-8") as f:
    f.write(src.replace(OLD, NEW))
print(f"{out}:{TARGET}")
