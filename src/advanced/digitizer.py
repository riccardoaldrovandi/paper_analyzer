import os
import json
import base64
import requests
import numpy as np
from google import genai
from google.genai import types
from src.utils.retry import call_gemini_with_retry, QuotaExceededError, GeminiOverloadedError
from src.utils.helpers import strip_json_fences
from src.llm.openrouter_client import analyze_image_with_openrouter

# Local vision model (Qwen2.5-VL-7B by default; LLaVA-NeXT, Idefics3 or Pixtral
# for the model comparison) runs as its own vLLM OpenAI-compatible server on a
# dedicated GPU, isolated from the text model (see scripts/run_dual_model.sbatch
# for the 2-GPU split) -- digitize_plot_local() below just talks to it over HTTP.
VISION_SERVER_URL = os.environ.get("LOCAL_VISION_SERVER_URL", "http://localhost:8001/v1")
VISION_MODEL_PATH = os.environ.get("LOCAL_VISION_MODEL_PATH", "/leonardo_work/IscrC_MultLLM/models/qwen2.5-vl-7b")
# Human-readable label for logs and dataset records, derived from the model
# directory so it can never go stale when a different vision model is served.
VISION_MODEL_NAME = os.path.basename(VISION_MODEL_PATH.rstrip("/"))
# Constrain the vision answer to VISION_SCHEMA (vLLM guided decoding). OFF by
# default: in run v7 (2026-10-09) it gained one figure with Qwen2.5-VL, but
# with Pixtral (Mistral-native tokenizer) xgrammar produced garbage and every
# figure was lost. scripts/run_dual_model.sbatch turns it on per family. If a
# guided request fails or its output can't be parsed, the figure is retried
# unconstrained.
VISION_GUIDED_JSON = os.environ.get("LOCAL_VISION_GUIDED_JSON", "0") == "1"

# Shared by both digitize_plot() (cloud) and digitize_plot_local() so the two
# paths always ask for, and get parsed as, the exact same output schema.
VISION_PROMPT = """
You are an expert data extraction algorithm. Analyze this plot image carefully.

Step 1: Determine if the plot is "continuous" (one or more line/curve trends showing
a mathematical relationship) or "discrete" (a bar chart or scatter plot comparing
independent categories/models with no continuous trend).

Step 2 (continuous plots only): This plot may contain MULTIPLE distinct curves
(e.g. differently colored or dashed lines identified by a legend). You MUST keep
them separate. For EACH distinct curve, extract 15 to 30 (X, Y) numerical float
coordinates sampled along that single curve, and give it a "label" taken from the
legend (or a short visual description like "upper line" if there is no legend).
Never mix points belonging to different curves into the same series.

Step 3 (discrete plots only): extract each individual data point (X, Y) and its
corresponding text label read directly from the plot.

You MUST return ONLY a valid JSON object. Do not include markdown blocks.
Format exactly like this:
{
    "plot_type": "continuous" or "discrete",
    "series": [
        {"label": "curve or point label", "points": [[x1, y1], [x2, y2]]}
    ],
    "summary": "For discrete plots only: a brief 1-sentence analytical summary of what the points show (e.g. 'Model X has higher accuracy but is slower...'). Leave empty for continuous plots."
}

For continuous plots, create one "series" entry per distinct curve found (usually
1 to 4). For discrete plots, create one "series" entry per data point, each with a
single-element "points" list.
"""


def _valid_points(points) -> list:
    """
    Keeps only entries that are exactly two numbers. Weaker vision models
    sometimes emit ragged points (e.g. [x] or [x, y, z]), which would otherwise
    make numpy build a 1-D object array and crash the parser for the whole figure.
    """
    valid = []
    for p in points if isinstance(points, list) else []:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                valid.append([float(p[0]), float(p[1])])
            except (TypeError, ValueError):
                continue
    return valid


def looks_synthetic(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Flags digitized series that are almost certainly not read from the image:
    constant y (e.g. all zeros), or evenly spaced X with y that is (piecewise)
    linear almost everywhere. Real curves sampled at even X steps still bend,
    so their second differences are not ~0. Catches the placeholder lines,
    "perfect triangles" and 0.999, 0.998, ... countdowns seen with LLaVA-NeXT,
    Idefics3 and Pixtral (2026-10-08 probe and comparison runs). Feeding such
    series to PySR only yields meaningless formulas with R^2 = 1.0. A genuine
    flat reference line is also flagged, which is harmless: it needs no fit.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    scale = max(np.ptp(y), 1e-12)
    if np.ptp(y) <= 1e-9 * max(1.0, np.abs(y).max()):
        return True
    # y copied from X (LLaVA-NeXT, GQA uptraining plot): PySR returns "y = x0".
    if len(X) >= 3 and np.allclose(y, X):
        return True
    if len(X) < 4:
        return False
    dx = np.diff(X)
    if np.std(dx) > 1e-6 * max(1.0, np.abs(dx).mean()):
        return False
    d2 = np.abs(np.diff(y, 2))
    return np.mean(d2 < 1e-6 * scale) >= 0.8


def discrete_points_degenerate(X, y) -> bool:
    """
    True when a discrete plot's points are all identical, e.g. every model at
    (10.0, 0.5) or (1.0, 0.0) -- a placeholder, not values read from the chart
    (Qwen2.5-VL on Net-proton fig. 9, LLaVA-NeXT on GQA, 2026-10-08 run).
    """
    pts = np.column_stack([np.asarray(X, dtype=float), np.asarray(y, dtype=float)])
    return len(pts) >= 2 and bool(np.all(pts == pts[0]))


def _clean_points(points: list) -> tuple:
    """
    Sorts extracted points by X and removes duplicate X values (keeping the first
    occurrence). Vision-extracted coordinates are noisy estimates, and unsorted or
    duplicated X values confuse symbolic regression more than a slightly smaller
    but clean dataset would.
    """
    arr = np.array(points, dtype=float)
    arr = arr[arr[:, 0].argsort()]
    _, unique_idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[np.sort(unique_idx)]
    return arr[:, 0], arr[:, 1]


def _parse_vision_output(raw_output: str, provider_used: str) -> dict:
    """
    Shared by digitize_plot() (cloud) and digitize_plot_local() so both paths
    are parsed identically against VISION_PROMPT's schema and can never drift
    apart from each other.
    """
    data = json.loads(raw_output)

    raw_series = data.get("series", [])
    plot_type = data.get("plot_type", "continuous")

    if plot_type == "continuous":
        parsed_series = []
        for entry in raw_series:
            points = _valid_points(entry.get("points", []))
            if len(points) < 2:
                continue
            X, y = _clean_points(points)
            if len(X) < 2:
                continue
            parsed_series.append({"label": entry.get("label", f"curve{len(parsed_series) + 1}"), "X": X, "y": y})

        if not parsed_series:
            print(f"[-] {provider_used} did not find enough points in any curve.")
            return None

        data["series"] = parsed_series
        print(f"[+] Success: detected '{plot_type}' plot with {len(parsed_series)} distinct curve(s).")

    else:
        kept = [(entry, _valid_points(entry.get("points", []))) for entry in raw_series]
        kept = [(entry, pts) for entry, pts in kept if pts]
        if len(kept) < 1:
            print(f"[-] {provider_used} did not find any data points in the graph.")
            return None

        data_matrix = np.array([pts[0] for _, pts in kept])
        data["X"] = data_matrix[:, 0]
        data["y"] = data_matrix[:, 1]
        data["labels"] = [entry.get("label", f"item{i + 1}") for i, (entry, _) in enumerate(kept)]
        print(f"[+] Success: detected '{plot_type}' plot with {len(kept)} data points.")

    return data


def digitize_plot(image_path: str) -> dict:
    """
    Use Gemini Vision to extract the data. It automatically classifies the chart
    as 'continuous' or 'discrete'. For continuous plots, each distinct curve
    (identified via color/legend) is extracted as a separate series instead of
    being merged into a single point cloud, since PySR cannot make sense of
    coordinates interleaved from multiple different curves.
    """
    if not os.path.exists(image_path):
        print(f"[!] Image not found: {image_path}")
        return None

    print(f"[~] Computer vision analysis in progress: {os.path.basename(image_path)}")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[!] GEMINI_API_KEY missing.")
        return None

    client = genai.Client(api_key=api_key)

    raw_output = ""
    provider_used = "Gemini"
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        try:
            response = call_gemini_with_retry(lambda: client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    VISION_PROMPT
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            ))
            raw_output = response.text.strip()
        except (QuotaExceededError, GeminiOverloadedError) as e:
            print(f"[!] Gemini unavailable ({e}). Falling back to OpenRouter free model...")
            provider_used = "OpenRouter"
            raw_output = strip_json_fences(analyze_image_with_openrouter(image_path, VISION_PROMPT))

        return _parse_vision_output(raw_output, provider_used)

    except QuotaExceededError:
        # OpenRouter itself is out of quota (and we only ever reach it after
        # Gemini already raised this) -- both free providers are genuinely
        # dead. Let this propagate instead of swallowing it like any other
        # failure, so the caller can stop and checkpoint precisely on this
        # condition rather than on unrelated per-image failures.
        raise
    except OSError:
        # DNS/connection failure: we never actually got a verdict on this
        # figure, so this is not a "no data in the chart" result. Propagate
        # instead of swallowing it, so the caller can stop and checkpoint AT
        # this figure and a rerun retries it once connectivity is back.
        raise
    except json.JSONDecodeError:
        print(f"[-] Error: {provider_used} returned a non-JSON output. Raw response:\n{raw_output}")
        return None
    except Exception as e:
        print(f"[-] Error during Vision API call: {e}")
        return None


def call_local_vision_server(image_path: str, prompt: str, max_tokens: int = None, extra_body: dict = None) -> str:
    """
    Sends one image + prompt to the local vLLM vision server and returns the raw
    text of the reply. Shared by digitize_plot_local() and
    scripts/vision_probe.py, so the probe exercises exactly the request format
    the pipeline uses.

    The image goes BEFORE the text: LLaVA-NeXT, Idefics3 and Qwen2.5-VL were all
    trained mostly with the image first, and templates that keep the given order
    (e.g. Idefics3's) would otherwise put the image after ~450 tokens of
    instructions.
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": VISION_MODEL_PATH,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.0,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if extra_body:
        # vLLM-specific request fields, e.g. {"guided_json": schema} to force
        # schema-valid JSON (tested by scripts/test_guided_json.py).
        payload.update(extra_body)
    response = requests.post(f"{VISION_SERVER_URL}/chat/completions", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def digitize_plot_local(image_path: str) -> dict:
    """
    Local-model equivalent of digitize_plot(), used for `--model local` so the
    whole pipeline can run on CINECA with no cloud dependency. The vision model
    (VISION_MODEL_PATH) runs as its own vLLM OpenAI-compatible server on a
    dedicated GPU (see scripts/run_dual_model.sbatch for the 2-GPU split) --
    this just talks to it over HTTP with the same VISION_PROMPT/schema, parsed
    by the same _parse_vision_output() as the cloud path.
    """
    if not os.path.exists(image_path):
        print(f"[!] Image not found: {image_path}")
        return None

    print(f"[~] Computer vision analysis in progress (local {VISION_MODEL_NAME}): {os.path.basename(image_path)}")

    raw_output = ""
    try:
        # 4096 caps runaway/looping outputs (LLaVA-NeXT produced ~5900 tokens on
        # a bar chart in the 2026-10-08 probe) while leaving room for the
        # largest valid answer (4 curves x 30 points ~ 2000-2500 tokens).
        extra_body = None
        if VISION_GUIDED_JSON:
            from src.llm.schemas import VISION_SCHEMA
            extra_body = {"guided_json": VISION_SCHEMA}
        if extra_body is not None:
            # Guided decoding can break for a single model/tokenizer: HTTP 500
            # (Pixtral + outlines) or unparseable garbage (Pixtral + xgrammar).
            # In both cases the same figure is retried without the constraint.
            try:
                raw_output = call_local_vision_server(image_path, VISION_PROMPT, max_tokens=4096, extra_body=extra_body)
                guided_result = _parse_vision_output(strip_json_fences(raw_output), f"{VISION_MODEL_NAME} (local, guided)")
                if guided_result is not None:
                    return guided_result
                print("[!] Guided vision output had no usable data; retrying without the schema constraint.")
            except requests.exceptions.HTTPError as e:
                print(f"[!] Guided vision request failed ({e}); retrying without the schema constraint.")
            except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as e:
                print(f"[!] Guided vision output could not be parsed ({type(e).__name__}); retrying without the schema constraint.")
        raw_output = strip_json_fences(call_local_vision_server(image_path, VISION_PROMPT, max_tokens=4096))

        return _parse_vision_output(raw_output, f"{VISION_MODEL_NAME} (local)")

    except requests.exceptions.RequestException as e:
        # Local vision server unreachable/down -- same "stop and let the
        # caller checkpoint" treatment as a cloud network failure (OSError)
        # in digitize_plot() above.
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status is not None and status >= 500:
            # A 5xx from vLLM means its engine process died (e.g. OOM-killed,
            # job 59727080), not a network problem: no later figure can succeed.
            print(f"[-] Local vision server failed (HTTP {status}): its engine has probably crashed -- check the vision_server_*.log.")
        else:
            print(f"[-] Local vision server error: {e}")
        raise OSError(str(e))
    except json.JSONDecodeError:
        print(f"[-] Error: local model returned a non-JSON output. Raw response:\n{raw_output}")
        return None
    except Exception as e:
        print(f"[-] Error during local Vision call: {e}")
        return None
