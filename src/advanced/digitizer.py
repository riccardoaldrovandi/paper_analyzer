import os
import json
import numpy as np
from google import genai
from google.genai import types
from src.utils.retry import call_gemini_with_retry, QuotaExceededError
from src.utils.helpers import strip_json_fences
from src.llm.openrouter_client import analyze_image_with_openrouter


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

    print(f"[~] Computer vision analysis in progress with Gemini: {os.path.basename(image_path)}")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[!] GEMINI_API_KEY missing.")
        return None

    client = genai.Client(api_key=api_key)

    vision_prompt = """
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

    raw_output = ""
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        try:
            response = call_gemini_with_retry(lambda: client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    vision_prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            ))
            raw_output = response.text.strip()
        except QuotaExceededError as e:
            print(f"[!] Gemini free-tier quota exhausted ({e}). Falling back to OpenRouter free model...")
            raw_output = strip_json_fences(analyze_image_with_openrouter(image_path, vision_prompt))

        data = json.loads(raw_output)

        raw_series = data.get("series", [])
        plot_type = data.get("plot_type", "continuous")

        if plot_type == "continuous":
            parsed_series = []
            for entry in raw_series:
                points = entry.get("points", [])
                if len(points) < 2:
                    continue
                X, y = _clean_points(points)
                if len(X) < 2:
                    continue
                parsed_series.append({"label": entry.get("label", f"curve{len(parsed_series) + 1}"), "X": X, "y": y})

            if not parsed_series:
                print("[-] Gemini did not find enough points in any curve.")
                return None

            data["series"] = parsed_series
            print(f"[+] Success: detected '{plot_type}' plot with {len(parsed_series)} distinct curve(s).")

        else:
            all_points = [entry["points"][0] for entry in raw_series if entry.get("points")]
            if len(all_points) < 1:
                print("[-] Gemini did not find any data points in the graph.")
                return None

            data_matrix = np.array(all_points)
            data["X"] = data_matrix[:, 0]
            data["y"] = data_matrix[:, 1]
            data["labels"] = [entry.get("label", f"item{i + 1}") for i, entry in enumerate(raw_series) if entry.get("points")]
            print(f"[+] Success: detected '{plot_type}' plot with {len(all_points)} data points.")

        return data

    except json.JSONDecodeError:
        print(f"[-] Error: Gemini returned a non-JSON output. Raw response:\n{raw_output}")
        return None
    except Exception as e:
        print(f"[-] Error during Vision API call: {e}")
        return None
