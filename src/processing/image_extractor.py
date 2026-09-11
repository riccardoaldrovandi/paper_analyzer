import os
import json
import pymupdf
import tarfile
from google import genai
from google.genai import types
from src.utils.retry import call_gemini_with_retry, QuotaExceededError
from src.utils.helpers import strip_json_fences
from src.llm.openrouter_client import analyze_image_with_openrouter

def _is_actual_plot(image_path: str) -> bool:
    """
    Smart support function: uses Gemini Vision to verify whether the rendered image actually 
    contains a graph or plot with curves or numerical data, discarding pages consisting solely of text 
    or logic diagrams.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        # If the key is not configured, for safety reasons we skip the filter and keep the image.
        return True

    try:
        client = genai.Client(api_key=api_key)
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            "Analyze this document page image. Does this page contain a primary data plot, "
            "statistical chart, bar chart, line graph, or mathematical curve with coordinate axes? "
            "Answer ONLY with 'YES' if it features a data plot/chart, or 'NO' if it is just text, "
            "tables, or architectural block diagrams without plotted data curves."
        )

        try:
            response = call_gemini_with_retry(lambda: client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0, # Deterministic at 100%
                )
            ))
            answer = response.text.strip().upper()
            return "YES" in answer
        except QuotaExceededError as e:
            # This is a binary keep/discard gate: a wrong "NO" silently and
            # permanently drops a legitimate chart with no visible error, which is
            # worse than just skipping the filter. The free OpenRouter fallback
            # model is noticeably weaker at this kind of visual classification
            # (unlike the digitizer/crop steps, there's no safe way to sanity-check
            # its verdict downstream), so on quota exhaustion we keep the page
            # instead of trusting it to decide what to throw away.
            print(f"[!] Gemini quota exhausted during smart filtering ({e}). Skipping filter and keeping the page.")
            return True
    except Exception as e:
        print(f"[!] Error during smart filtering: {e}")
        return True  # In case of connection error, we preserve the image


def _crop_individual_plots(page, page_filepath: str, output_dir: str, base_name: str, page_index: int) -> list:
    """
    Uses Gemini Vision to identify the bounding boxes of individual charts/plots 
    on the page and crops them into separate, clean image files.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return [page_filepath] # Fallback to the full page if the API key is missing

    try:
        client = genai.Client(api_key=api_key)
        
        # Prompt to request normalized bounding box coordinates for each distinct plot
        prompt = (
            "Analyze this document page. Locate every distinct data plot, statistical chart, "
            "bar chart, or line graph present on the page. "
            "For each plot, provide its bounding box coordinates [ymin, xmin, ymax, xmax] scaled from 0 to 1000. "
            "You MUST return ONLY a valid JSON array of objects. Example: "
            '[{"box_2d": [100, 100, 300, 400], "label": "fig5"}]'
        )
        
        with open(page_filepath, "rb") as f:
            image_bytes = f.read()

        try:
            response = call_gemini_with_retry(lambda: client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            ))
            raw_output = response.text.strip()
        except QuotaExceededError as e:
            print(f"[!] Gemini quota exhausted during smart cropping ({e}). Falling back to OpenRouter free model...")
            raw_output = strip_json_fences(analyze_image_with_openrouter(page_filepath, prompt))

        boxes = json.loads(raw_output)
        cropped_images = []
        rect = page.rect # Original PDF page dimensions
        
        for i, item in enumerate(boxes):
            box = item.get("box_2d") # Scaled coordinates [ymin, xmin, ymax, xmax]
            if not box or len(box) != 4:
                continue
                
            # Convert 0-1000 scale coordinates back to real PDF points/pixels
            y0 = (box[0] / 1000.0) * rect.height
            x0 = (box[1] / 1000.0) * rect.width
            y1 = (box[2] / 1000.0) * rect.height
            x1 = (box[3] / 1000.0) * rect.width
            
            # Apply a small safety margin to avoid cutting axes labels
            clip_rect = pymupdf.Rect(max(0, x0 - 10), max(0, y0 - 10), min(rect.width, x1 + 10), min(rect.height, y1 + 10))
            
            # Render and save the cropped area at high resolution (300 DPI)
            pix = page.get_pixmap(dpi=300, clip=clip_rect)
            label = item.get("label", f"plot{i+1}")
            crop_filename = f"{base_name}_page{page_index+1}_crop_{label}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            pix.save(crop_path)
            cropped_images.append(crop_path)
            
        # If cropping succeeded, return cropped plots and remove the bulky full-page render
        if cropped_images:
            if os.path.exists(page_filepath):
                os.remove(page_filepath)
            return cropped_images

    except Exception as e:
        print(f"[!] Error during smart cropping, falling back to full page: {e}")

    return [page_filepath]


def extract_images_from_pdf(pdf_path: str, base_output_dir: str = "extracted_images") -> list:
    """
    Extracts raster images and uses a smart filter (Gemini Vision) alongside 
    automatic smart cropping to save strictly isolated charts and plots.
    """
    if not os.path.exists(pdf_path):
        print(f"[!] PDF not found: {pdf_path}")
        return []

    os.makedirs(base_output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(base_output_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    saved_images = []

    print(f"[~] Scanning PDF and applying smart filtering & cropping to charts...")
    for page_index in range(len(doc)):
        page = doc[page_index]
        
        # 1. Extracting standard raster images (e.g., diagrams saved as pure images)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            if base_image["width"] < 200 or base_image["height"] < 200:
                continue
                
            image_filename = f"{base_name}_page{page_index+1}_raster{img_index+1}.{image_ext}"
            image_filepath = os.path.join(output_dir, image_filename)
            
            with open(image_filepath, "wb") as f:
                f.write(image_bytes)
            saved_images.append(image_filepath)

        # 2. Page rendering with mentions of figures and intelligent filtering/cropping
        page_text = page.get_text("text").lower()
        if "fig" in page_text or "figure" in page_text or "plot" in page_text:
            pix = page.get_pixmap(dpi=200)
            page_filename = f"{base_name}_page{page_index+1}_full_render.png"
            page_filepath = os.path.join(output_dir, page_filename)
            pix.save(page_filepath)
            
            # --- SMART FILTER ---
            print(f"[~] Visual page analysis {page_index+1} via Gemini Vision...")
            if _is_actual_plot(page_filepath):
                print(f"[+] Found valid chart(s) at page {page_index+1}. Cropping individual plots...")
                # --- SMART CROP ---
                valid_plots = _crop_individual_plots(page, page_filepath, output_dir, base_name, page_index)
                saved_images.extend(valid_plots)
            else:
                print(f"[-] Page {page_index+1} mentions a figure but is not a numerical plot. Removed.")
                os.remove(page_filepath)

    doc.close()
    return saved_images


def extract_images_from_latex(tar_path: str, base_output_dir: str = "extracted_images") -> list:
    """
    Extracts images from a LaTeX archive by converting vector files to PNG.
    """
    if not os.path.exists(tar_path):
        print(f"[!] LaTeX archive not found: {tar_path}")
        return []

    os.makedirs(base_output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(tar_path))[0]
    if base_name.endswith("_source"):
        base_name = base_name[:-7]
        
    output_dir = os.path.join(base_output_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    saved_images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.pdf', '.eps')
    
    print(f"[~] Extracting images from LaTeX archive...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                name_lower = member.name.lower()
                if member.isfile() and name_lower.endswith(valid_extensions):
                    tar.extract(member, path=output_dir)
                    extracted_path = os.path.join(output_dir, member.name)
                    
                    if name_lower.endswith(".pdf") or name_lower.endswith(".eps"):
                        try:
                            plot_doc = pymupdf.open(extracted_path)
                            plot_pix = plot_doc[0].get_pixmap(dpi=300)
                            png_path = extracted_path[:-4] + ".png"
                            plot_pix.save(png_path)
                            plot_doc.close()
                            
                            saved_images.append(png_path)
                            os.remove(extracted_path)
                        except Exception as e:
                            print(f"[!] Error converting chart {member.name}: {e}")
                    else:
                        saved_images.append(extracted_path)
    except Exception as e:
        print(f"[!] Error during processing of {tar_path}: {e}")

    return saved_images