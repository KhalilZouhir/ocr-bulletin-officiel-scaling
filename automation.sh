#!/bin/bash

# Base path
BASE_PATH="/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel"
BATCHS_PATH="$BASE_PATH/batchs"

# Check if the batchs directory exists
if [ ! -d "$BATCHS_PATH" ]; then
    echo "Error: Directory '$BATCHS_PATH' does not exist!"
    exit 1
fi

# Navigate to the batchs directory
cd "$BATCHS_PATH" || exit 1

# Find the latest batch folder (only purely numeric ones)
latest_folder=$(find . -maxdepth 1 -type d -name 'batch[0-9]*' | sed 's|./batch||' | grep -E '^[0-9]+$' | sort -n | tail -1)

if [ -z "$latest_folder" ]; then
    echo "No 'batch' folders found in $BATCHS_PATH"
    echo "Starting from batch1..."
    latest_num=0
else
    latest_num=$latest_folder
    echo "The latest folder is: batch$latest_num"
fi

# Ask user how many new folders to create
read -p "How many new folders do you want to create? " num_folders

# Validate input
if ! [[ "$num_folders" =~ ^[0-9]+$ ]] || [ "$num_folders" -le 0 ]; then
    echo "Error: Please enter a valid positive number!"
    exit 1
fi

# Create new folders with subdirectories and files
echo "Creating $num_folders new folder(s)..."
for ((i=1; i<=num_folders; i++)); do
    new_num=$((latest_num + i))
    new_folder="batch$new_num"
    
    # Create main folder
    mkdir "$new_folder"
    echo "Created: $new_folder"
    
    # Create subdirectories
    mkdir -p "$new_folder/logs$new_num"
    mkdir -p "$new_folder/pdf_documents_$new_num"
    mkdir -p "$new_folder/processing_results_$new_num"
    echo "  ├─ Created subdirectories: logs$new_num, pdf_documents_$new_num, processing_results_$new_num"
    
    # Create split_code PBS file
    cat > "$new_folder/split_code$new_num.pbs" << 'SPLIT_PBS'
#!/bin/bash
#PBS -N split_codeNUM
#PBS -l select=1:ncpus=20:mem=180gb:ngpus=1
#PBS -q gpu_1d
#PBS -o logsNUM/split_output_NUM.log
#PBS -e logsNUM/split_error_NUM.log

# === Load modules & activate conda ===
module use /app/common/modules
module load anaconda3-2024.10
source activate  khalil_vllm

# === Move to working directory ===
cd /home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/batchs/batchNUM/

python split_codeNUM.py 
SPLIT_PBS
    
    # Replace NUM with actual number
    sed -i "s/NUM/$new_num/g" "$new_folder/split_code$new_num.pbs"
    
    # Create split_code Python file
    cat > "$new_folder/split_code$new_num.py" << 'SPLIT_PY'
# ============================================================================
# UNIFIED PDF OCR PIPELINE WITH COLUMN SPLITTING
# ============================================================================

import os
import time
import base64
import requests
import glob
import math
import shutil
import cv2
import numpy as np
from typing import Tuple, Union, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from tqdm import tqdm

# ========== CONFIGURATION ==========
# VLLM API Configuration
VLLM_API_URL = "http://localhost:9996/v1/chat/completions"
MODEL_PATH = "/home/skiredj.abderrahman/khalil/chandra"

# LLM Generation Parameters
GENERATION_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 30000,
    "top_p": 0.9,
}

# Request timeout
REQUEST_TIMEOUT_SECONDS = 180

# Parallel processing
N_WORKERS = 17

# OCR Prompt
OCR_PROMPT = """Extract all text exactly as it appears in this image.
Preserve the original layout, reading order, formatting, and structure.
Keep all tables as proper Markdown tables.
Maintain all mathematical formulas, equations, and special characters.
Do not rewrite, summarize, or modify any content.
If any text is unclear or unreadable, write [UNCLEAR].
Output only the extracted text without any preamble or explanation."""

# Folders
PDF_FOLDER = r"/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/batchs/batchNUM/pdf_documents_NUM"
PROCESSING_BASE = r"/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/batchs/batchNUM/processing_results_NUM"
OUTPUT_FOLDER = os.path.join(PROCESSING_BASE, "documents_transformed_to_markdown")
TEMP_FOLDER = os.path.join(PROCESSING_BASE, "tempo")

# ========== IMAGE PREPROCESSING FUNCTIONS (FROM CODE 2) ==========

def normalize_illumination(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, blur, scale=255)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

def enhance_contrast_and_sharpness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    return sharp

def upscale_opencv(img, scale=2):
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)
    return upscaled

def preprocess_page(img):
    img = normalize_illumination(img)
    img = enhance_contrast_and_sharpness(img)
    img = upscale_opencv(img, scale=2)
    return img

def _read_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Impossible de lire l'image : {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path.copy()
    else:
        raise TypeError("image_or_path doit être un chemin ou une image numpy.")
    img = preprocess_page(img)
    return img

# ========== TABLE DETECTION (FROM CODE 2) ==========

def is_big_table(img, debug=False, base_name=""):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = th if np.mean(th) > 127 else cv2.bitwise_not(th)
    bin_img = cv2.medianBlur(bin_img, 3)

    # Vertical lines
    vert_len = max(10, H // 20)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vertical = cv2.morphologyEx(255 - bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
    _, vertical = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY)

    table_big_min_height_ratio = 0.85
    table_many_verticals = 5
    table_min_spread_ratio = 0.5
    table_min_width_coverage = 0.06

    tall_verticals = []
    total_v_width = 0
    contours_vert, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours_vert:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(table_big_min_height_ratio * H) and (1 <= w <= int(0.03 * W)):
            tall_verticals.append((x, y, w, h))
            total_v_width += w

    spread_ok = False
    if len(tall_verticals) >= table_many_verticals:
        xs = np.array([x + w / 2.0 for x, _, w, _ in tall_verticals], dtype=float)
        width_spread = np.ptp(xs) if xs.size else 0.0
        if width_spread >= table_min_spread_ratio * W:
            spread_ok = True

    width_coverage = (total_v_width / float(W)) if W > 0 else 0.0
    looks_like_big_table = (
            (len(tall_verticals) >= table_many_verticals)
            and spread_ok
            and (width_coverage >= table_min_width_coverage)
    )

    # Horizontal lines
    horiz_len = max(10, W // 20)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horizontal = cv2.morphologyEx(255 - bin_img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horiz_kernel, iterations=1)
    _, horizontal = cv2.threshold(horizontal, 0, 255, cv2.THRESH_BINARY)
    contours_horiz, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_horizontal_lines = []
    medium_horizontal_lines = []

    for c in contours_horiz:
        x, y, w, h = cv2.boundingRect(c)
        line_width_ratio = w / W
        if line_width_ratio > 0.5:
            long_horizontal_lines.append((x, y, w, h))
        elif line_width_ratio > 0.2:
            medium_horizontal_lines.append((x, y, w, h))

    many_horizontal = len(long_horizontal_lines)
    medium_horizontal = len(medium_horizontal_lines)

    line_density = many_horizontal / (H / 100.0)

    if long_horizontal_lines:
        line_ys = [y + h / 2 for x, y, w, h in long_horizontal_lines]
        vertical_spread = (max(line_ys) - min(line_ys)) / H if len(line_ys) > 1 else 0
    else:
        vertical_spread = 0

    spacing_regularity = 0
    if len(long_horizontal_lines) >= 3:
        line_ys = sorted([y + h / 2 for x, y, w, h in long_horizontal_lines])
        spacings = [line_ys[i + 1] - line_ys[i] for i in range(len(line_ys) - 1)]
        if spacings:
            avg_spacing = np.mean(spacings)
            spacing_variance = np.var(spacings) / (avg_spacing + 1)
            spacing_regularity = 1.0 / (1.0 + spacing_variance)

    text_pixels = np.sum(255 - bin_img > 127)
    line_pixels = np.sum(horizontal > 127) + np.sum(vertical > 127)
    line_to_text_ratio = line_pixels / (text_pixels + 1)

    old_table_detection = many_horizontal >= 4

    dense_horizontal_table = (
            many_horizontal >= 5 and
            line_density >= 2.0 and
            vertical_spread >= 0.6
    )

    regular_table = (
            many_horizontal >= 4 and
            spacing_regularity >= 0.7 and
            vertical_spread >= 0.5
    )

    line_heavy_document = (
            many_horizontal >= 5 and
            line_to_text_ratio >= 0.25 and
            (medium_horizontal + many_horizontal) >= 12 and
            vertical_spread >= 0.4
    )

    table_detected = (
            looks_like_big_table or
            old_table_detection or
            dense_horizontal_table or
            regular_table or
            line_heavy_document
    )

    return table_detected

# ========== COLUMN SPLITTING FUNCTIONS (FROM CODE 2) ==========

def has_significant_text(img_region, min_text_ratio=0.01):
    if len(img_region.shape) == 3:
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_region
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_pixels = np.sum(binary > 0)
    total_pixels = binary.shape[0] * binary.shape[1]
    text_ratio = text_pixels / total_pixels
    return text_ratio > min_text_ratio

def find_text_density_split(img, debug=False):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    H, W = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 15), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_h_lines = sum(1 for c in h_contours if cv2.boundingRect(c)[2] > 0.7 * W)
    medium_h_lines = sum(1 for c in h_contours if 0.4 * W < cv2.boundingRect(c)[2] <= 0.7 * W)
    total_h_lines = long_h_lines + medium_h_lines

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, H // 20)))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_v_lines = sum(1 for c in v_contours if cv2.boundingRect(c)[3] > 0.6 * H)
    medium_v_lines = sum(1 for c in v_contours if 0.3 * H < cv2.boundingRect(c)[3] <= 0.6 * H)
    total_v_lines = long_v_lines + medium_v_lines

    if total_h_lines >= 2 and total_v_lines >= 2:
        grid_mask = cv2.bitwise_and(horizontal_lines, vertical_lines)
        intersection_contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        intersections = len(intersection_contours)
        intersections_per_area = intersections / ((W * H) / 10000)

        if intersections > 0:
            intersection_points = []
            for c in intersection_contours:
                x, y, w, h = cv2.boundingRect(c)
                intersection_points.append((x + w // 2, y + h // 2))

            if len(intersection_points) >= 4:
                distances = []
                for i, (x1, y1) in enumerate(intersection_points):
                    for j, (x2, y2) in enumerate(intersection_points[i + 1:], i + 1):
                        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        distances.append(dist)

                if distances:
                    dist_variance = np.var(distances)
                    dist_mean = np.mean(distances)
                    regularity_score = dist_variance / (dist_mean ** 2) if dist_mean > 0 else 0
                else:
                    regularity_score = 0
            else:
                regularity_score = 0
        else:
            regularity_score = 0
            intersections_per_area = 0

        if intersections >= 20 and regularity_score < 0.1 and intersections_per_area > 0.3:
            return None
        if total_h_lines >= 6 and intersections >= 15 and intersections_per_area > 0.2:
            return None
        if total_h_lines >= 4 and total_v_lines >= 4 and intersections >= 25:
            return None

    vertical_projection = np.sum(binary, axis=0) / 255
    window_size = max(1, int(W * 0.02))
    cand_points = []
    maxpv = vertical_projection.max() if vertical_projection.size else 0.0

    for x in range(window_size, W - window_size):
        a = max(0, x - window_size // 2)
        b = min(W, x + window_size // 2)
        window_density = np.mean(vertical_projection[a:b])
        if window_density < maxpv * 0.1:
            cand_points.append((x, window_density))

    if not cand_points:
        return None

    min_center = int(0.25 * W)
    max_center = int(0.75 * W)
    cand_points = [p for p in cand_points if min_center <= p[0] <= max_center]
    if not cand_points:
        return None

    center_x = W // 2
    best_split = min(cand_points, key=lambda p: abs(p[0] - center_x))[0]

    def text_ratio(region):
        if len(region.shape) == 3:
            rg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            rg = region
        _, b = cv2.threshold(rg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return (np.sum(b > 0) / float(b.size)) if b.size else 0.0

    left_ratio = text_ratio(img[:, :best_split])
    right_ratio = text_ratio(img[:, best_split:])

    min_text_ratio = 0.01
    balance_min_factor = 0.5

    if left_ratio < min_text_ratio or right_ratio < min_text_ratio:
        return None

    if min(left_ratio, right_ratio) < balance_min_factor * max(left_ratio, right_ratio):
        return None

    return best_split

def has_unique_central_vertical_rule(
    image_or_path: Union[str, np.ndarray],
    center_band_ratio: float = 0.20,
    min_height_ratio: float = 0.70,
    max_line_width_ratio: float = 0.03,
    min_line_width_px: int = 1,
    max_gap_ratio: float = 0.10,
    max_gap_runs: int = 2,
    gap_run_ratio: float = 0.01,
    max_candidates_allowed: int = 3,
    debug: bool = False,
    base_name: str = ""
) -> Tuple[bool, Dict]:
    bgr_orig = _read_image(image_or_path)
    orig_H, orig_W = bgr_orig.shape[:2]
    scale_factor = 1.0
    max_side = max(orig_H, orig_W)
    bgr = bgr_orig
    if max_side > 2000:
        scale_factor = 2000.0 / max_side
        bgr = cv2.resize(
            bgr_orig, (int(orig_W*scale_factor), int(orig_H*scale_factor)),
            interpolation=cv2.INTER_AREA
        )
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = th_otsu if np.mean(th_otsu) > 127 else cv2.bitwise_not(th_otsu)
    bin_img = cv2.medianBlur(bin_img, 3)
    vert_len = max(10, H // 20)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vertical = cv2.morphologyEx(255 - bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
    _, vertical = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY)

    table_big_min_height_ratio = 0.85
    table_many_verticals = 5
    table_min_spread_ratio = 0.50
    table_min_width_coverage = 0.06
    tall_verticals = []
    total_v_width = 0
    contours_all, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_all:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(table_big_min_height_ratio * H) and (min_line_width_px <= w <= int(max_line_width_ratio * W)):
            tall_verticals.append((x, y, w, h))
            total_v_width += w
    spread_ok = False
    if len(tall_verticals) >= table_many_verticals:
        xs = np.array([x + w/2.0 for x,_,w,_ in tall_verticals], dtype=float)
        width_spread = np.ptp(xs) if xs.size else 0.0
        if width_spread >= table_min_spread_ratio * W:
            spread_ok = True
    width_coverage = (total_v_width / float(W)) if W > 0 else 0.0
    looks_like_big_table = (len(tall_verticals) >= table_many_verticals) and spread_ok and (width_coverage >= table_min_width_coverage)

    mid = W // 2
    band_half = int(center_band_ratio * W)
    x0 = max(0, mid - band_half)
    x1 = min(W, mid + band_half)
    central_band = np.zeros_like(vertical)
    central_band[:, x0:x1] = vertical[:, x0:x1]
    contours, _ = cv2.findContours(central_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < int(min_height_ratio * H):
            continue
        if w < min_line_width_px or w > int(max_line_width_ratio * W):
            continue
        x_center = x + w / 2.0
        if not (mid - band_half <= x_center <= mid + band_half):
            continue
        roi = (255 - bin_img)[y:y+h, max(0, x):min(W, x+w)]
        row_has_ink = (np.max(roi, axis=1) > 0).astype(np.uint8)
        min_gap = max(3, int(gap_run_ratio * H))
        gaps, run = [], 0
        for val in row_has_ink:
            if val == 0:
                run += 1
            else:
                if run >= min_gap:
                    gaps.append(run)
                run = 0
        if run >= min_gap:
            gaps.append(run)
        total_gap = sum(gaps)
        if total_gap > max_gap_ratio * H or len(gaps) > max_gap_runs:
            continue
        candidates.append((x, y, w, h, {"gaps": gaps, "total_gap": total_gap}))

    central_tall_verticals = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= int(min_height_ratio * H) and (min_line_width_px <= w <= int(max_line_width_ratio * W)):
            central_tall_verticals += 1
    if central_tall_verticals > 1:
        return (False, {
            "original_image_size": (orig_H, orig_W),
            "processed_image_size": (H, W),
            "scale_factor": scale_factor,
            "center_band": (x0, x1),
            "candidates_found": len(candidates),
            "central_tall_verticals": central_tall_verticals,
            "table_blocked": True,
        })

    global_like = 0
    for c in contours_all:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(min_height_ratio * H) and (min_line_width_px <= w <= int(max_line_width_ratio * W)):
            global_like += 1

    details = {
        "original_image_size": (orig_H, orig_W),
        "processed_image_size": (H, W),
        "scale_factor": scale_factor,
        "center_band": (x0, x1),
        "candidates_found": len(candidates),
        "global_vertical_candidates": global_like,
        "candidates": [{"bbox": (int(x), int(y), int(w), int(h)), **stats}
                       for (x, y, w, h, stats) in candidates],
        "big_table_heuristic": {
            "looks_like_big_table": bool(looks_like_big_table),
            "very_tall_verticals": int(len(tall_verticals)),
            "spread_ok": bool(spread_ok),
            "width_coverage": float(width_coverage)
        }
    }
    
    if global_like > max_candidates_allowed:
        return (False, details)
    if looks_like_big_table:
        return (False, details)
    is_true = (len(candidates) == 1)
    return (is_true, details)

def get_split_x_from_details(details: Dict) -> Optional[int]:
    if details.get("candidates_found", 0) != 1:
        return None
    (x, y, w, h) = details["candidates"][0]["bbox"]
    split_x_processed = int(round(x + w / 2))
    scale = float(details.get("scale_factor", 1.0))
    if scale <= 0:
        scale = 1.0
    split_x_original = int(round(split_x_processed / scale))
    orig_W = details.get("original_image_size", (0, 0))[1]
    if orig_W:
        split_x_original = max(1, min(orig_W - 1, split_x_original))
    return split_x_original

def _derive_names(image_or_path: Union[str, np.ndarray],
                  base_name: Optional[str]) -> Tuple[str, str]:
    if isinstance(image_or_path, str):
        stem, ext = os.path.splitext(os.path.basename(image_or_path))
        if ext == "":
            ext = ".png"
        return stem, ext
    else:
        if not base_name:
            raise ValueError("For ndarray input, please provide base_name.")
        stem, ext = os.path.splitext(base_name)
        if ext == "":
            ext = ".png"
        return stem, ext

def perform_split(img, split_x, dest_dir, stem, ext, left_suffix, right_suffix, details):
    os.makedirs(dest_dir, exist_ok=True)
    left_img = img[:, :split_x]
    right_img = img[:, split_x:]
    left_path = os.path.join(dest_dir, f"{stem}{left_suffix}{ext}")
    right_path = os.path.join(dest_dir, f"{stem}{right_suffix}{ext}")
    ok_left = cv2.imwrite(left_path, left_img)
    ok_right = cv2.imwrite(right_path, right_img)
    if ok_left and ok_right:
        return {
            "split": True,
            "left_path": left_path,
            "right_path": right_path,
            "split_x": split_x,
            "details": details,
        }
    else:
        out_path = os.path.join(dest_dir, f"{stem}{ext}")
        cv2.imwrite(out_path, img)
        return {
            "split": False,
            "reason": "write_error",
            "copied_path": out_path
        }

def enhanced_split_for_administrative_docs(
        image_or_path: Union[str, np.ndarray],
        dest_dir: str,
        base_name: Optional[str] = None,
        left_suffix: str = "_left",
        right_suffix: str = "_right",
        debug=False
) -> Dict:
    stem, ext = _derive_names(image_or_path, base_name)
    img = _read_image(image_or_path)
    orig_H, orig_W = img.shape[:2]

    is_two_col, details = has_unique_central_vertical_rule(
        image_or_path, debug=debug, base_name=stem)

    if is_two_col:
        split_x = get_split_x_from_details(details)
        if split_x is not None:
            split_x = max(1, min(orig_W - 1, int(split_x)))
            left_has_text = has_significant_text(img[:, :split_x])
            right_has_text = has_significant_text(img[:, split_x:])

            if left_has_text and right_has_text:
                return perform_split(img, split_x, dest_dir, stem, ext, left_suffix, right_suffix, details)

    alternative_split_x = find_text_density_split(img, debug=debug)
    if alternative_split_x is not None:
        left_has_text = has_significant_text(img[:, :alternative_split_x])
        right_has_text = has_significant_text(img[:, alternative_split_x:])

        if left_has_text and right_has_text:
            return perform_split(img, alternative_split_x, dest_dir, stem, ext, left_suffix, right_suffix,
                                 {"method": "text_density", "split_x": alternative_split_x})

    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"{stem}{ext}")
    if isinstance(image_or_path, str) and os.path.isfile(image_or_path):
        shutil.copy2(image_or_path, out_path)
    else:
        cv2.imwrite(out_path, img)

    return {
        "split": False,
        "reason": "no_valid_split_both_sides",
        "copied_path": out_path
    }

def split_if_two_column(
        image_or_path,
        dest_dir,
        base_name=None,
        debug=False
):
    img = _read_image(image_or_path)
    stem, ext = _derive_names(image_or_path, base_name)

    if is_big_table(img, debug=debug, base_name=stem):
        if debug:
            print("🛑 Table détectée : aucun split.")
        out_path = os.path.join(dest_dir, f"{stem}{ext}")
        if isinstance(image_or_path, str) and os.path.isfile(image_or_path):
            shutil.copy2(image_or_path, out_path)
        else:
            cv2.imwrite(out_path, img)
        return {
            "split": False,
            "reason": "big_table_detected",
            "copied_path": out_path
        }

    return enhanced_split_for_administrative_docs(
        image_or_path, dest_dir, base_name=base_name, debug=debug
    )

# ========== OCR FUNCTIONS (FROM CODE 1) ==========

def call_vllm_ocr(image_path, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                "model": MODEL_PATH,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT
                            }
                        ]
                    }
                ],
                **GENERATION_PARAMS
            }
            
            response = requests.post(
                VLLM_API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and result['choices']:
                extracted_text = result['choices'][0]['message']['content']
                return extracted_text
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except requests.exceptions.Timeout:
            attempt += 1
            print(f"Timeout on attempt {attempt}/{retries} for {image_path}")
            if attempt < retries:
                time.sleep(5)
            else:
                raise Exception(f"Request timed out after {retries} attempts")
                
        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"Request error on attempt {attempt}/{retries} for {image_path}: {e}")
            if attempt < retries:
                time.sleep(3)
            else:
                raise Exception(f"Request failed after {retries} attempts: {e}")
                
        except Exception as e:
            attempt += 1
            print(f"Error on attempt {attempt}/{retries} for {image_path}: {e}")
            if attempt < retries:
                time.sleep(3)
            else:
                raise

def process_image(input_image_path, output_text_path, retries=3):
    if os.path.exists(output_text_path):
        print(f"Skipping already processed file: {output_text_path}")
        return

    try:
        print(f"Processing image: {input_image_path}")
        extracted_text = call_vllm_ocr(input_image_path, retries=retries)
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"Finished processing: {input_image_path}")
        
    except Exception as e:
        print(f"Failed to process {input_image_path}: {e}")
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(f"[OCR FAILED: {str(e)}]")

def clean_unnecessary_linebreaks(text):
    lines = text.split('\n')
    cleaned_text = []

    for i in range(len(lines)):
        if lines[i].startswith("=== Page"):
            cleaned_text.append(lines[i])
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                cleaned_text.append("")
        else:
            if (i > 0 and
                not lines[i - 1].startswith("=== Page") and
                not lines[i - 1].rstrip().endswith('.')):
                if cleaned_text and cleaned_text[-1] != "":
                    cleaned_text[-1] = cleaned_text[-1].rstrip() + " " + lines[i].lstrip()
                else:
                    cleaned_text.append(lines[i])
            else:
                cleaned_text.append(lines[i])

    return '\n'.join(cleaned_text)

# ========== MAIN UNIFIED PIPELINE ==========

def pdf_to_images_with_split(pdf_path, temp_img_dir, split_dir, pdf_base_name, zoom=4, debug=False):
    """
    Convert PDF to images and apply column splitting
    """
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    
    # Step 1: Convert PDF to images
    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_filename = f"{pdf_base_name}_{page_number + 1}.png"
        image_path = os.path.join(temp_img_dir, image_filename)
        pix.save(image_path)
    pdf_document.close()
    
    # Step 2: Apply column splitting to each image
    image_files = sorted(glob.glob(os.path.join(temp_img_dir, "*.png")))
    for img_path in image_files:
        result = split_if_two_column(img_path, split_dir, debug=debug)
        if debug:
            base_name = os.path.basename(img_path)
            print(f"[{base_name}] Split result: {result.get('split')}")

def ocr_split_images(split_dir, pdf_base_name, output_md_path):
    """
    Run OCR on split images and generate markdown output
    """
    txt_dir = os.path.join(TEMP_FOLDER, f"tempo_res_{pdf_base_name}")
    os.makedirs(txt_dir, exist_ok=True)
    
    # Get all images from split directory
    # Get all images from split directory
    image_files = glob.glob(os.path.join(split_dir, "*.png"))
    
    # Custom sort: for Arabic, process RIGHT pages before LEFT pages
    def arabic_sort_key(path):
        filename = os.path.basename(path)
        
        # Extract page number using regex
        import re
        
        # Remove the extension first
        name_without_ext = filename.replace('.png', '')
        
        # Check for _right or _left suffix
        if name_without_ext.endswith('_right'):
            base_name = name_without_ext[:-6]  # Remove '_right'
            suffix_order = 0  # Right comes first
        elif name_without_ext.endswith('_left'):
            base_name = name_without_ext[:-5]  # Remove '_left'
            suffix_order = 1  # Left comes second
        else:
            base_name = name_without_ext
            suffix_order = 0  # No split, treat like right
        
        # ✅ FIXED: Extract ALL numbers, get the last one (handles -bis, -ter, etc.)
        numbers = re.findall(r'\d+', base_name)
        
        if numbers:
            page_num = int(numbers[-1])  # Get the LAST number found
        else:
            page_num = 0
        
        return (page_num, suffix_order)
    
    image_files = sorted(image_files, key=arabic_sort_key)
    
    if not image_files:
        print(f"No images found in {split_dir}")
        return
    
    # Prepare OCR tasks
    tasks = []
    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_text_path = os.path.join(txt_dir, f"{img_name}.txt")
        tasks.append((img_path, output_text_path))
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_image, in_path, out_path): (in_path, out_path) 
                  for in_path, out_path in tasks}
        for future in as_completed(futures):
            in_path, out_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Unhandled error processing {in_path}: {e}")
    
    # Combine all page texts
    combined_text = []
    for i, img_path in enumerate(image_files, 1):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        text_path = os.path.join(txt_dir, f"{img_name}.txt")
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                page_content = f.read()
                combined_text.append(f"=== Page {i} ({img_name}) ===\n{page_content}\n")
        else:
            combined_text.append(f"=== Page {i} ({img_name}) ===\n[OCR failed for this page]\n")
    
    raw_text = "\n".join(combined_text)
    cleaned_text = clean_unnecessary_linebreaks(raw_text)
    
    # Save output
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    print(f"Saved OCR output to {output_md_path}")

def step1_pdf_to_split_images_all():
    """
    Step 1: Convert all PDFs to images and apply splitting
    Run this in one Jupyter cell
    """
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    
    pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
        return
    
    print(f"Found {len(pdf_files)} PDFs to process.")
    
    for pdf_path in tqdm(pdf_files, desc="Step 1: Converting & Splitting PDFs"):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_img_dir = os.path.join(TEMP_FOLDER, f"tempo_{base_name}")
        split_dir = os.path.join(TEMP_FOLDER, f"tempo_split_{base_name}")
        
        print(f"\n{'='*60}")
        print(f"Processing: {base_name}")
        print(f"{'='*60}")
        
        try:
            pdf_to_images_with_split(pdf_path, temp_img_dir, split_dir, base_name, zoom=4, debug=False)
            print(f"✅ Step 1 complete for {base_name}")
        except Exception as e:
            print(f"❌ Error in step 1 for {base_name}: {e}")
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE! All PDFs converted and split.")
    print(f"Split images saved in: {TEMP_FOLDER}")
    print("="*60)


def step2_ocr_all_split_images():
    """
    Step 2: Run OCR on all split images
    Run this in another Jupyter cell
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all split directories
    split_dirs = sorted(glob.glob(os.path.join(TEMP_FOLDER, "tempo_split_*")))
    
    if not split_dirs:
        print(f"No split directories found in {TEMP_FOLDER}")
        return
    
    print(f"Found {len(split_dirs)} PDFs to OCR process.")
    
    for split_dir in tqdm(split_dirs, desc="Step 2: Running OCR"):
        # Extract PDF base name from directory name
        dir_name = os.path.basename(split_dir)
        pdf_base_name = dir_name.replace("tempo_split_", "")
        
        output_md_path = os.path.join(OUTPUT_FOLDER, pdf_base_name + ".md")
        
        if os.path.exists(output_md_path):
            print(f"Skipping {pdf_base_name} (already exists)")
            continue
        
        print(f"\n{'='*60}")
        print(f"OCR Processing: {pdf_base_name}")
        print(f"{'='*60}")
        
        try:
            ocr_split_images(split_dir, pdf_base_name, output_md_path)
            print(f"✅ Step 2 complete for {pdf_base_name}")
        except Exception as e:
            print(f"❌ Error in step 2 for {pdf_base_name}: {e}")
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(f"[OCR PROCESSING FAILED: {str(e)}]")
    
    print("\n" + "="*60)
    print("STEP 2 COMPLETE! All OCR processing done.")
    print(f"Output saved to: {OUTPUT_FOLDER}")
    print("="*60)

    
def main():

    step1_pdf_to_split_images_all()


if __name__ == "__main__":
    main()




SPLIT_PY
    
    # Replace NUM with actual number
    sed -i "s/NUM/$new_num/g" "$new_folder/split_code$new_num.py"
    
    # Create ocr_code PBS file
    cat > "$new_folder/ocr_code$new_num.pbs" << 'OCR_PBS'
#!/bin/bash
#PBS -N ocr_codeNUM
#PBS -l select=1:ncpus=20:mem=180gb:ngpus=1
#PBS -q gpu_1d
#PBS -o logsNUM/output.log
#PBS -e logsNUM/error.log

# === Load modules & activate conda ===
module load /app/common/modules/anaconda3-2024.10
source activate khalil_vllm

# === Move to working directory ===
cd /home/skiredj.abderrahman/khalil

# === Confirm node ===
echo "==== Job running on node: $(hostname -s) ===="


# === Start vLLM server ===
start_vllm() {
    echo "==== Starting vLLM server ===="
    vllm serve /home/skiredj.abderrahman/khalil/chandra \
      --max-model-len 122000 \
      --tensor-parallel-size 1 \
      --port 9996 >> logs/vllmNUM.log 2>&1
}

# === Run OCR processing ===
run_ocr() {
    echo "==== Starting OCR processing ===="
    module load /app/common/modules/anaconda3-2024.10
    source activate khalil_vllm
    cd /home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/batchs/batchNUM/
    python ocr_codeNUM.py
}

# === Launch vLLM in background ===
start_vllm &
VLLM_PID=$!

# === Wait until vLLM metrics endpoint is up ===
PORT=9996
echo "Waiting for vLLM server to be ready..."
until curl -s "http://localhost:${PORT}/metrics" >/dev/null 2>&1; do
  sleep 2
done
echo "vLLM server is ready!"

# === Optional: Add buffer time ===
sleep 10

# === Now run OCR processing ===
run_ocr

# === Wait until everything exits ===
exit 0

OCR_PBS
    
    # Replace NUM with actual number
    sed -i "s/NUM/$new_num/g" "$new_folder/ocr_code$new_num.pbs"
    
    # Create ocr_code Python file
    cat > "$new_folder/ocr_code$new_num.py" << 'OCR_PY'

# ============================================================================
# UNIFIED PDF OCR PIPELINE WITH COLUMN SPLITTING
# ============================================================================

import os
import time
import base64
import requests
import glob
import math
import shutil
import cv2
import numpy as np
from typing import Tuple, Union, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from tqdm import tqdm

# ========== CONFIGURATION ==========
# VLLM API Configuration
VLLM_API_URL = "http://localhost:9996/v1/chat/completions"
MODEL_PATH = "/home/skiredj.abderrahman/khalil/chandra"

# LLM Generation Parameters
GENERATION_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 30000,
    "top_p": 0.9,
}

# Request timeout
REQUEST_TIMEOUT_SECONDS = 180

# Parallel processing
N_WORKERS = 17

# OCR Prompt
OCR_PROMPT = """Extract all text exactly as it appears in this image.
Preserve the original layout, reading order, formatting, and structure.
Keep all tables as proper Markdown tables.
Maintain all mathematical formulas, equations, and special characters.
Do not rewrite, summarize, or modify any content.
If any text is unclear or unreadable, write [UNCLEAR].
Output only the extracted text without any preamble or explanation."""

# Folders
PDF_FOLDER = r"/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/batchs/batchNUM/pdf_documents_NUM"
PROCESSING_BASE = r"/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/batchs/batchNUM/processing_results_NUM"
OUTPUT_FOLDER = os.path.join(PROCESSING_BASE, "documents_transformed_to_markdown")
TEMP_FOLDER = os.path.join(PROCESSING_BASE, "tempo")

# ========== IMAGE PREPROCESSING FUNCTIONS (FROM CODE 2) ==========

def normalize_illumination(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, blur, scale=255)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

def enhance_contrast_and_sharpness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    return sharp

def upscale_opencv(img, scale=2):
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)
    return upscaled

def preprocess_page(img):
    img = normalize_illumination(img)
    img = enhance_contrast_and_sharpness(img)
    img = upscale_opencv(img, scale=2)
    return img

def _read_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Impossible de lire l'image : {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path.copy()
    else:
        raise TypeError("image_or_path doit être un chemin ou une image numpy.")
    img = preprocess_page(img)
    return img

# ========== TABLE DETECTION (FROM CODE 2) ==========

def is_big_table(img, debug=False, base_name=""):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = th if np.mean(th) > 127 else cv2.bitwise_not(th)
    bin_img = cv2.medianBlur(bin_img, 3)

    # Vertical lines
    vert_len = max(10, H // 20)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vertical = cv2.morphologyEx(255 - bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
    _, vertical = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY)

    table_big_min_height_ratio = 0.85
    table_many_verticals = 5
    table_min_spread_ratio = 0.5
    table_min_width_coverage = 0.06

    tall_verticals = []
    total_v_width = 0
    contours_vert, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours_vert:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(table_big_min_height_ratio * H) and (1 <= w <= int(0.03 * W)):
            tall_verticals.append((x, y, w, h))
            total_v_width += w

    spread_ok = False
    if len(tall_verticals) >= table_many_verticals:
        xs = np.array([x + w / 2.0 for x, _, w, _ in tall_verticals], dtype=float)
        width_spread = np.ptp(xs) if xs.size else 0.0
        if width_spread >= table_min_spread_ratio * W:
            spread_ok = True

    width_coverage = (total_v_width / float(W)) if W > 0 else 0.0
    looks_like_big_table = (
            (len(tall_verticals) >= table_many_verticals)
            and spread_ok
            and (width_coverage >= table_min_width_coverage)
    )

    # Horizontal lines
    horiz_len = max(10, W // 20)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horizontal = cv2.morphologyEx(255 - bin_img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horiz_kernel, iterations=1)
    _, horizontal = cv2.threshold(horizontal, 0, 255, cv2.THRESH_BINARY)
    contours_horiz, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_horizontal_lines = []
    medium_horizontal_lines = []

    for c in contours_horiz:
        x, y, w, h = cv2.boundingRect(c)
        line_width_ratio = w / W
        if line_width_ratio > 0.5:
            long_horizontal_lines.append((x, y, w, h))
        elif line_width_ratio > 0.2:
            medium_horizontal_lines.append((x, y, w, h))

    many_horizontal = len(long_horizontal_lines)
    medium_horizontal = len(medium_horizontal_lines)

    line_density = many_horizontal / (H / 100.0)

    if long_horizontal_lines:
        line_ys = [y + h / 2 for x, y, w, h in long_horizontal_lines]
        vertical_spread = (max(line_ys) - min(line_ys)) / H if len(line_ys) > 1 else 0
    else:
        vertical_spread = 0

    spacing_regularity = 0
    if len(long_horizontal_lines) >= 3:
        line_ys = sorted([y + h / 2 for x, y, w, h in long_horizontal_lines])
        spacings = [line_ys[i + 1] - line_ys[i] for i in range(len(line_ys) - 1)]
        if spacings:
            avg_spacing = np.mean(spacings)
            spacing_variance = np.var(spacings) / (avg_spacing + 1)
            spacing_regularity = 1.0 / (1.0 + spacing_variance)

    text_pixels = np.sum(255 - bin_img > 127)
    line_pixels = np.sum(horizontal > 127) + np.sum(vertical > 127)
    line_to_text_ratio = line_pixels / (text_pixels + 1)

    old_table_detection = many_horizontal >= 4

    dense_horizontal_table = (
            many_horizontal >= 5 and
            line_density >= 2.0 and
            vertical_spread >= 0.6
    )

    regular_table = (
            many_horizontal >= 4 and
            spacing_regularity >= 0.7 and
            vertical_spread >= 0.5
    )

    line_heavy_document = (
            many_horizontal >= 5 and
            line_to_text_ratio >= 0.25 and
            (medium_horizontal + many_horizontal) >= 12 and
            vertical_spread >= 0.4
    )

    table_detected = (
            looks_like_big_table or
            old_table_detection or
            dense_horizontal_table or
            regular_table or
            line_heavy_document
    )

    return table_detected

# ========== COLUMN SPLITTING FUNCTIONS (FROM CODE 2) ==========

def has_significant_text(img_region, min_text_ratio=0.01):
    if len(img_region.shape) == 3:
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_region
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_pixels = np.sum(binary > 0)
    total_pixels = binary.shape[0] * binary.shape[1]
    text_ratio = text_pixels / total_pixels
    return text_ratio > min_text_ratio

def find_text_density_split(img, debug=False):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    H, W = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 15), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_h_lines = sum(1 for c in h_contours if cv2.boundingRect(c)[2] > 0.7 * W)
    medium_h_lines = sum(1 for c in h_contours if 0.4 * W < cv2.boundingRect(c)[2] <= 0.7 * W)
    total_h_lines = long_h_lines + medium_h_lines

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, H // 20)))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_v_lines = sum(1 for c in v_contours if cv2.boundingRect(c)[3] > 0.6 * H)
    medium_v_lines = sum(1 for c in v_contours if 0.3 * H < cv2.boundingRect(c)[3] <= 0.6 * H)
    total_v_lines = long_v_lines + medium_v_lines

    if total_h_lines >= 2 and total_v_lines >= 2:
        grid_mask = cv2.bitwise_and(horizontal_lines, vertical_lines)
        intersection_contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        intersections = len(intersection_contours)
        intersections_per_area = intersections / ((W * H) / 10000)

        if intersections > 0:
            intersection_points = []
            for c in intersection_contours:
                x, y, w, h = cv2.boundingRect(c)
                intersection_points.append((x + w // 2, y + h // 2))

            if len(intersection_points) >= 4:
                distances = []
                for i, (x1, y1) in enumerate(intersection_points):
                    for j, (x2, y2) in enumerate(intersection_points[i + 1:], i + 1):
                        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        distances.append(dist)

                if distances:
                    dist_variance = np.var(distances)
                    dist_mean = np.mean(distances)
                    regularity_score = dist_variance / (dist_mean ** 2) if dist_mean > 0 else 0
                else:
                    regularity_score = 0
            else:
                regularity_score = 0
        else:
            regularity_score = 0
            intersections_per_area = 0

        if intersections >= 20 and regularity_score < 0.1 and intersections_per_area > 0.3:
            return None
        if total_h_lines >= 6 and intersections >= 15 and intersections_per_area > 0.2:
            return None
        if total_h_lines >= 4 and total_v_lines >= 4 and intersections >= 25:
            return None

    vertical_projection = np.sum(binary, axis=0) / 255
    window_size = max(1, int(W * 0.02))
    cand_points = []
    maxpv = vertical_projection.max() if vertical_projection.size else 0.0

    for x in range(window_size, W - window_size):
        a = max(0, x - window_size // 2)
        b = min(W, x + window_size // 2)
        window_density = np.mean(vertical_projection[a:b])
        if window_density < maxpv * 0.1:
            cand_points.append((x, window_density))

    if not cand_points:
        return None

    min_center = int(0.25 * W)
    max_center = int(0.75 * W)
    cand_points = [p for p in cand_points if min_center <= p[0] <= max_center]
    if not cand_points:
        return None

    center_x = W // 2
    best_split = min(cand_points, key=lambda p: abs(p[0] - center_x))[0]

    def text_ratio(region):
        if len(region.shape) == 3:
            rg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            rg = region
        _, b = cv2.threshold(rg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return (np.sum(b > 0) / float(b.size)) if b.size else 0.0

    left_ratio = text_ratio(img[:, :best_split])
    right_ratio = text_ratio(img[:, best_split:])

    min_text_ratio = 0.01
    balance_min_factor = 0.5

    if left_ratio < min_text_ratio or right_ratio < min_text_ratio:
        return None

    if min(left_ratio, right_ratio) < balance_min_factor * max(left_ratio, right_ratio):
        return None

    return best_split

def has_unique_central_vertical_rule(
    image_or_path: Union[str, np.ndarray],
    center_band_ratio: float = 0.20,
    min_height_ratio: float = 0.70,
    max_line_width_ratio: float = 0.03,
    min_line_width_px: int = 1,
    max_gap_ratio: float = 0.10,
    max_gap_runs: int = 2,
    gap_run_ratio: float = 0.01,
    max_candidates_allowed: int = 3,
    debug: bool = False,
    base_name: str = ""
) -> Tuple[bool, Dict]:
    bgr_orig = _read_image(image_or_path)
    orig_H, orig_W = bgr_orig.shape[:2]
    scale_factor = 1.0
    max_side = max(orig_H, orig_W)
    bgr = bgr_orig
    if max_side > 2000:
        scale_factor = 2000.0 / max_side
        bgr = cv2.resize(
            bgr_orig, (int(orig_W*scale_factor), int(orig_H*scale_factor)),
            interpolation=cv2.INTER_AREA
        )
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = th_otsu if np.mean(th_otsu) > 127 else cv2.bitwise_not(th_otsu)
    bin_img = cv2.medianBlur(bin_img, 3)
    vert_len = max(10, H // 20)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vertical = cv2.morphologyEx(255 - bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
    _, vertical = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY)

    table_big_min_height_ratio = 0.85
    table_many_verticals = 5
    table_min_spread_ratio = 0.50
    table_min_width_coverage = 0.06
    tall_verticals = []
    total_v_width = 0
    contours_all, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_all:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(table_big_min_height_ratio * H) and (min_line_width_px <= w <= int(max_line_width_ratio * W)):
            tall_verticals.append((x, y, w, h))
            total_v_width += w
    spread_ok = False
    if len(tall_verticals) >= table_many_verticals:
        xs = np.array([x + w/2.0 for x,_,w,_ in tall_verticals], dtype=float)
        width_spread = np.ptp(xs) if xs.size else 0.0
        if width_spread >= table_min_spread_ratio * W:
            spread_ok = True
    width_coverage = (total_v_width / float(W)) if W > 0 else 0.0
    looks_like_big_table = (len(tall_verticals) >= table_many_verticals) and spread_ok and (width_coverage >= table_min_width_coverage)

    mid = W // 2
    band_half = int(center_band_ratio * W)
    x0 = max(0, mid - band_half)
    x1 = min(W, mid + band_half)
    central_band = np.zeros_like(vertical)
    central_band[:, x0:x1] = vertical[:, x0:x1]
    contours, _ = cv2.findContours(central_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < int(min_height_ratio * H):
            continue
        if w < min_line_width_px or w > int(max_line_width_ratio * W):
            continue
        x_center = x + w / 2.0
        if not (mid - band_half <= x_center <= mid + band_half):
            continue
        roi = (255 - bin_img)[y:y+h, max(0, x):min(W, x+w)]
        row_has_ink = (np.max(roi, axis=1) > 0).astype(np.uint8)
        min_gap = max(3, int(gap_run_ratio * H))
        gaps, run = [], 0
        for val in row_has_ink:
            if val == 0:
                run += 1
            else:
                if run >= min_gap:
                    gaps.append(run)
                run = 0
        if run >= min_gap:
            gaps.append(run)
        total_gap = sum(gaps)
        if total_gap > max_gap_ratio * H or len(gaps) > max_gap_runs:
            continue
        candidates.append((x, y, w, h, {"gaps": gaps, "total_gap": total_gap}))

    central_tall_verticals = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= int(min_height_ratio * H) and (min_line_width_px <= w <= int(max_line_width_ratio * W)):
            central_tall_verticals += 1
    if central_tall_verticals > 1:
        return (False, {
            "original_image_size": (orig_H, orig_W),
            "processed_image_size": (H, W),
            "scale_factor": scale_factor,
            "center_band": (x0, x1),
            "candidates_found": len(candidates),
            "central_tall_verticals": central_tall_verticals,
            "table_blocked": True,
        })

    global_like = 0
    for c in contours_all:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(min_height_ratio * H) and (min_line_width_px <= w <= int(max_line_width_ratio * W)):
            global_like += 1

    details = {
        "original_image_size": (orig_H, orig_W),
        "processed_image_size": (H, W),
        "scale_factor": scale_factor,
        "center_band": (x0, x1),
        "candidates_found": len(candidates),
        "global_vertical_candidates": global_like,
        "candidates": [{"bbox": (int(x), int(y), int(w), int(h)), **stats}
                       for (x, y, w, h, stats) in candidates],
        "big_table_heuristic": {
            "looks_like_big_table": bool(looks_like_big_table),
            "very_tall_verticals": int(len(tall_verticals)),
            "spread_ok": bool(spread_ok),
            "width_coverage": float(width_coverage)
        }
    }
    
    if global_like > max_candidates_allowed:
        return (False, details)
    if looks_like_big_table:
        return (False, details)
    is_true = (len(candidates) == 1)
    return (is_true, details)

def get_split_x_from_details(details: Dict) -> Optional[int]:
    if details.get("candidates_found", 0) != 1:
        return None
    (x, y, w, h) = details["candidates"][0]["bbox"]
    split_x_processed = int(round(x + w / 2))
    scale = float(details.get("scale_factor", 1.0))
    if scale <= 0:
        scale = 1.0
    split_x_original = int(round(split_x_processed / scale))
    orig_W = details.get("original_image_size", (0, 0))[1]
    if orig_W:
        split_x_original = max(1, min(orig_W - 1, split_x_original))
    return split_x_original

def _derive_names(image_or_path: Union[str, np.ndarray],
                  base_name: Optional[str]) -> Tuple[str, str]:
    if isinstance(image_or_path, str):
        stem, ext = os.path.splitext(os.path.basename(image_or_path))
        if ext == "":
            ext = ".png"
        return stem, ext
    else:
        if not base_name:
            raise ValueError("For ndarray input, please provide base_name.")
        stem, ext = os.path.splitext(base_name)
        if ext == "":
            ext = ".png"
        return stem, ext

def perform_split(img, split_x, dest_dir, stem, ext, left_suffix, right_suffix, details):
    os.makedirs(dest_dir, exist_ok=True)
    left_img = img[:, :split_x]
    right_img = img[:, split_x:]
    left_path = os.path.join(dest_dir, f"{stem}{left_suffix}{ext}")
    right_path = os.path.join(dest_dir, f"{stem}{right_suffix}{ext}")
    ok_left = cv2.imwrite(left_path, left_img)
    ok_right = cv2.imwrite(right_path, right_img)
    if ok_left and ok_right:
        return {
            "split": True,
            "left_path": left_path,
            "right_path": right_path,
            "split_x": split_x,
            "details": details,
        }
    else:
        out_path = os.path.join(dest_dir, f"{stem}{ext}")
        cv2.imwrite(out_path, img)
        return {
            "split": False,
            "reason": "write_error",
            "copied_path": out_path
        }

def enhanced_split_for_administrative_docs(
        image_or_path: Union[str, np.ndarray],
        dest_dir: str,
        base_name: Optional[str] = None,
        left_suffix: str = "_left",
        right_suffix: str = "_right",
        debug=False
) -> Dict:
    stem, ext = _derive_names(image_or_path, base_name)
    img = _read_image(image_or_path)
    orig_H, orig_W = img.shape[:2]

    is_two_col, details = has_unique_central_vertical_rule(
        image_or_path, debug=debug, base_name=stem)

    if is_two_col:
        split_x = get_split_x_from_details(details)
        if split_x is not None:
            split_x = max(1, min(orig_W - 1, int(split_x)))
            left_has_text = has_significant_text(img[:, :split_x])
            right_has_text = has_significant_text(img[:, split_x:])

            if left_has_text and right_has_text:
                return perform_split(img, split_x, dest_dir, stem, ext, left_suffix, right_suffix, details)

    alternative_split_x = find_text_density_split(img, debug=debug)
    if alternative_split_x is not None:
        left_has_text = has_significant_text(img[:, :alternative_split_x])
        right_has_text = has_significant_text(img[:, alternative_split_x:])

        if left_has_text and right_has_text:
            return perform_split(img, alternative_split_x, dest_dir, stem, ext, left_suffix, right_suffix,
                                 {"method": "text_density", "split_x": alternative_split_x})

    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"{stem}{ext}")
    if isinstance(image_or_path, str) and os.path.isfile(image_or_path):
        shutil.copy2(image_or_path, out_path)
    else:
        cv2.imwrite(out_path, img)

    return {
        "split": False,
        "reason": "no_valid_split_both_sides",
        "copied_path": out_path
    }

def split_if_two_column(
        image_or_path,
        dest_dir,
        base_name=None,
        debug=False
):
    img = _read_image(image_or_path)
    stem, ext = _derive_names(image_or_path, base_name)

    if is_big_table(img, debug=debug, base_name=stem):
        if debug:
            print("🛑 Table détectée : aucun split.")
        out_path = os.path.join(dest_dir, f"{stem}{ext}")
        if isinstance(image_or_path, str) and os.path.isfile(image_or_path):
            shutil.copy2(image_or_path, out_path)
        else:
            cv2.imwrite(out_path, img)
        return {
            "split": False,
            "reason": "big_table_detected",
            "copied_path": out_path
        }

    return enhanced_split_for_administrative_docs(
        image_or_path, dest_dir, base_name=base_name, debug=debug
    )

# ========== OCR FUNCTIONS (FROM CODE 1) ==========

def call_vllm_ocr(image_path, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                "model": MODEL_PATH,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT
                            }
                        ]
                    }
                ],
                **GENERATION_PARAMS
            }
            
            response = requests.post(
                VLLM_API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and result['choices']:
                extracted_text = result['choices'][0]['message']['content']
                return extracted_text
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except requests.exceptions.Timeout:
            attempt += 1
            print(f"Timeout on attempt {attempt}/{retries} for {image_path}")
            if attempt < retries:
                time.sleep(5)
            else:
                raise Exception(f"Request timed out after {retries} attempts")
                
        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"Request error on attempt {attempt}/{retries} for {image_path}: {e}")
            if attempt < retries:
                time.sleep(3)
            else:
                raise Exception(f"Request failed after {retries} attempts: {e}")
                
        except Exception as e:
            attempt += 1
            print(f"Error on attempt {attempt}/{retries} for {image_path}: {e}")
            if attempt < retries:
                time.sleep(3)
            else:
                raise

def process_image(input_image_path, output_text_path, retries=3):
    if os.path.exists(output_text_path):
        print(f"Skipping already processed file: {output_text_path}")
        return

    try:
        print(f"Processing image: {input_image_path}")
        extracted_text = call_vllm_ocr(input_image_path, retries=retries)
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"Finished processing: {input_image_path}")
        
    except Exception as e:
        print(f"Failed to process {input_image_path}: {e}")
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(f"[OCR FAILED: {str(e)}]")

def clean_unnecessary_linebreaks(text):
    lines = text.split('\n')
    cleaned_text = []

    for i in range(len(lines)):
        if lines[i].startswith("=== Page"):
            cleaned_text.append(lines[i])
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                cleaned_text.append("")
        else:
            if (i > 0 and
                not lines[i - 1].startswith("=== Page") and
                not lines[i - 1].rstrip().endswith('.')):
                if cleaned_text and cleaned_text[-1] != "":
                    cleaned_text[-1] = cleaned_text[-1].rstrip() + " " + lines[i].lstrip()
                else:
                    cleaned_text.append(lines[i])
            else:
                cleaned_text.append(lines[i])

    return '\n'.join(cleaned_text)

# ========== MAIN UNIFIED PIPELINE ==========

def pdf_to_images_with_split(pdf_path, temp_img_dir, split_dir, pdf_base_name, zoom=4, debug=False):
    """
    Convert PDF to images and apply column splitting
    """
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    
    # Step 1: Convert PDF to images
    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_filename = f"{pdf_base_name}_{page_number + 1}.png"
        image_path = os.path.join(temp_img_dir, image_filename)
        pix.save(image_path)
    pdf_document.close()
    
    # Step 2: Apply column splitting to each image
    image_files = sorted(glob.glob(os.path.join(temp_img_dir, "*.png")))
    for img_path in image_files:
        result = split_if_two_column(img_path, split_dir, debug=debug)
        if debug:
            base_name = os.path.basename(img_path)
            print(f"[{base_name}] Split result: {result.get('split')}")

def ocr_split_images(split_dir, pdf_base_name, output_md_path):
    """
    Run OCR on split images and generate markdown output
    """
    txt_dir = os.path.join(TEMP_FOLDER, f"tempo_res_{pdf_base_name}")
    os.makedirs(txt_dir, exist_ok=True)
    
    # Get all images from split directory
    # Get all images from split directory
    image_files = glob.glob(os.path.join(split_dir, "*.png"))
    
    # Custom sort: for Arabic, process RIGHT pages before LEFT pages
    def arabic_sort_key(path):
        filename = os.path.basename(path)
        
        # Extract page number using regex
        import re
        
        # Remove the extension first
        name_without_ext = filename.replace('.png', '')
        
        # Check for _right or _left suffix
        if name_without_ext.endswith('_right'):
            base_name = name_without_ext[:-6]  # Remove '_right'
            suffix_order = 0  # Right comes first
        elif name_without_ext.endswith('_left'):
            base_name = name_without_ext[:-5]  # Remove '_left'
            suffix_order = 1  # Left comes second
        else:
            base_name = name_without_ext
            suffix_order = 0  # No split, treat like right
        
        # ✅ FIXED: Extract ALL numbers, get the last one (handles -bis, -ter, etc.)
        numbers = re.findall(r'\d+', base_name)
        
        if numbers:
            page_num = int(numbers[-1])  # Get the LAST number found
        else:
            page_num = 0
        
        return (page_num, suffix_order)
    
    image_files = sorted(image_files, key=arabic_sort_key)
    
    if not image_files:
        print(f"No images found in {split_dir}")
        return
    
    # Prepare OCR tasks
    tasks = []
    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_text_path = os.path.join(txt_dir, f"{img_name}.txt")
        tasks.append((img_path, output_text_path))
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_image, in_path, out_path): (in_path, out_path) 
                  for in_path, out_path in tasks}
        for future in as_completed(futures):
            in_path, out_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Unhandled error processing {in_path}: {e}")
    
    # Combine all page texts
    combined_text = []
    for i, img_path in enumerate(image_files, 1):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        text_path = os.path.join(txt_dir, f"{img_name}.txt")
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                page_content = f.read()
                combined_text.append(f"=== Page {i} ({img_name}) ===\n{page_content}\n")
        else:
            combined_text.append(f"=== Page {i} ({img_name}) ===\n[OCR failed for this page]\n")
    
    raw_text = "\n".join(combined_text)
    cleaned_text = clean_unnecessary_linebreaks(raw_text)
    
    # Save output
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    print(f"Saved OCR output to {output_md_path}")

def step1_pdf_to_split_images_all():
    """
    Step 1: Convert all PDFs to images and apply splitting
    Run this in one Jupyter cell
    """
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    
    pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
        return
    
    print(f"Found {len(pdf_files)} PDFs to process.")
    
    for pdf_path in tqdm(pdf_files, desc="Step 1: Converting & Splitting PDFs"):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_img_dir = os.path.join(TEMP_FOLDER, f"tempo_{base_name}")
        split_dir = os.path.join(TEMP_FOLDER, f"tempo_split_{base_name}")
        
        print(f"\n{'='*60}")
        print(f"Processing: {base_name}")
        print(f"{'='*60}")
        
        try:
            pdf_to_images_with_split(pdf_path, temp_img_dir, split_dir, base_name, zoom=4, debug=False)
            print(f"✅ Step 1 complete for {base_name}")
        except Exception as e:
            print(f"❌ Error in step 1 for {base_name}: {e}")
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE! All PDFs converted and split.")
    print(f"Split images saved in: {TEMP_FOLDER}")
    print("="*60)


def step2_ocr_all_split_images():
    """
    Step 2: Run OCR on all split images
    Run this in another Jupyter cell
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all split directories
    split_dirs = sorted(glob.glob(os.path.join(TEMP_FOLDER, "tempo_split_*")))
    
    if not split_dirs:
        print(f"No split directories found in {TEMP_FOLDER}")
        return
    
    print(f"Found {len(split_dirs)} PDFs to OCR process.")
    
    for split_dir in tqdm(split_dirs, desc="Step 2: Running OCR"):
        # Extract PDF base name from directory name
        dir_name = os.path.basename(split_dir)
        pdf_base_name = dir_name.replace("tempo_split_", "")
        
        output_md_path = os.path.join(OUTPUT_FOLDER, pdf_base_name + ".md")
        
        if os.path.exists(output_md_path):
            print(f"Skipping {pdf_base_name} (already exists)")
            continue
        
        print(f"\n{'='*60}")
        print(f"OCR Processing: {pdf_base_name}")
        print(f"{'='*60}")
        
        try:
            ocr_split_images(split_dir, pdf_base_name, output_md_path)
            print(f"✅ Step 2 complete for {pdf_base_name}")
        except Exception as e:
            print(f"❌ Error in step 2 for {pdf_base_name}: {e}")
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(f"[OCR PROCESSING FAILED: {str(e)}]")
    
    print("\n" + "="*60)
    print("STEP 2 COMPLETE! All OCR processing done.")
    print(f"Output saved to: {OUTPUT_FOLDER}")
    print("="*60)
 


def main():

    step2_ocr_all_split_images()


if __name__ == "__main__":
    main()


OCR_PY
    
    # Replace NUM with actual number
    sed -i "s/NUM/$new_num/g" "$new_folder/ocr_code$new_num.py"
    
    echo "  ├─ Created files: split_code$new_num.pbs, split_code$new_num.py, ocr_code$new_num.pbs, ocr_code$new_num.py"
    
    # Move 20 PDFs from all_pdf_documents_test to this batch folder
    echo "  └─ Moving 15 PDFs to pdf_documents_$new_num..."
    find "$BASE_PATH/all_pdf_documents" -type f 2>/dev/null | sort | head -n 15 | xargs -I {} mv "{}" "$BATCHS_PATH/$new_folder/pdf_documents_$new_num/"
    
done

echo ""
echo "Done! Created folders from batch$((latest_num + 1)) to batch$((latest_num + num_folders))"
echo ""
echo "=== PDF Document Count per Folder ==="
for ((i=1; i<=num_folders; i++)); do
    new_num=$((latest_num + i))
    count=$(ls -1 "batch$new_num/pdf_documents_$new_num/" 2>/dev/null | wc -l)
    echo "batch$new_num: $count files"
done
