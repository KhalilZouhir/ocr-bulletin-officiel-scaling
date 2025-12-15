#!/bin/bash

# Base path
BASE_PATH="/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/deepseek_post_ocr"

# Check if the deepseek_post_ocr directory exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Directory '$BASE_PATH' does not exist!"
    exit 1
fi

# Navigate to the directory
cd "$BASE_PATH" || exit 1

# Find the latest ds folder (only purely numeric ones)
latest_folder=$(find . -maxdepth 1 -type d -name 'ds[0-9]*' | sed 's|./ds||' | grep -E '^[0-9]+$' | sort -n | tail -1)

if [ -z "$latest_folder" ]; then
    echo "No 'ds' folders found in $BASE_PATH"
    echo "Starting from ds1..."
    latest_num=0
else
    latest_num=$latest_folder
    echo "The latest folder is: ds$latest_num"
fi

# Ask user how many new folders to create
read -p "How many new folders do you want to create? " num_folders

# Validate input
if ! [[ "$num_folders" =~ ^[0-9]+$ ]] || [ "$num_folders" -le 0 ]; then
    echo "Error: Please enter a valid positive number!"
    exit 1
fi

# Create new folders with files
echo "Creating $num_folders new folder(s)..."
for ((i=1; i<=num_folders; i++)); do
    new_num=$((latest_num + i))
    new_folder="ds$new_num"
    
    # Create main folder
    mkdir "$new_folder"
    echo "Created: $new_folder"
    
    # Create batchs subdirectory
    mkdir -p "$new_folder/batchs"
    mkdir -p "$new_folder/logs"
    echo "  ├─ Created subdirectory: batchs and logs"
    
    # Create ds PBS file
    cat > "$new_folder/ds$new_num.pbs" << 'DS_PBS'
#!/bin/bash
#PBS -N ds_NUM
#PBS -l select=1:ncpus=20:mem=180gb:ngpus=1
#PBS -q gpu_1d
#PBS -o logs/split_output_NUM.log
#PBS -e logs/split_error_NUM.log

# === Load modules & activate conda ===
module use /app/common/modules
module load anaconda3-2024.10
source activate  deepseek_khalil


# === Move to working directory ===
cd /home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/deepseek_post_ocr/dsNUM
python dsNUM.py
DS_PBS
    
    # Replace NUM with actual number
    sed -i "s/NUM/$new_num/g" "$new_folder/ds$new_num.pbs"
    
    # Create ds Python file
    cat > "$new_folder/ds$new_num.py" << 'DS_PY'

"""
DeepSeek OCR Cleanup Script
============================
Processes failed OCR pages from Chandra using DeepSeek model.
Processes one batch at a time in sorted order.
Converts DeepSeek markdown output to HTML to match Chandra's output format.
"""
import shutil  
import os
import re
import glob
import torch
import time
import markdown
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import shutil

# ========== CONFIGURATION ==========
BATCHES_ROOT = "/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/deepseek_post_ocr/dsKIKA/batchs"  # ⚠️ UPDATE THIS PATH
DEEPSEEK_MODEL_NAME = "/home/skiredj.abderrahman/khalil/DeepSeek-OCR"
CUDA_DEVICE = "0"
DEEPSEEK_TIMEOUT_SECONDS = 180  # Timeout per image (3 minutes)
BATCH_REST_SECONDS = 120  # Rest 2 minutes between batches

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# DeepSeek prompt
DEEPSEEK_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


def load_deepseek_model():
    """
    Load DeepSeek OCR model once per batch
    """
    print("Loading DeepSeek model...")
    tokenizer = AutoTokenizer.from_pretrained(
        DEEPSEEK_MODEL_NAME, 
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        DEEPSEEK_MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    print("✅ DeepSeek model loaded successfully!")
    return model, tokenizer


def markdown_to_html(markdown_text):
    """
    Convert markdown text to HTML to match Chandra's output format
    """
    try:
        html_output = markdown.markdown(
            markdown_text,
            extensions=['tables', 'fenced_code', 'nl2br']
        )
        return html_output
    except Exception as e:
        print(f"  ⚠️  Markdown to HTML conversion failed: {e}")
        # Return original text wrapped in basic HTML if conversion fails
        return f"<div>{markdown_text}</div>"


def scan_markdown_for_failures(markdown_path):
    """
    Scan a markdown file and extract all failed OCR pages.
    Returns list of tuples: [(page_number, page_identifier), ...]
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    failures = []
    for i, line in enumerate(lines):
        # Look for OCR failure marker
        if "[OCR FAILED: Request timed out after 3 attempts]" in line:
            # Look back to find the page header
            if i > 0:
                prev_line = lines[i - 1]
                # Extract page identifier from: === Page X (BO_XXXXX_Ar_YY_left) ===
                match = re.search(r'=== Page \d+ \((.+?)\) ===', prev_line)
                if match:
                    page_identifier = match.group(1)
                    failures.append((i, page_identifier))
    
    return failures


def get_pdf_base_name(page_identifier):
    """
    Extract PDF base name from page identifier.
    Examples:
      - BO_5705_Ar_23_left → BO_5705_Ar
      - BO_5705_Ar_23 → BO_5705_Ar
      - BO_5705-bis_Ar_23_right → BO_5705-bis_Ar
    """
    # Remove _left or _right suffix if present
    if page_identifier.endswith('_left'):
        base = page_identifier[:-5]
    elif page_identifier.endswith('_right'):
        base = page_identifier[:-6]
    else:
        base = page_identifier
    
    # Remove the page number (last _XX)
    parts = base.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def run_deepseek_ocr(model, tokenizer, image_path, output_dir, timeout=180):
    """
    Run DeepSeek OCR on a single image with timeout
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
    def _run_inference():
        return model.infer(
            tokenizer,
            prompt=DEEPSEEK_PROMPT,
            image_file=image_path,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True
        )
    
    try:
        print(f"  Running DeepSeek on: {os.path.basename(image_path)}")
        
        # Run with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_inference)
            try:
                res = future.result(timeout=timeout)
                return res
            except TimeoutError:
                print(f"  ⏱️  DeepSeek timed out after {timeout} seconds")
                return None
        
    except Exception as e:
        print(f"  ❌ DeepSeek failed: {e}")
        return None


def update_markdown_file(markdown_path, deepseek_results):
    """
    Update markdown file by replacing [OCR FAILED: ...] with DeepSeek results
    deepseek_results: dict mapping page_identifier → ocr_text
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a failure line
        if "[OCR FAILED: Request timed out after 3 attempts]" in line:
            # Extract page identifier from previous line
            if i > 0:
                prev_line = lines[i - 1]
                match = re.search(r'=== Page \d+ \((.+?)\) ===', prev_line)
                if match:
                    page_identifier = match.group(1)
                    
                    # Replace with DeepSeek result if available
                    if page_identifier in deepseek_results:
                        if deepseek_results[page_identifier] is not None:
                            # Successful DeepSeek OCR
                            updated_lines.append(deepseek_results[page_identifier] + '\n')
                        else:
                            # DeepSeek also failed
                            updated_lines.append("[OCR FAILED WITH DEEPSEEK: Could not extract text]\n")
                    else:
                        # Should not happen, but keep original if no result
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
        
        i += 1
    
    # Write updated content back
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"  ✅ Updated: {os.path.basename(markdown_path)}")


def process_batch(batch_path):
    """
    Process a single batch folder
    """
    batch_name = os.path.basename(batch_path)
    print(f"\n{'='*70}")
    print(f"Processing Batch: {batch_name}")
    print(f"{'='*70}")
    
    # Paths - handle numbered folder suffixes
    batch_num = re.search(r'batch(\d+)', os.path.basename(batch_path)).group(1)
    processing_results = os.path.join(batch_path, f"processing_results_{batch_num}")
    markdown_dir = os.path.join(processing_results, "documents_transformed_to_markdown")
    tempo_dir = os.path.join(processing_results, "tempo")
    
    if not os.path.exists(markdown_dir):
        print(f"❌ No markdown directory found in {batch_name}, skipping...")
        return
    
    # Step 1: Scan all markdowns for failures
    print("\nStep 1: Scanning markdowns for failed OCR pages...")
    all_failures = {}  # markdown_path → [(line_num, page_identifier), ...]
    
    markdown_files = glob.glob(os.path.join(markdown_dir, "*.md"))
    if not markdown_files:
        print(f"No markdown files found in {batch_name}")
        return
    
    for md_path in markdown_files:
        failures = scan_markdown_for_failures(md_path)
        if failures:
            all_failures[md_path] = failures
            print(f"  Found {len(failures)} failures in {os.path.basename(md_path)}")
    
    if not all_failures:
        print(f"✅ No failures found in {batch_name}, skipping...")
        return
    
    total_failures = sum(len(f) for f in all_failures.values())
    print(f"\nTotal failures to process: {total_failures}")
    
    # Step 2: Load DeepSeek model
    print("\nStep 2: Loading DeepSeek model...")
    model, tokenizer = load_deepseek_model()
    
    # Step 3: Process each failed page
    print("\nStep 3: Running DeepSeek OCR on failed pages...")
    
    for md_path, failures in all_failures.items():
        md_basename = os.path.basename(md_path)
        print(f"\nProcessing failures from: {md_basename}")
        
        deepseek_results = {}  # page_identifier → ocr_text
        
        for line_num, page_identifier in tqdm(failures, desc=f"  {md_basename}"):
            # Get PDF base name
            pdf_base = get_pdf_base_name(page_identifier)
            
            # Build image path
            split_folder = os.path.join(tempo_dir, f"tempo_split_{pdf_base}")
            image_path = os.path.join(split_folder, f"{page_identifier}.png")
            
            if not os.path.exists(image_path):
                print(f"  ⚠️  Image not found: {image_path}")
                deepseek_results[page_identifier] = None
                continue
            
            # Create output directory for DeepSeek
            output_dir = os.path.join(tempo_dir, f"tempo_res2_{pdf_base}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run DeepSeek OCR
            result = run_deepseek_ocr(model, tokenizer, image_path, output_dir, timeout=DEEPSEEK_TIMEOUT_SECONDS)
            
            # Read result from result.mmd file (DeepSeek saves here)
            result_mmd_path = os.path.join(output_dir, "result.mmd")
            if os.path.exists(result_mmd_path):
                try:
                    with open(result_mmd_path, 'r', encoding='utf-8') as f:
                        markdown_text = f.read()
                    
                    # Convert markdown to HTML to match Chandra's output format
                    html_text = markdown_to_html(markdown_text)
                    
                    # Save both markdown and HTML for tracking
                    output_text_path = os.path.join(output_dir, f"{page_identifier}.txt")
                    with open(output_text_path, 'w', encoding='utf-8') as f:
                        f.write(html_text)
                    
                    deepseek_results[page_identifier] = html_text
                    print(f"    ✅ Success! Converted to HTML ({len(html_text)} characters)")
                except Exception as e:
                    print(f"    ⚠️  Could not process result.mmd: {e}")
                    deepseek_results[page_identifier] = None
            else:
                print(f"    ❌ result.mmd not found in {output_dir}")
                deepseek_results[page_identifier] = None
        
        # Step 4: Update markdown file
        print(f"\nStep 4: Updating markdown file: {md_basename}")
        update_markdown_file(md_path, deepseek_results)
    
    # Cleanup
    print("\nCleaning up GPU memory...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print(f"\nDeleting tempo folder for {batch_name}...")
    if os.path.exists(tempo_dir):
        shutil.rmtree(tempo_dir)
        print(f"✅ Deleted: {tempo_dir}")
    print(f"\n✅ Batch {batch_name} processing complete!")


def main():
    """
    Main function to process batches one at a time in sorted order
    """
    if not os.path.exists(BATCHES_ROOT):
        print(f"❌ Batches root directory not found: {BATCHES_ROOT}")
        return
    
    # Get all batch directories
    batch_dirs = [d for d in glob.glob(os.path.join(BATCHES_ROOT, "batch*")) 
                  if os.path.isdir(d)]
    
    if not batch_dirs:
        print(f"❌ No batch directories found in {BATCHES_ROOT}")
        return
    
    # Sort batch directories
    batch_dirs = sorted(batch_dirs, key=lambda x: int(re.search(r'batch(\d+)', x).group(1)))
    
    print(f"Found {len(batch_dirs)} batches to process:")
    for bd in batch_dirs:
        print(f"  - {os.path.basename(bd)}")
    
    # Process each batch
    for idx, batch_path in enumerate(batch_dirs):
        try:
            process_batch(batch_path)
            
            # Rest between batches (except after the last one)
            if idx < len(batch_dirs) - 1:
                print(f"\n{'='*70}")
                print(f"💤 Resting GPU for {BATCH_REST_SECONDS} seconds before next batch...")
                print(f"{'='*70}")
                time.sleep(BATCH_REST_SECONDS)
                
        except Exception as e:
            batch_name = os.path.basename(batch_path)
            print(f"\n❌ Error processing {batch_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuing to next batch...\n")
            continue
    
    print("\n" + "="*70)
    print("ALL BATCHES PROCESSED!")
    print("="*70)

if __name__ == "__main__":
    
    BATCHES_ROOT = "/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/deepseek_post_ocr/dsKIKA/batchs"  # ⚠️ UPDATE THIS PATH
    DEEPSEEK_MODEL_NAME = "/home/skiredj.abderrahman/khalil/DeepSeek-OCR"
    
    main()
DS_PY
    
    # Replace NUM with actual number
    sed -i "s/KIKA/$new_num/g" "$new_folder/ds$new_num.py"
    
    echo "  └─ Created files: ds$new_num.pbs, ds$new_num.py"
    
done

echo ""
echo "Done! Created folders from ds$((latest_num + 1)) to ds$((latest_num + num_folders))"
