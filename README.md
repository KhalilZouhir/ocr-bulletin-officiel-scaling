# OCR Scaling for *Bulletin Officiel* using Chandra + DeepSeek (INPT HPC)

## Overview

This repository documents the **full, end‑to‑end workflow** to run large‑scale OCR on "Bulletins Officiels" PDFs using **INPT HPC** infrastructure.

The pipeline is **two‑stage**:

1. **Primary OCR with Chandra** (vLLM‑based)
2. **Post‑OCR recovery with DeepSeek‑OCR** for pages that failed during Chandra OCR

The design goal is **scalability, modularity, and efficient GPU utilization**, achieved by splitting PDFs into batches and processing each batch as an independent PBS job.

---

## Repository Structure (High Level)

- Two separate **Conda virtual environments**
- Two OCR engines (Chandra + DeepSeek)
- Automated batch creation and job submission
- GPU‑parallel execution on INPT HPC

---

## Virtual Environments & Requirements

Two environments are required to avoid dependency conflicts.
> ⚠️ **IMPORTANT**
> the 2 VENVs already exists do NOT MAKE ANY CHANGES ON THESE 2 VENVS plz 

### 1. Chandra Environment

- **VENV name:** `khalil_vllm`
- **Requirements file:** `khalil_vllm_packages.txt`

```bash
conda create -n khalil_vllm python=3.10 -y
conda activate khalil_vllm
pip install -r khalil_vllm_packages.txt
```
- or
```bash
# 1) CUDA toolkit (keep as is)
conda install -y -c nvidia -c conda-forge cuda-toolkit=12.4 ninja cmake

# 2) PyTorch (keep as is)
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3) Install latest transformers FIRST
pip install --upgrade transformers

# 4) Install vLLM (latest version for Qwen3-VL support)
pip install --upgrade vllm

# 5) Other dependencies
pip install accelerate cffi lazy-loader packaging soxr
```
  
This environment is used **only for Chandra OCR (vLLM runtime)**.

---

### 2. DeepSeek Post‑OCR Environment

- **VENV name:** `deepseek_khalil`
- **Requirements file:** `deepseek_khalil_packages.txt`

```bash
conda create -n deepseek_khalil python=3.10 -y
conda activate deepseek_khalil
pip install -r deepseek_khalil_packages.txt
```
-or

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r ds_requirements.txt
```

This environment is used **only for post‑processing failed OCR pages**.

---

## Model & Code Cloning

### Chandra

Clone Chandra locally using Hugging Face CLI:

```bash
huggingface-cli download datalab-to/chandra --local-dir ./chandra
```

**Local path used in this setup:**

```text
/home/skiredj.abderrahman/khalil/chandra
```

---

### DeepSeek‑OCR

```bash
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir ./DeepSeek‑OCR
```

**Local path used in this setup:**

```text
/home/skiredj.abderrahman/khalil/DeepSeek-OCR
```

---

## Logs Directory

Create a centralized logs directory for vLLM execution:

```bash
mkdir -p /home/skiredj.abderrahman/khalil/logs
```

---

## Main Working Directory (Racine)

All OCR operations happen inside a single root directory:

```text
/home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel
```

### Inside this directory:

```bash
mkdir batchs
mkdir all_pdf_documents
```

### Purpose

- **`all_pdf_documents/`**
  - Contains *all* Bulletin Officiel PDFs to be processed
  - PDFs can be imported manually or programmatically

- **`batchs/`**
  - Each batch represents an **independent PBS job**
  - Enables modular processing and GPU parallelization

---

## Batch Automation (Chandra Stage)

Copy the following script into the root directory:

```text
automation.sh
```
modify the file access permissions

```text
chmod +x automation.sh
```

### What `automation.sh` does

- Splits PDFs into batches
- Creates batch folders automatically
- Generates:
  - Split scripts
  - OCR scripts
  - PBS job files
  - Required directory structure
  - 
<img width="790" height="594" alt="image" src="https://github.com/user-attachments/assets/2cddcfc1-5880-4600-a0d3-a4a760938620" />

---

### Required Configuration (IMPORTANT)

Before execution, **update paths** inside `automation.sh`:

- `SPLIT_PBS`
- `OCR_PBS`
- `SPLIT_PY`
- `OCR_PY`

Adjust them to match your local filesystem layout.

---

### Batch Size Configuration

Locate **line ~2004** in `automation.sh`:

```bash
head -n 15
```

Change `15` to the desired number of PDFs per batch.

**Recommended:**

- **15 PDFs per batch** (based on empirical GPU/memory stability)

---

## Running Chandra OCR

From the root directory:

```bash
./automation.sh
```

This creates multiple batch folders inside:

```text
batchs/batch1
batchs/batch2
...
```

### For each batch:

```bash
cd batchs/batchN
qsub split_codeN.pbs
```

After split jobs finish:

```bash
qsub ocr_codeN.pbs
```

**Important:**

- Do **NOT** delete temporary folders after OCR
- They are required for DeepSeek post‑processing

---

## DeepSeek Post‑OCR Stage

### Setup Directory

Create the DeepSeek working directory:

```bash
mkdir /home/skiredj.abderrahman/khalil/OCR_scaling_bulletin_officiel/deepseek_post_ocr
```

Copy the script:

```text
automation_deepseek.sh
```
modify the file access permissions

```text
chmod +x automation_deepseek.sh
```
---

### Running DeepSeek Automation

From `deepseek_post_ocr`:

```bash
./automation_deepseek.sh
```

This creates multiple directories:

```text
ds1
ds2
...
```

Each `dsX` folder corresponds to one DeepSeek GPU job.

---

## Moving Batches to DeepSeek Jobs

Move Chandra batches into DeepSeek folders:

```bash
mv batch1 ../deepseek_post_ocr/ds1/batchs
```

### Best Practice

- You can assign more than one batch per DS folder, but it is recommended to assign **one batch per DS folder**
- Maximizes GPU utilization

---

## Running DeepSeek OCR

Inside each DS folder:

```bash
cd ds1
qsub ds1.pbs
```

### Behavior

- Processes failed OCR pages only
- Automatically cleans all temporary files after completion

---

## Final Notes

- This pipeline is optimized for **HPC batch scheduling**
- Separation of concerns (Chandra vs DeepSeek) avoids dependency conflicts
- Modular batching allows horizontal GPU scaling

This setup is production‑ready for large Bulletin Officiel OCR workloads on INPT HPC.

