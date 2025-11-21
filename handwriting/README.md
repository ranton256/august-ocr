# Handwriting OCR with Transformer Models

This subdirectory contains an exploration of modern transformer-based OCR approaches for handwriting recognition, building on the printed-text OCR techniques demonstrated in the main project.

## Overview

While the main project focuses on traditional OCR for printed historical documents (the August Anton memoir), this companion project explores state-of-the-art transformer models specifically designed for handwriting recognition:

1. **TrOCR** - Microsoft's transformer-based OCR model
2. **DeepSeek-OCR** - Advanced OCR with 4-bit quantization for efficient inference

These approaches are particularly well-suited for:
- Photos of handwritten notes taken with phone cameras
- Cursive and printed handwriting
- Varied lighting conditions and angles
- Personal note-taking digitization

## Context

This work builds on the foundational OCR concepts from the main printed-text project, adapting them for the unique challenges of handwriting recognition. The core pipeline remains similar (preprocessing → extraction → optional AI correction), but uses transformer models instead of traditional OCR engines.

## Available Models

### 1. TrOCR (Microsoft)

**Script:** `handwriting_ocr.py`

A transformer-based OCR model using an encoder-decoder architecture. Pre-trained on both printed and handwritten text datasets.

**Features:**
- Works on CPU or GPU (GPU recommended)
- Good accuracy on cursive and printed handwriting
- Base and Large model variants available
- Optional perspective correction for angled photos
- Optional GPT-5 correction for improved accuracy

**Usage:**
```bash
# Process handwritten images
python handwriting_ocr.py --input test_images/

# With perspective correction and AI correction
python handwriting_ocr.py --input test_images/ --perspective-correction --llm-correction
```

### 2. DeepSeek-OCR (4-bit Quantized)

**Script:** `test_deepseek_ocr_4bit.py`

An advanced OCR model using 4-bit quantization to reduce memory requirements while maintaining accuracy.

**Features:**
- Memory-efficient (runs on ~3GB GPU memory)
- 4-bit quantization for faster inference
- Grounding mode for layout-aware extraction
- Saves results in multiple formats (.mmd, .json, .txt)

**Usage:**
```bash
# Process single image
python test_deepseek_ocr_4bit.py --image test_images/myqueue.png

# Process multiple images (batch mode)
python test_deepseek_ocr_4bit.py --input-dir test_images/ --output output_deepseek_4bit

# First 5 images only
python test_deepseek_ocr_4bit.py --input-dir test_images/ --output output_deepseek_4bit --max 5
```

## Setup

### System Requirements

**For TrOCR:**
- Python 3.11+
- GPU recommended (CUDA support), CPU also works
- ~2-4GB GPU memory for base model

**For DeepSeek-OCR:**
- Python 3.11+
- GPU with CUDA support required
- ~3GB GPU memory (with 4-bit quantization)

### Installation

```bash
# Install dependencies
pip install -r requirements_handwriting.txt

# For DeepSeek 4-bit quantization specifically
pip install -r requirements_deepseek_4bit.txt

# Set up OpenAI API key (if using LLM correction)
echo "OPENAI_API_KEY=your-key-here" > ../.env
```

## Test Images

The `test_images/` directory contains sample handwritten images:
- `acceptance_poem_robert_frost_handwritten.png` - Cursive poetry
- `myqueue.png` - Technical notes
- `pyraven_hawk_lang_notes.png` - Mixed handwriting

## Results

Results are saved to output directories:
- TrOCR: `output/handwriting_results.csv`, `output/extracted_handwriting.txt`
- DeepSeek: `output_deepseek_4bit/all_results.json`, `output_deepseek_4bit/all_extracted_text.txt`

## Documentation

- **`HANDWRITING_DATASETS.md`** - Comprehensive guide to handwriting OCR datasets (IAM, RIMES, CVL, etc.)
- **`DEEPSEEK_4BIT_TEST.md`** - Technical documentation for DeepSeek-OCR 4-bit quantization
- **`ARTICLE_DRAFT.md`** - Draft article on handwriting recognition techniques

## Benchmarking

Handwriting OCR can be benchmarked using the root `benchmark.py` script:

```bash
# Benchmark TrOCR
cd ..
python benchmark.py --input handwriting/test_images/ --methods trocr trocr_gpt5
```

## Academic References

### TrOCR
Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., ... & Wei, F. (2023). **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models**. *Proceedings of the AAAI Conference on Artificial Intelligence, 37*(11), 13094-13102. [https://arxiv.org/abs/2109.10282](https://arxiv.org/abs/2109.10282)

## Example Scripts

- **`ds_ocr_example.py`** - Basic DeepSeek-OCR usage example
- **`ds_vllm_example.py`** - DeepSeek-OCR with vLLM integration

## Notes

This is an independent exploration of handwriting OCR techniques. The models and approaches here are more computationally intensive than the traditional OCR used in the main project, but offer significantly better accuracy for handwritten text.
