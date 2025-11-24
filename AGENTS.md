# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python OCR (Optical Character Recognition) project demonstrating two complementary approaches:

1. **Document OCR**: Traditional OCR using PyTesseract for scanned documents and PDFs
2. **Handwriting OCR**: Transformer-based OCR using Microsoft's TrOCR for handwritten notes

Both approaches use OpenAI's GPT-5 model for intelligent error correction to improve accuracy.

The project includes benchmarking tools to compare methods and an interactive Streamlit viewer for visualizing results.

## Development Setup

### System Dependencies

Install tesseract and poppler (required by Python packages):

```bash
# macOS/Linux with Homebrew
brew install tesseract poppler
```

### Python Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Running the Tools

```bash
# Document OCR (pytesseract)
python text_from_pdfs.py [--max N]

# Handwriting OCR (TrOCR)
python handwriting_ocr.py --input handwriting_images/ [--perspective-correction] [--llm-correction]

# Benchmarking
python benchmark.py --input images/ --methods pytesseract trocr --create-template
python benchmark.py --input images/ --methods pytesseract trocr pytesseract_gpt5 trocr_gpt5

# Interactive viewer
streamlit run viewer_app.py
```

## Code Architecture

### Core Pipeline Pattern

Both OCR approaches follow a similar multi-stage pipeline:

1. **Image Preprocessing** → 2. **Text Extraction** → 3. **LLM Correction** → 4. **Output Generation**

This separation allows easy comparison between methods and optional correction.

### Key Scripts and Their Roles

- **`text_from_pdfs.py`**: Document OCR pipeline (pytesseract)
  - `preprocess_image()`: Grayscale → noise removal → Otsu thresholding
  - `extract_text()`: Region-based text extraction using morphological operations
  - `ask_the_english_prof()`: GPT-5 correction for OCR errors

- **`handwriting_ocr.py`**: Handwriting OCR pipeline (TrOCR)
  - `TrOCRModel` class: HuggingFace transformer model wrapper
  - `perspective_correction()`: Correct angled photos using perspective transform
  - `correct_with_llm()`: GPT-5 correction tailored for handwriting errors

- **`benchmark.py`**: OCR method comparison and accuracy measurement
  - `OCRBenchmark` class: Unified benchmarking framework
  - Ground truth loading from `ground_truth/{image_basename}_ref.txt` files
  - Calculates CER (Character Error Rate) and WER (Word Error Rate)
  - Supports methods: `pytesseract`, `pytesseract_no_preprocess`, `trocr`, `pytesseract_gpt5`, `trocr_gpt5`

- **`viewer_app.py`**: Streamlit app for interactive result viewing
  - Two modes: Document OCR vs. Handwriting OCR
  - Displays: original image, preprocessed image, extracted text, corrected text
  - Navigation controls and page slider

- **`common.py`**: Shared utilities
  - `get_preproc_path()`: Generate preprocessed image filename

### Image Processing Design

**Document OCR preprocessing** (for typed/printed text):
- Aggressive preprocessing: grayscale → median blur → binary threshold
- Region-based extraction using morphological dilation to find text blocks
- Sorts regions by Y-coordinate to preserve reading order

**Handwriting OCR preprocessing** (for photos):
- Light preprocessing: CLAHE for contrast enhancement
- Optional perspective correction for angled photos
- TrOCR model handles most preprocessing internally

### LLM Correction Strategy

Two-stage correction approach:
1. **OCR extraction**: Get raw text from image
2. **GPT-5 correction**: Fix common OCR errors using context

The prompts are carefully crafted to preserve original content while fixing errors:
- Document OCR: "Correct any typos caused by bad OCR..."
- Handwriting OCR: Specialized prompt addressing handwriting-specific errors (similar letters, cursive word boundaries)

### Data Flow

```
Input: image files listed in input_file_list.txt
  ↓
Preprocessing: save to output/{basename}_proc.jpg
  ↓
Extraction: raw OCR text
  ↓
Correction: GPT-5 corrected text
  ↓
Output: results.csv (or handwriting_results.csv)
        extracted.txt
        corrected.txt
```

### Benchmarking Architecture

Ground truth is stored in individual files rather than a central JSON:
- Format: `ground_truth/{image_basename}_ref.txt`
- Example: `images/IMG_8479.jpg` → `ground_truth/IMG_8479_ref.txt`
- Files can contain comment lines starting with `#`

The benchmark compares methods using Levenshtein distance for CER/WER calculations.

## Important Configuration Details

- **OpenAI Model**: Uses `gpt-5` model (defined in `text_from_pdfs.py:18` and `handwriting_ocr.py:355`)
- **TrOCR Models**: Default is `microsoft/trocr-base-handwritten`, can also use `microsoft/trocr-large-handwritten`
- **Output Directory**: All outputs go to `output/` directory (created automatically)
- **Morphological Kernel**: 50×40 rectangle for document region detection (`text_from_pdfs.py:24`)

## Common Pitfalls

1. **Missing system dependencies**: pytesseract requires tesseract to be installed at system level
2. **Ground truth format**: Benchmark now uses individual `*_ref.txt` files, not a central JSON
3. **Model download**: First run of TrOCR downloads ~1GB model from HuggingFace
4. **GPU vs CPU**: TrOCR can run on CPU but GPU is much faster (`--no-gpu` flag available)
5. **Image paths**: The viewer app uses different path conventions for document vs handwriting results

## Testing Approach

This project uses manual benchmarking rather than unit tests:
- Create ground truth files with correct text
- Run benchmark comparing different methods
- Review generated markdown reports

The benchmark script generates both CSV results and human-readable markdown reports.

## Library and API Documentation

Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.
