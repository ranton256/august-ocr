# DeepSeek-OCR 4-bit Quantization Test

This document describes how to use the DeepSeek-OCR model with 4-bit quantization for handwriting recognition, which significantly reduces memory requirements compared to the full model.

## Background

Previously, we removed DeepSeek-OCR from this project due to high memory requirements (requiring GPUs only available with Colab Pro). However, by using 4-bit quantization, we can run the model on more modest hardware like a T4 GPU.

## Memory Comparison

- **Full DeepSeek-OCR**: ~24GB+ GPU memory required
- **4-bit Quantized**: ~6-8GB GPU memory (can run on T4 GPU)

## Installation

### Option 1: Install specific versions (recommended)

Use the verified working versions from the Colab notebook:

```bash
pip install -r requirements_deepseek_4bit.txt
```

### Option 2: Install individual packages

```bash
pip install bitsandbytes
pip install addict transformers==4.46.3 tokenizers==0.20.3
```

## HuggingFace Token

DeepSeek-OCR requires a HuggingFace token. Set it as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Or create a `.env` file:

```
HF_TOKEN=your_huggingface_token_here
```

## Usage

### Test on a single image

```bash
./test_deepseek_ocr_4bit.py --image images/IMG_8478.jpg
```

### Test on a batch of images

```bash
# Process all images in the images/ directory
./test_deepseek_ocr_4bit.py --input-dir images

# Process only first 3 images
./test_deepseek_ocr_4bit.py --input-dir images --max 3
```

### Customize the output directory

```bash
./test_deepseek_ocr_4bit.py --image images/IMG_8478.jpg --output my_results
```

### Use different prompts

The script supports several OCR prompts:

```bash
# Structured markdown conversion (default)
./test_deepseek_ocr_4bit.py --image test.jpg --prompt "<image>\n<|grounding|>Convert the document to markdown."

# General OCR
./test_deepseek_ocr_4bit.py --image test.jpg --prompt "<image>\n<|grounding|>OCR this image."

# Free OCR without layout
./test_deepseek_ocr_4bit.py --image test.jpg --prompt "<image>\nFree OCR."

# Detailed description
./test_deepseek_ocr_4bit.py --image test.jpg --prompt "<image>\nDescribe this image in detail."
```

### Adjust image processing settings

Different size configurations for different quality/speed tradeoffs:

```bash
# Tiny (fastest, lowest quality)
./test_deepseek_ocr_4bit.py --image test.jpg --base-size 512 --image-size 512

# Small
./test_deepseek_ocr_4bit.py --image test.jpg --base-size 640 --image-size 640

# Base (default, good balance)
./test_deepseek_ocr_4bit.py --image test.jpg --base-size 1024 --image-size 1024

# Large (best quality, slowest)
./test_deepseek_ocr_4bit.py --image test.jpg --base-size 1280 --image-size 1280

# Gundam mode (crop mode enabled)
./test_deepseek_ocr_4bit.py --image test.jpg --base-size 1024 --image-size 640 --crop-mode
```

## Output Files

The script creates several output files:

- `deepseek_4bit_result.json` - Complete results with metadata
- `extracted_text.txt` - Just the extracted text
- `result.mmd` - Markdown formatted result (if save_results=True)
- `result_with_boxes.jpg` - Image with bounding boxes (if save_results=True)

## Comparison with TrOCR

### TrOCR (current implementation)
- **Pros**: Lower memory requirements, runs on CPU, faster inference
- **Cons**: Less accurate on complex handwriting, simpler output

### DeepSeek-OCR 4-bit
- **Pros**: More accurate, structured output (markdown), handles complex layouts
- **Cons**: Requires GPU, higher memory than TrOCR, slower inference

## Example Workflow

```bash
# 1. Set up environment
export HF_TOKEN=your_token_here

# 2. Install dependencies
pip install -r requirements_deepseek_4bit.txt

# 3. Test on a single image first
./test_deepseek_ocr_4bit.py --image images/IMG_8478.jpg --output test_output

# 4. Check the results
cat test_output/extracted_text.txt

# 5. If satisfied, process more images
./test_deepseek_ocr_4bit.py --input-dir images --max 5 --output batch_output
```

## Troubleshooting

### CUDA Out of Memory

If you get OOM errors, try:

1. Use smaller image sizes: `--base-size 640 --image-size 640`
2. Process one image at a time
3. Restart Python kernel between runs to clear GPU memory

### Model Download Issues

If the model fails to download:

1. Check your HuggingFace token is valid
2. Ensure you have internet connection
3. Try: `huggingface-cli login`

### Version Conflicts

If you get version conflicts with the main requirements.txt:

```bash
# Create a separate virtual environment
python -m venv venv_deepseek
source venv_deepseek/bin/activate
pip install -r requirements_deepseek_4bit.txt
```

## Technical Details

The 4-bit quantization uses:

- **Method**: NF4 (Normal Float 4-bit)
- **Double quantization**: Enabled
- **Compute dtype**: torch.float
- **Loading strategy**: 4-bit with automatic device mapping

This configuration is based on the verified working setup from:
https://colab.research.google.com/github/Alireza-Akhavan/LLM/blob/main/deepseek_ocr_inference_4bit.ipynb

## Source Files

- **Test script**: `test_deepseek_ocr_4bit.py`
- **Jupyter notebook**: `DeepSeek_OCR_Quantized_to_4bit.ipynb`
- **Requirements**: `requirements_deepseek_4bit.txt`
