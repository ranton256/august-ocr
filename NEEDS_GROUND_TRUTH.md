# Files Requiring Ground Truth Data for CER/WER Calculation

## Ground Truth Status

**Images with ground truth (3):**

- ✅ **images/IMG_8478.jpg** - Title page ("REFLECTIONS", "Memories of the Life of August Anton")
- ✅ **images/IMG_8479.jpg** - First page of text (introduction and early life)
- ✅ **images/IMG_8480.jpg** - Historical context about Anhalt princes

**Images needing ground truth (2):**

- ⚠️ **images/IMG_8481.jpg** - Story about the forester and life at the mill/forestry
- ⚠️ **images/IMG_8482.jpg** - Continuation of forester story and Knight Roland

## All Available Images (28 total)

The following images are available in the `images/` directory but have not been benchmarked yet:

### Already Benchmarked (5)

- IMG_8478.jpg ✓
- IMG_8479.jpg ✓
- IMG_8480.jpg ✓
- IMG_8481.jpg ✓
- IMG_8482.jpg ✓

### Not Yet Benchmarked (23)

- IMG_8483.jpg
- IMG_8484.jpg
- IMG_8485.jpg
- IMG_8486.jpg
- IMG_8487.jpg
- IMG_8488.jpg
- IMG_8489.jpg
- IMG_8490.jpg
- IMG_8491.jpg
- IMG_8492.jpg
- IMG_8493.jpg
- IMG_8494.jpg
- IMG_8496.jpg
- IMG_8497.jpg
- IMG_8498.jpg
- IMG_8499.jpg
- IMG_8500.jpg
- IMG_8501.jpg
- IMG_8502.jpg
- IMG_8503.jpg
- IMG_8504.jpg
- IMG_8505.jpg
- IMG_8506.jpg
- IMG_8507.jpg
- IMG_8508.jpg

## How to Create Ground Truth Data

Ground truth is stored as individual text files in the `ground_truth/` directory. Each file is named after the image with `_ref.txt` replacing the image extension.

**File naming pattern:**

- `images/IMG_8479.jpg` → `ground_truth/IMG_8479_ref.txt`
- `images/IMG_8480.jpg` → `ground_truth/IMG_8480_ref.txt`

1. **Create template files:**

   ```bash
   python benchmark.py --input images/ --create-template
   ```

   This creates template files in `ground_truth/` directory with headers for each image. Files that already exist are skipped.

2. **Edit the `*_ref.txt` files** to add the correct text for each image:

   Each file has a header (lines starting with `#`) that you can keep or remove. Add the correct text below the header:

   ```
   # Ground Truth for images/IMG_8481.jpg
   #
   # Enter the correct text from this image below:
   #
   
   [Your transcribed text goes here]
   ```

   Example: `ground_truth/IMG_8478_ref.txt` contains:

   ```
   "REFLECTIONS"

   Memories of the Life of August Anton
   written as a result of the motivation of his friends
   published by the writer
   (Eagle)
   ```

3. **Re-run benchmarks** (ground truth is automatically loaded):

   ```bash
   python benchmark.py --input images/ --methods pytesseract trocr pytesseract_gpt5 trocr_gpt5 --max 5
   ```

   The benchmark script automatically looks for `ground_truth/{image_basename}_ref.txt` files and uses them if found. No `--ground-truth` flag needed!

## Priority for Ground Truth Creation

**High Priority** (for initial validation):

- IMG_8478.jpg - Short title page, easiest to transcribe
- IMG_8479.jpg - First page, important for establishing baseline

**Medium Priority** (for comprehensive metrics):

- IMG_8481.jpg and IMG_8482.jpg - Already benchmarked, need ground truth to calculate CER/WER

**Lower Priority** (for full dataset):

- Remaining 23 images - Can be added incrementally

## Notes

- Ground truth should be the **exact correct text** as it appears in the original document
- Text should be normalized (consistent line breaks, spacing) for accurate comparison
- The benchmark script normalizes text before calculating CER/WER (removes punctuation, lowercases, normalizes whitespace)
- You can use the existing `august_anton.md` file as a reference for the correct text, but you'll need to match it to specific images
