# Digitizing Historical Documents: OCR with PyTesseract and AI Correction

## Overview

This project demonstrates a complete pipeline for digitizing historical documents using traditional OCR combined with modern AI correction. The approach uses [PyTesseract](https://github.com/h/pytesseract), [OpenCV](https://opencv.org/), and OpenAI's GPT-5 model to convert scanned pages into accurate, searchable text.

**What makes this project unique:**

- **Real-world example** - Digitizes 30 pages of my great-great-grandfather's autobiography
- **Complete pipeline** - From raw scans to formatted markdown/PDF output
- **AI-powered correction** - Uses GPT-5 to fix OCR errors while preserving original meaning
- **Benchmarked results** - Measured accuracy improvements with preprocessing and correction
- **Production-ready** - Batch processing, interactive viewer, and comprehensive documentation

The project includes:

- Image preprocessing pipeline optimized for aged documents
- Batch processing capabilities for multi-page documents
- Interactive [Streamlit](https://streamlit.io) viewer for result validation
- Benchmarking tools for measuring accuracy improvements
- Complete tutorial for building similar systems

## The August Anton Project

This OCR pipeline was developed to digitize a 30-page autobiography written by my paternal great-great-grandfather, August Anton, sometime before his death in 1911. The document chronicles his fascinating life, including participation in the 1848 German revolution, immigration to America, and establishing a successful carpentry business in Birmingham, Alabama.

The challenge: Convert 30 pages of photocopied, aged printed text into searchable, accurate digital form while preserving the original voice and content.

## License

This code, text, and other original work on my part in this repo is under the MIT license.

The original text by August Anton, described below, is to the best of my understanding in the public domain in the United States at this point since the author passed away in 1911.

## Example Screenshot of the App

![image-20240714165057654](screenshots/image-20240714165057654.png)

## Background on the Text

The text used in this project are photos of a brief autobiographical work by my paternal great great grandfather, August Anton.

He had quite an interesting life, including the 1848 revolution in Germany, being banned from Bremen for running a study group, immigrating to America, running a business in Birmingham where he evenutally settled. He was a master carpenter, in the literal sense of having apprenticed, worked and traveled as a journeyman, and then passed a master examination.

The text was provided to me as thirty pages of photocopies by my uncle, James (Jim) Anton in 1999, a short while after my father passed away.  I believe he provided the same text to a number of his other relatives as well.

According to [Ancestry.com](https://www.ancestry.com),

> August Fredrick Anton was born in 1830 in Zerbst, Saxony-Anhalt,  Germany, the son of Sophia and August. He married Sophia Bertha Tiebetz  in 1858 in Germany. They had six children in 13 years. He died on  January 2, 1911, in Birmingham, Alabama, having lived a long life of 81  years.

## Setup

### System Dependencies

The python packages used in this project require the tesseract and poppler libraries.

You can install them with Homebrew on MacOS or Linux. See the tesseract or poppler documentation for instructions for other platforms.

```bash
# macOS/Linux with Homebrew
brew install tesseract poppler
```

### Python Environment

Create a virtual environment and install dependencies:

```bash
# Create a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt
```

### API Configuration

Create a `.env` file with your OpenAI API key (required for LLM correction):

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Usage

### Basic OCR Pipeline

Process scanned documents or PDFs:

```bash
python text_from_pdfs.py [--max N]
```

The script processes images listed in `input_file_list.txt` and performs:

1. Image preprocessing (grayscale → noise reduction → thresholding)
2. Region-based text extraction with PyTesseract
3. GPT-5 error correction
4. Output generation

Results are saved to:

- `output/results.csv` - Processing results with all stages
- `output/extracted.txt` - Raw OCR text
- `output/corrected.txt` - AI-corrected text
- `output/*_proc.jpg` - Preprocessed images for inspection

### Benchmarking

Measure accuracy and compare preprocessing approaches:

```bash
# Create ground truth template files
python benchmark.py --input images/ --create-template

# Edit ground_truth/*_ref.txt files with correct text for each image

# Run benchmark comparing different methods
python benchmark.py --input images/ \
  --methods pytesseract pytesseract_no_preprocess pytesseract_gpt5
```

**Supported methods:**
- `pytesseract` - Document OCR with preprocessing
- `pytesseract_no_preprocess` - Document OCR without preprocessing
- `pytesseract_gpt5` - Document OCR with GPT-5 correction

**Ground Truth Format:**
Ground truth is stored in individual text files:
- Format: `ground_truth/{image_basename}_ref.txt`
- Example: `images/IMG_8479.jpg` → `ground_truth/IMG_8479_ref.txt`
- Files can contain comment lines starting with `#`

The benchmark measures:

- **Character Error Rate (CER)** - Percentage of characters incorrectly recognized
- **Word Error Rate (WER)** - Percentage of words with errors
- **Processing time** - Seconds per page

### Interactive Viewer

Launch the Streamlit app to view and validate results:

```bash
streamlit run viewer_app.py
```

The app allows you to:

- View original and preprocessed images side-by-side
- Compare raw OCR output with AI-corrected text
- Navigate through pages with buttons or slider
- Validate results before final export

The draft, combined version including some hand edits is in [august_anton.md](august_anton.md).

You can generate a LaTex or PDF version of the combined, corrected pages like this.

```bash
# generate Latex 
pandoc -f markdown -t latex --wrap=preserve  august_anton.md -o august_anton.tex

# Generate DPF
pandoc -f markdown -t pdf --wrap=preserve  august_anton.md -o august_anton.pdf
```

A PDF version formatted in LaTex using overleaf to adjust the one generated by pandoc is at [August_Anton___Reflections.pdf](August_Anton___Reflections.pdf).

## Code Architecture

### Pipeline Pattern

The OCR pipeline follows a multi-stage process:

1. **Image Preprocessing** → 2. **Text Extraction** → 3. **LLM Correction** → 4. **Output Generation**

This separation allows easy comparison between methods and optional correction.

### Key Scripts

- **`text_from_pdfs.py`**: Document OCR pipeline (pytesseract)
  - Region-based text extraction using morphological operations
  - Sorts regions by Y-coordinate (then X) to preserve reading order
  - GPT-5 correction for OCR errors

- **`benchmark.py`**: OCR method comparison and accuracy measurement
  - Calculates CER (Character Error Rate) and WER (Word Error Rate)
  - Generates CSV results and markdown reports

- **`viewer_app.py`**: Streamlit app for interactive result viewing

- **`common.py`**: Shared utilities

### Image Processing

**Document OCR preprocessing** (for typed/printed text):
- Aggressive preprocessing: grayscale → median blur → binary threshold
- Region-based extraction using morphological dilation to find text blocks
- Morphological kernel: 50×40 rectangle for document region detection

### Output Directory

All outputs go to `output/` directory (created automatically):
- Preprocessed images: `output/{basename}_proc.jpg`
- Results CSV: `output/results.csv`
- Extracted text: `output/extracted.txt`
- Corrected text: `output/corrected.txt`

## Common Pitfalls

1. **Missing system dependencies**: pytesseract requires tesseract to be installed at system level
2. **Ground truth format**: Benchmark uses individual `*_ref.txt` files in `ground_truth/` directory, not a central JSON

## Important Configuration

- **OpenAI Model**: Uses `gpt-5` model (defined in `text_from_pdfs.py`)
- **Input file list**: Document OCR reads image paths from `input_file_list.txt`

## Academic Citations

This project builds on important research works and open-source tools:

### Tesseract OCR

Smith, R. (2007). **An Overview of the Tesseract OCR Engine**. In *Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR '07)*, pp. 629-633. IEEE Computer Society.

> Tesseract is the foundational open-source OCR engine used for document text extraction in this project. Originally developed by HP in the 1980s, it was open-sourced by Google and remains one of the most accurate OCR engines for printed text.

### GPT-5

OpenAI. (2025). **GPT-5 System Card**. <https://openai.com/index/gpt-5-system-card/>

> GPT-5 is used in this project for intelligent OCR error correction, leveraging its language understanding to fix common OCR mistakes while preserving the original meaning and style.

### BibTeX Entries

```bibtex
@inproceedings{smith2007tesseract,
  author = {Ray Smith},
  title = {An Overview of the Tesseract OCR Engine},
  booktitle = {ICDAR '07: Proceedings of the Ninth International Conference on Document Analysis and Recognition},
  year = {2007},
  pages = {629--633},
  publisher = {IEEE Computer Society}
}

@manual{openai2025gpt5,
  title        = {GPT-5 System Card},
  author       = {OpenAI},
  year         = {2025},
  note         = {\url{https://cdn.openai.com/gpt-5-system-card.pdf}},
  howpublished = {\url{https://cdn.openai.com/gpt-5-system-card.pdf}},
  organization = {OpenAI}
}
```

### Key Libraries and Tools

- **Tesseract OCR** - Open-source OCR engine: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
- **PyTesseract** - Python wrapper for Tesseract: [https://github.com/h/pytesseract](https://github.com/h/pytesseract)
- **OpenCV** - Computer vision and image preprocessing: [https://opencv.org/](https://opencv.org/)
- **Pillow (PIL)** - Python image processing: [https://python-pillow.org/](https://python-pillow.org/)
- **Streamlit** - Interactive web applications: [https://streamlit.io](https://streamlit.io)
- **OpenAI API** - GPT-5 for error correction: [https://platform.openai.com/docs](https://platform.openai.com/docs)

## Credits

I took inspiration and some code snippets from various articles online:

This article has a good overview of image preprocessing methods for OCR.

- <https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7>

The article provided a great example of identifying rectangles in the image to process separately.

- <https://medium.com/@siromermer/extracting-text-from-images-ocr-using-opencv-pytesseract-aa5e2f7ad513>

## Future Improvements

- Change the Streamlit viewer app to use actual Streamlit pages and better controls for next and previous.

- Use the dewarping algorithm discussed in <https://mzucker.github.io/2016/08/15/page-dewarping.html> and implemented in <https://github.com/tachylatus/page_dewarp>.

- Use [GPT-4o vision support]( https://platform.openai.com/docs/guides/vision) to use the image as part of the input to the correction process for the extracted text.

---

If you enjoyed content like this, I write The Cognitive Engineering Newsletter — short essays on attention, learning systems, and AI agents.
👉 <https://ranton.org/newsletter>
