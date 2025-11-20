# Python OCR using Pytesseract, Pillow, and openCV

## Overview

This project demonstrates two approaches to OCR (Optical Character Recognition):

1. **Document OCR**: Extract text from scanned documents using [PyTesseract](https://github.com/h/pytesseract), [Pillow](https://python-pillow.org/), and [opencv-python](https://github.com/opencv/opencv-python)

Both approaches use OpenAI's [GPT-5](https://openai.com/index/hello-gpt-5/) to correct OCR errors and improve accuracy.

The project includes:
- Image preprocessing pipelines optimized for each use case
- Batch processing capabilities
- Interactive [Streamlit](https://streamlit.io) viewer for comparing results
- Benchmarking tools for measuring accuracy and performance
- **Tutorial outline** for building similar OCR systems (see [TUTORIAL_OUTLINE.md](TUTORIAL_OUTLINE.md))

The process is visualized with a Streamlit app that shows the original image, preprocessed image, extracted text, and corrected text for each page.

## Tutorial

This repository serves as a comprehensive tutorial for building OCR systems. See [TUTORIAL_OUTLINE.md](TUTORIAL_OUTLINE.md) for a detailed guide covering:
- Traditional document OCR with pytesseract
- Handwriting recognition with Microsoft's TrOCR (transformer-based architecture)
- AI-powered error correction
- Performance benchmarking
- Production deployment considerations

## Use Cases

| Feature | Document OCR | Handwriting OCR |
|---------|--------------|----------------|
| Input | Scanned documents, PDFs | Photos of handwritten notes |
| Best for | Typed/printed text | Cursive or printed handwriting |
| Challenges | Aged paper, fading | Varied angles, lighting |
| Script | `text_from_pdfs.py` | `handwriting_ocr.py` |

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
brew install tesseract
brew install poppler
```

### Python Environment

T TODO: 


### API Configuration

Create a `.env` file with your OpenAI API key (required for LLM correction):

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Usage

### Document OCR (Pytesseract)

Process scanned documents or PDFs:

```bash
python text_from_pdfs.py [--max N]
```

Results are saved to:
- `output/results.csv` - Processing results
- `output/extracted.txt` - Raw OCR text
- `output/corrected.txt` - LLM-corrected text

### Handwriting OCR (TrOCR)

Process photos of handwritten notes using Microsoft's TrOCR:

```bash
# Create input directory and add your handwritten note photos
mkdir handwriting_images
# Add your photos to handwriting_images/

# Run handwriting OCR
python handwriting_ocr.py --input handwriting_images/

# Optional: with perspective correction and LLM correction
python handwriting_ocr.py --input handwriting_images/ --perspective-correction --llm-correction
```

Results are saved to:
- `output/handwriting_results.csv` - Processing results
- `output/extracted_handwriting.txt` - Raw OCR text
- `output/corrected_handwriting.txt` - Corrected text

### Benchmarking

Compare OCR methods and measure accuracy:

```bash
# Create ground truth template
python benchmark.py --input handwriting_images/ --create-template

# Edit ground_truth.json to add correct text for each image

# Run benchmark
python benchmark.py --input handwriting_images/ --ground-truth ground_truth.json --methods pytesseract trocr
```

### Interactive Viewer

Launch the Streamlit app to view results:

```bash
streamlit run viewer_app.py
```

The app allows you to:
- Switch between document OCR and handwriting OCR results
- Compare original, preprocessed, extracted, and corrected versions
- Navigate through pages with buttons or slider

The draft, combined version including some hand edits is in [august_anton.md](august_anton.md).

You can generate a LaTex or PDF version of the combined, corrected pages like this.

```bash
# generate Latex 
pandoc -f markdown -t latex --wrap=preserve  august_anton.md -o august_anton.tex

# Generate DPF
pandoc -f markdown -t pdf --wrap=preserve  august_anton.md -o august_anton.pdf
```

A PDF version formatted in LaTex using overleaf to adjust the one generated by pandoc is at [August_Anton___Reflections.pdf](August_Anton___Reflections.pdf).

## Academic Citations

This project builds on several important research works and open-source tools:

### TrOCR

Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., ... & Wei, F. (2023). **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models**. *Proceedings of the AAAI Conference on Artificial Intelligence, 37*(11), 13094-13102. [https://arxiv.org/abs/2109.10282](https://arxiv.org/abs/2109.10282)

> TrOCR is a transformer-based OCR model that leverages pre-trained image and text transformers for state-of-the-art text recognition. It uses an encoder-decoder architecture and achieves excellent results on handwritten and printed text.

### GPT-4o

OpenAI. (2024). **GPT-4o System Card**. *arXiv preprint arXiv:2410.21276*. [https://arxiv.org/abs/2410.21276](https://arxiv.org/abs/2410.21276)

> GPT-4o is used in this project for intelligent OCR error correction, leveraging its language understanding to fix common OCR mistakes while preserving the original meaning.

### Tesseract OCR

Smith, R. (2007). **An Overview of the Tesseract OCR Engine**. In *Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR '07)*, pp. 629-633. IEEE Computer Society.

> Tesseract is the foundational open-source OCR engine used for traditional document text extraction in this project.

### BibTeX Entries

```bibtex
@inproceedings{li2023trocr,
  title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models},
  author={Li, Minghao and Lv, Tengchao and Chen, Jingye and Cui, Lei and Lu, Yijuan and Florencio, Dinei and Zhang, Cha and Li, Zhoujun and Wei, Furu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={11},
  pages={13094--13102},
  year={2023},
  url={https://arxiv.org/abs/2109.10282}
}

@article{openai2024gpt4o,
  title={GPT-4o System Card},
  author={OpenAI},
  journal={arXiv preprint arXiv:2410.21276},
  year={2024},
  url={https://arxiv.org/abs/2410.21276}
}

@inproceedings{smith2007tesseract,
  author = {Ray Smith},
  title = {An Overview of the Tesseract OCR Engine},
  booktitle = {ICDAR '07: Proceedings of the Ninth International Conference on Document Analysis and Recognition},
  year = {2007},
  pages = {629--633},
  publisher = {IEEE Computer Society}
}
```

### Key Libraries and Tools

- **OpenCV** - Computer vision and image preprocessing: [https://opencv.org/](https://opencv.org/)
- **PyTesseract** - Python wrapper for Tesseract: [https://github.com/h/pytesseract](https://github.com/h/pytesseract)
- **HuggingFace Transformers** - Deep learning model loading: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- **Pillow (PIL)** - Python image processing: [https://python-pillow.org/](https://python-pillow.org/)
- **Streamlit** - Interactive web applications: [https://streamlit.io](https://streamlit.io)

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
