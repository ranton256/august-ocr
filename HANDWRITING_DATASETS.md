# Handwriting Recognition Datasets: A Comprehensive Guide

This guide provides recommendations for obtaining datasets for handwriting recognition, whether you're testing the OCR system, training custom models, or benchmarking performance.

## Table of Contents

1. [Creating Your Own Dataset](#creating-your-own-dataset)
2. [Public Handwriting Datasets](#public-handwriting-datasets)
3. [Synthetic Datasets](#synthetic-datasets)
4. [Dataset Preparation Best Practices](#dataset-preparation-best-practices)
5. [Recommended Datasets for Different Use Cases](#recommended-datasets-for-different-use-cases)

---

## Creating Your Own Dataset

### Option 1: Personal Notes Method (Recommended for Testing)

This is the best approach for testing the handwriting OCR pipeline with realistic data.

**Steps:**

1. **Gather handwritten materials**
   - Personal notes, journals, or diaries
   - Grocery lists, to-do lists
   - Meeting notes or class notes
   - Greeting cards or letters

2. **Photograph with your phone**
   - Use good lighting (natural light works best)
   - Take photos straight-on to minimize perspective distortion
   - Ensure text is in focus
   - Capture at high resolution (modern phones are fine)

3. **Organize your images**
   ```bash
   mkdir handwriting_images
   mv *.jpg handwriting_images/
   ```

4. **Create ground truth annotations**
   ```bash
   # Generate template
   python benchmark.py --input handwriting_images/ --create-template

   # Edit ground_truth.json and type the correct text for each image
   ```

**Pros:**
- Free and immediate
- Realistic lighting and angle variations
- Tests your actual use case
- Complete control over content

**Cons:**
- Time-consuming to create ground truth
- Limited volume
- May contain sensitive information

**Privacy Note:** If using personal notes, create a sanitized test set without sensitive information.

---

### Option 2: Crowdsourced Collection

Create a custom dataset by collecting handwriting samples from multiple people.

**Steps:**

1. **Design collection templates**
   - Create printable forms with specific text to copy
   - Include sentences with varied vocabulary
   - Add blank areas for free-form writing

2. **Collect samples**
   - Ask friends, colleagues, or family to fill out forms
   - Use Google Forms or similar for remote collection
   - Request photos of completed forms

3. **Quality control**
   - Verify images are legible
   - Ensure consistent lighting
   - Standardize image sizes

**Example Collection Form:**

```
┌─────────────────────────────────────────────┐
│  Handwriting Sample Collection Form         │
├─────────────────────────────────────────────┤
│                                             │
│  Please write the following sentence:      │
│  "The quick brown fox jumps over the lazy   │
│   dog near 1234 Oak Street."                │
│                                             │
│  ______________________________________     │
│  ______________________________________     │
│                                             │
│  Now write a few sentences about your day:  │
│                                             │
│  ______________________________________     │
│  ______________________________________     │
│  ______________________________________     │
│  ______________________________________     │
│                                             │
└─────────────────────────────────────────────┘
```

**Pros:**
- Custom vocabulary and content
- Diverse handwriting styles
- Known ground truth (you provided the text)
- Scalable with effort

**Cons:**
- Requires coordination and participation
- Time-intensive
- May need incentives for participation

---

## Public Handwriting Datasets

### 1. IAM Handwriting Database ⭐ **Most Recommended**

**Overview:**
- 1,539 pages of scanned handwritten text
- 13,353 isolated text lines
- 115,320 words
- Written by 657 different writers
- English language
- Modern handwriting (1990s-2000s)

**Access:**
- Website: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
- Registration required (free for academic use)
- License: Free for research, commercial licenses available

**Best for:**
- Benchmarking English handwriting OCR
- Training custom models
- Academic research

**Dataset Structure:**
```
iam/
├── forms/          # Full page images
├── lines/          # Individual line images
├── sentences/      # Sentence-level images
└── words/          # Word-level images
```

**How to use with this project:**
```bash
# Download and extract IAM dataset
# Place line images in handwriting_images/

python handwriting_ocr.py --input handwriting_images/ --max 100

# Use provided ground truth for benchmarking
python benchmark.py --input handwriting_images/ --ground-truth iam_ground_truth.json
```

---

### 2. RIMES Database (French Handwriting)

**Overview:**
- French language handwritten documents
- 12,723 training pages
- 1,500 test pages
- Letters and forms
- Modern handwriting

**Access:**
- Website: http://www.a2ialab.com/doku.php?id=rimes_database
- Free for research purposes

**Best for:**
- French language OCR
- Multi-language testing
- Comparing performance across languages

---

### 3. CVL Database (Historical Documents)

**Overview:**
- Single-writer and multi-writer datasets
- Modern handwriting
- 310 writers
- 7 pages per writer
- German and English text

**Access:**
- Website: https://cvl.tuwien.ac.at/research/cvl-databases/
- Free for research

**Best for:**
- Testing writer-specific models
- Historical document digitization

---

### 4. MNIST and EMNIST (Digits and Characters)

**Overview:**
- MNIST: 70,000 handwritten digits (0-9)
- EMNIST: Extended to include letters
- Preprocessed and normalized
- 28×28 grayscale images

**Access:**
- Website: http://yann.lecun.com/exdb/mnist/
- EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
- Direct download, no registration

**Best for:**
- Character-level recognition
- Form processing (phone numbers, zip codes)
- Quick testing and prototyping

**How to use:**
```python
# Example: Testing digit recognition
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Save a few test images
for i in range(10):
    img = Image.fromarray(train_images[i])
    img.save(f'handwriting_images/digit_{train_labels[i]}_{i}.png')
```

---

### 5. Google Handwriting Recognition (QuickDraw)

**Overview:**
- 50 million drawings
- 345 categories
- Time-series stroke data
- Simple sketches and text

**Access:**
- Website: https://quickdraw.withgoogle.com/data
- Direct download via Google Cloud Storage
- Open dataset, freely available

**Best for:**
- Stroke-based recognition research
- Understanding handwriting dynamics
- Creative applications

---

### 6. HierText (Scene Text + Handwriting)

**Overview:**
- 11,639 images
- Mix of printed and handwritten text
- Real-world photos (signs, documents, etc.)
- Multilingual

**Access:**
- Website: https://github.com/google-research-datasets/hiertext
- Free download from GitHub

**Best for:**
- Mixed content (printed + handwritten)
- Real-world photo testing
- Scene text OCR

---

## Synthetic Datasets

### Option 1: Using Handwriting Fonts

Generate synthetic handwriting using fonts.

**Tools:**
- [Handwriting fonts](https://www.1001fonts.com/handwritten-fonts.html)
- PIL/Pillow for rendering text
- OpenCV for adding realistic variations

**Example Code:**

```python
from PIL import Image, ImageDraw, ImageFont
import random

def generate_synthetic_handwriting(text, font_path, output_path):
    # Create image
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)

    # Load handwriting font
    font = ImageFont.truetype(font_path, size=40)

    # Add slight rotation for realism
    angle = random.uniform(-2, 2)

    # Draw text
    draw.text((50, 50), text, font=font, fill='black')

    # Rotate slightly
    img = img.rotate(angle, fillcolor='white', expand=False)

    # Add slight blur to simulate pen width variation
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    img.save(output_path)

# Generate dataset
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    # ... more sentences
]

for i, sentence in enumerate(sentences):
    generate_synthetic_handwriting(
        sentence,
        'handwriting_font.ttf',
        f'synthetic_{i}.png'
    )
```

**Pros:**
- Infinite scalability
- Perfect ground truth
- Controlled variations
- No privacy concerns

**Cons:**
- Not realistic (lacks natural variation)
- May not generalize to real handwriting
- Doesn't capture real-world challenges

---

### Option 2: GANs and Synthetic Generation

Use generative models to create realistic handwriting.

**Tools:**
- [Handwriting Synthesis](https://github.com/sjvasquez/handwriting-synthesis)
- [Handwritten Text Generation](https://github.com/omni-us/research-seq2seq-handwriting-synthesis)

**Best for:**
- Large-scale data augmentation
- Training custom models
- Research on handwriting generation

---

## Dataset Preparation Best Practices

### 1. Image Quality Standards

**Recommended specifications:**
- **Resolution**: Minimum 300 DPI for scans, 8+ megapixels for photos
- **Format**: JPEG or PNG
- **Color**: RGB or grayscale (grayscale reduces file size)
- **Lighting**: Even, diffuse lighting without harsh shadows
- **Focus**: Sharp text, no motion blur

### 2. Ground Truth Annotation

**Tools for creating ground truth:**

1. **Manual transcription**
   - Use a text editor with the image side-by-side
   - Type exactly what appears in the image
   - Include punctuation and capitalization

2. **Semi-automated approach**
   ```bash
   # Run OCR first to get initial text
   python handwriting_ocr.py --input dataset/

   # Edit the extracted text in extracted_handwriting.txt
   # Save corrected version as ground_truth.json
   ```

3. **Annotation tools**
   - [Label Studio](https://labelstud.io/) - General-purpose annotation
   - [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) - Browser-based
   - Custom Streamlit app (can extend viewer_app.py)

**Ground truth format:**

```json
{
  "handwriting_images/note1.jpg": "This is the first handwritten note.",
  "handwriting_images/note2.jpg": "This is the second note with numbers: 1234.",
  "handwriting_images/note3.jpg": "Meeting notes from Jan 15th..."
}
```

### 3. Dataset Organization

**Recommended structure:**

```
dataset/
├── raw/                    # Original images
│   ├── writer_001/
│   ├── writer_002/
│   └── ...
├── processed/              # Preprocessed images
│   ├── writer_001/
│   └── ...
├── ground_truth.json       # Text annotations
├── metadata.json           # Image metadata
└── splits/
    ├── train.txt          # Training set file list
    ├── val.txt            # Validation set
    └── test.txt           # Test set
```

**Metadata example:**

```json
{
  "raw/writer_001/page1.jpg": {
    "writer_id": "001",
    "date_captured": "2024-01-15",
    "device": "iPhone 13",
    "lighting": "natural",
    "angle": "straight-on",
    "language": "en"
  }
}
```

### 4. Train/Val/Test Split

**Recommended splits:**
- Training: 70-80%
- Validation: 10-15%
- Test: 10-15%

**Important:** Split by writer, not by page, to avoid data leakage.

```python
import random
from pathlib import Path

# Get unique writers
writers = set(Path('dataset/raw').glob('writer_*'))

# Shuffle and split
writers = list(writers)
random.shuffle(writers)

train_size = int(len(writers) * 0.7)
val_size = int(len(writers) * 0.15)

train_writers = writers[:train_size]
val_writers = writers[train_size:train_size+val_size]
test_writers = writers[train_size+val_size:]
```

---

## Recommended Datasets for Different Use Cases

### Use Case 1: Testing the Handwriting OCR Pipeline

**Recommended approach:**
1. **Start with**: Your own handwritten notes (10-20 images)
2. **Then try**: IAM Database sample (for comparison)
3. **For multilingual**: RIMES (French) or CVL (German/English)

**Why:** Personal notes test your real use case. IAM provides a benchmark.

---

### Use Case 2: Benchmarking OCR Accuracy

**Recommended datasets:**
1. **IAM Handwriting Database** - Standard benchmark
2. **RIMES** - French benchmark
3. **CVL** - Writer-specific testing

**Metrics to calculate:**
- Character Error Rate (CER)
- Word Error Rate (WER)
- Processing time

```bash
python benchmark.py --input iam_test_set/ --ground-truth iam_ground_truth.json
```

---

### Use Case 3: Training/Fine-tuning Custom Models

**Recommended datasets:**
1. **IAM Database** (primary training set)
2. **Augmented with**: Your domain-specific data
3. **Synthetic data** (for data augmentation)

**Training approach:**
1. Pre-train on IAM (large, general dataset)
2. Fine-tune on your specific handwriting style
3. Validate on held-out test set

---

### Use Case 4: Historical Document Digitization

**Recommended datasets:**
1. **CVL Database** - Modern historical documents
2. **Custom collection** - Your specific historical materials
3. **IAM Database** - For baseline comparison

**Special considerations:**
- Historical handwriting styles differ significantly
- Paper degradation and aging artifacts
- Different ink types and fading

---

### Use Case 5: Real-time Note Capture (Phone App)

**Recommended datasets:**
1. **HierText** - Real-world photos
2. **Custom phone photos** - Varied lighting/angles
3. **Synthetic with distortions** - Augmented data

**Testing priorities:**
- Various lighting conditions (bright, dim, mixed)
- Different angles (straight-on, 15°, 30°)
- Camera shake and slight blur
- Background clutter

---

## Quick Start Recommendations

### Absolute Beginner (1 hour setup)

```bash
# 1. Take photos of 5-10 handwritten notes with your phone
mkdir handwriting_images
# Move photos to handwriting_images/

# 2. Run OCR
python handwriting_ocr.py --input handwriting_images/

# 3. View results
streamlit run viewer_app.py
```

### Intermediate (1 day setup)

```bash
# 1. Download IAM Database (requires registration)
# Extract to iam_dataset/

# 2. Create ground truth template
python benchmark.py --input iam_dataset/test/ --create-template

# 3. Run benchmark
python benchmark.py \
  --input iam_dataset/test/ \
  --ground-truth ground_truth.json \
  --methods trocr pytesseract
```

### Advanced (1 week setup)

```bash
# 1. Collect diverse dataset (100+ images)
#    - Personal notes: 20 images
#    - IAM Database: 50 images
#    - Crowdsourced: 30 images

# 2. Create comprehensive ground truth
python benchmark.py --input dataset/ --create-template
# Edit ground_truth.json

# 3. Run full evaluation
python benchmark.py \
  --input dataset/ \
  --ground-truth ground_truth.json \
  --methods trocr pytesseract \
  --output full_evaluation.csv \
  --report evaluation_report.md

# 4. Analyze results and iterate
```

---

## Resources and Links

### Dataset Repositories
- [Papers with Code - Handwriting Recognition Datasets](https://paperswithcode.com/task/handwriting-recognition)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Hugging Face Datasets](https://huggingface.co/datasets?task_categories=task_categories:text-recognition)

### Tools
- [Label Studio](https://labelstud.io/) - Annotation tool
- [CVAT](https://github.com/opencv/cvat) - Computer Vision Annotation Tool
- [Roboflow](https://roboflow.com/) - Dataset management and augmentation

### Academic Papers
- [IAM Database Paper](https://ieeexplore.ieee.org/document/990556)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234) - State-of-the-art vision-language OCR
- [Handwriting Recognition Survey](https://arxiv.org/abs/2104.03064)

---

## Summary

**Best approach for most users:**

1. **Start small**: Use your own handwritten notes (10-20 images)
2. **Benchmark**: Download IAM Database test set for comparison
3. **Iterate**: Based on results, collect more domain-specific data
4. **Scale**: Use public datasets for training custom models if needed

**Key takeaway:** The best dataset depends on your specific use case. Start with readily available data (your own notes), validate with standard benchmarks (IAM), and expand as needed.
