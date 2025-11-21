# Handwriting Recognition with Transformer-Based OCR Models

*This article explores modern approaches to handwriting recognition using transformer-based OCR models, extending foundational OCR techniques to the unique challenges of handwritten text.*

## Introduction

This work builds on traditional OCR approaches (demonstrated in our printed-text OCR project for digitizing historical documents) and extends them to handwriting recognition. While printed text OCR has largely been "solved" by tools like PyTesseract, handwriting presents unique challenges that benefit from modern deep learning approaches.

### The Handwriting Challenge

Handwritten text defies the assumptions that make traditional OCR work:

- **Inconsistent character shapes** - Same letter varies between instances
- **Connected letters** - Cursive writing lacks clear boundaries
- **Variable spacing** - Word boundaries are ambiguous
- **Personal styles** - Thousands of writing variations
- **Lighting and angle issues** - Photos of notes have varied conditions

Traditional OCR engines struggle with these variations because they rely on template matching and heuristics designed for printed fonts.

## Part I: Handwriting Recognition with TrOCR

Handwritten text requires a different approach. Transformer-based models like TrOCR excel where traditional OCR fails.

### Why TrOCR? The Transformer Advantage

Microsoft's TrOCR leverages transformer architecture:

- **Pre-trained** on millions of handwritten samples
- **Context-aware** - understands words, not just characters
- **Handles variations** - different writing styles, cursive, print
- **Modest hardware** - works on CPU, faster on GPU

**TrOCR vs. Traditional OCR:**

| Feature | Pytesseract | TrOCR |
|---------|-------------|--------|
| Printed text | Excellent | Good |
| Handwriting | Poor | Excellent |
| Speed (CPU) | Fast (~1s) | Moderate (~3-5s) |
| GPU acceleration | No | Yes |
| Training data | Rule-based + ML | Millions of samples |
| Cursive handling | Struggles | Strong |

### Loading the TrOCR Model

Our implementation uses HuggingFace transformers:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

class TrOCRModel:
    """Handwriting OCR using Microsoft's TrOCR"""

    def __init__(self, model_name="microsoft/trocr-base-handwritten", use_gpu=True):
        """
        Initialize TrOCR model

        Args:
            model_name: HuggingFace model identifier
                - microsoft/trocr-base-handwritten (faster, good accuracy)
                - microsoft/trocr-large-handwritten (slower, better accuracy)
            use_gpu: Use GPU if available
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        print(f"Loading {model_name} on {self.device}...")

        # Load processor (handles image preprocessing)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

        # Load model (transformer encoder-decoder)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print("TrOCR loaded successfully!")
```

**Model variants:**

- **trocr-base-handwritten** - 334M parameters, ~1GB download ([HuggingFace](https://huggingface.co/microsoft/trocr-base-handwritten))
- **trocr-large-handwritten** - 558M parameters, ~2GB download ([HuggingFace](https://huggingface.co/microsoft/trocr-large-handwritten))

For most applications, base is sufficient. Large provides marginal accuracy improvements at significantly higher cost.

### Preprocessing for Handwritten Photos

Photos of handwritten notes present unique challenges:

```python
def preprocess_image(self, image):
    """
    Preprocess photos for better handwriting recognition

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to handle varied lighting conditions
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply CLAHE for contrast enhancement
    # This helps with varied lighting in photos
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels and convert back to RGB
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image
```

**CLAHE explained:**

- **Adaptive** - Applies different thresholds to different regions
- **Contrast limited** - Prevents over-amplification of noise
- **Tile-based** - Divides image into 8×8 grids for local processing
- **Effective for** - Shadows, glare, uneven lighting

### Running TrOCR Inference

The inference pipeline is straightforward:

```python
def extract_text(self, image_path, preprocess=True):
    """Extract text from handwritten image"""
    # Load image
    image = cv2.imread(image_path)

    # Preprocess if requested
    if preprocess:
        image = self.preprocess_image(image)

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Process with TrOCR processor
    pixel_values = self.processor(
        pil_image,
        return_tensors="pt"
    ).pixel_values

    # Move to device (CPU or GPU)
    pixel_values = pixel_values.to(self.device)

    # Generate text
    with torch.no_grad():
        generated_ids = self.model.generate(pixel_values)

    # Decode to text
    text = self.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return {
        'text': text,
        'model': self.model_name,
        'device': self.device
    }
```

### Handling Photo-Specific Challenges

Photos of handwritten notes present several challenges beyond what scanned documents face:

#### 1. Shadow Removal

Shadows from hands, overhead lights, or uneven surfaces can obscure text:

```python
def remove_shadows(image):
    """
    Remove shadows using morphological operations

    Approach:
    1. Convert to LAB color space
    2. Apply morphological closing to estimate background
    3. Subtract background from original
    """
    # Convert to LAB (L = lightness, A/B = color channels)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Estimate background using morphological closing
    # Large kernel removes text, leaving only background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)

    # Subtract background from original
    # This removes shadows while preserving text
    diff = 255 - cv2.absdiff(l, background)

    # Normalize to enhance contrast
    normalized = cv2.normalize(
        diff,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1
    )

    # Merge back with color channels
    lab_corrected = cv2.merge([normalized, a, b])
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

    return result
```

#### 2. Glare and Reflection Handling

Glossy paper or pen ink can create specular reflections:

```python
def reduce_glare(image):
    """
    Reduce glare using bilateral filtering and adaptive thresholding

    Bilateral filter preserves edges while smoothing flat regions
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter
    # This smooths while preserving edges
    bilateral = cv2.bilateralFilter(
        gray,
        d=9,          # Diameter of pixel neighborhood
        sigmaColor=75,  # Filter sigma in color space
        sigmaSpace=75   # Filter sigma in coordinate space
    )

    # Use adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # Constant subtracted from mean
    )

    return thresh
```

**When to apply each technique:**

- **Shadow removal** - When you see dark regions that aren't text
- **Glare reduction** - For photos of glossy notebooks or pen ink
- **CLAHE** (from earlier) - General lighting normalization
- **Combine techniques** - Apply shadow removal → CLAHE → glare reduction for challenging images

#### 3. Perspective Correction

Photos taken at an angle need perspective correction:

```python
def perspective_correction(image, debug=False):
    """
    Correct perspective distortion in angled photos

    Steps:
    1. Edge detection to find document boundaries
    2. Identify four corners
    3. Apply perspective transform to rectangularize
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Find largest rectangular contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            # Found rectangle - apply perspective transform
            pts = approx.reshape(4, 2)

            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = order_points(pts)

            # Compute dimensions
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            # Destination points (perfect rectangle)
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            # Apply perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

            return warped

    return None  # Couldn't find document edges
```

### Batch Processing Handwritten Notes

Process entire directories of handwritten images:

```python
def process_batch(
    input_dir,
    output_dir,
    model_name="microsoft/trocr-base-handwritten",
    use_gpu=True,
    apply_perspective_correction=False,
    use_llm_correction=False,
    max_images=None
):
    """Process multiple handwritten images"""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model once (reuse for all images)
    ocr_model = TrOCRModel(model_name=model_name, use_gpu=use_gpu)

    # Initialize OpenAI client if needed
    openai_client = None
    if use_llm_correction:
        load_dotenv()
        openai_client = OpenAI()

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))

    image_files = sorted(image_files)[:max_images] if max_images else sorted(image_files)

    print(f"Found {len(image_files)} images to process")

    # Process each image
    results = []
    for image_path in image_files:
        result = process_image(
            str(image_path),
            output_dir,
            ocr_model,
            apply_perspective_correction=apply_perspective_correction,
            use_llm_correction=use_llm_correction,
            openai_client=openai_client
        )

        if result:
            results.append(result)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'handwriting_results.csv'), index=False)

    return df
```

### Running the Handwriting OCR Pipeline

**Basic usage:**

```bash
# Create input directory and add handwritten note photos
mkdir handwriting_images
# Add your photos to handwriting_images/

# Run handwriting OCR
python handwriting_ocr.py --input handwriting_images/
```

**Advanced options:**

```bash
# With perspective correction and LLM correction
python handwriting_ocr.py \
  --input handwriting_images/ \
  --perspective-correction \
  --llm-correction

# Use large model for better accuracy
python handwriting_ocr.py \
  --input handwriting_images/ \
  --model microsoft/trocr-large-handwritten

# Process on CPU (no GPU)
python handwriting_ocr.py \
  --input handwriting_images/ \
  --no-gpu
```

**Outputs:**

- `output/handwriting_results.csv` - Processing results
- `output/extracted_handwriting.txt` - Raw OCR text
- `output/corrected_handwriting.txt` - LLM-corrected text
- `output/*_preprocessed.jpg` - Preprocessed images

### Optional: LLM Correction for Handwriting

While TrOCR is significantly better than traditional OCR for handwriting, it still makes errors. LLM correction can provide additional accuracy improvements.

#### When to Apply LLM Correction to Handwriting

**Recommended scenarios:**

- **Production applications** - Where accuracy is critical
- **Legal/medical documents** - Zero tolerance for errors
- **Historical documents** - Where context helps interpret faded text
- **Mixed writing styles** - Cursive, print, and shorthand combined

**Skip LLM correction if:**

- You're just testing/prototyping
- Budget is extremely constrained
- Handwriting is very clear (>95% confidence)
- Errors don't significantly impact usefulness

#### Modified Prompts for Handwritten Text

Handwriting presents unique challenges compared to typed text:

```python
def correct_handwriting_with_llm(client, text, is_cursive=False):
    """
    Correct OCR errors from handwritten text

    Different prompt than printed text correction:
    - Accounts for common handwriting OCR errors
    - Understands cursive writing challenges
    - Preserves personal writing style
    """
    if is_cursive:
        system_prompt = """You are an expert at correcting OCR errors from
        cursive handwritten text. Cursive handwriting OCR commonly struggles with:
        - Connected letters (e.g., "rn" read as "m", "cl" as "d")
        - Inconsistent letter heights (e.g., "a" vs "o", "u" vs "v")
        - Word boundaries (spaces missing or extra)
        - Similar-looking letters (e, l, i confusions)

        Correct the text while:
        - Preserving the original meaning and tone
        - Keeping informal language and personal style
        - Not adding punctuation that wasn't there
        - Not "improving" the writing, just fixing OCR errors"""
    else:
        system_prompt = """You are an expert at correcting OCR errors from
        printed handwriting. Common OCR errors in handwritten text:
        - Similar letters: I/l, O/0, S/5, B/8
        - Incomplete characters from light pen strokes
        - Extra characters from crossed-out text
        - Spacing issues (merged or split words)

        Correct only OCR errors, preserve the original style."""

    user_prompt = f"""Correct any OCR errors in this handwritten text.
    Respond only with the corrected text, no explanations:

{text}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-5",
        messages=messages
    )

    return completion.choices[0].message.content
```

## Part II: DeepSeek-OCR with 4-bit Quantization

*For more details on DeepSeek-OCR, see [DEEPSEEK_4BIT_TEST.md](DEEPSEEK_4BIT_TEST.md)*

DeepSeek-OCR represents a newer approach to OCR using larger vision-language models with quantization for efficiency.

### Key Features

- **4-bit quantization** - Reduces memory from ~12GB to ~3GB
- **Layout-aware extraction** - Preserves document structure
- **Grounding mode** - Provides bounding boxes for text regions
- **Markdown output** - Structured text extraction

### Usage Example

```bash
# Process single image
python test_deepseek_ocr_4bit.py --image test_images/myqueue.png

# Batch processing
python test_deepseek_ocr_4bit.py --input-dir test_images/ --max 5
```

### Comparison: TrOCR vs. DeepSeek-OCR

| Feature | TrOCR | DeepSeek-OCR |
|---------|-------|--------------|
| Model size | 334M-558M params | Much larger (quantized) |
| Memory (GPU) | 2-4GB | ~3GB (with 4-bit quant) |
| Speed | Fast (~2-3s) | Moderate (~5-10s) |
| Layout awareness | No | Yes |
| Structured output | Text only | Markdown, bounding boxes |
| Ease of use | Very easy | Moderate complexity |

## Benchmarking Handwriting OCR

Compare different approaches using the benchmarking tools:

```bash
# Create ground truth template
python ../benchmark.py --input test_images/ --create-template

# Edit ground_truth/*_ref.txt files with correct text

# Run benchmark comparing methods
python ../benchmark.py --input test_images/ --methods trocr trocr_gpt5
```

### Metrics Explained

- **Character Error Rate (CER)** - Percentage of characters incorrectly recognized
- **Word Error Rate (WER)** - Percentage of words with errors
- **Processing time** - Seconds per image

## Datasets for Training and Evaluation

For comprehensive information on handwriting OCR datasets, see [HANDWRITING_DATASETS.md](HANDWRITING_DATASETS.md), which covers:

- IAM Handwriting Database (English)
- RIMES (French)
- CVL Database (German and English)
- And many more...

## Conclusion

Transformer-based models like TrOCR and DeepSeek-OCR represent a significant leap forward in handwriting recognition compared to traditional OCR engines. Key takeaways:

1. **TrOCR excels at handwriting** - Pre-trained transformer models understand context
2. **Preprocessing matters** - CLAHE, shadow removal, and perspective correction improve results
3. **LLM correction is optional** - Adds cost but improves accuracy for critical applications
4. **Hardware requirements** - GPU recommended but CPU works (just slower)
5. **DeepSeek-OCR adds layout awareness** - Better for structured documents

These approaches make it practical to digitize handwritten notes, historical documents, and personal archives at scale.

## Academic References

### TrOCR
Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., ... & Wei, F. (2023). **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models**. *Proceedings of the AAAI Conference on Artificial Intelligence, 37*(11), 13094-13102. [https://arxiv.org/abs/2109.10282](https://arxiv.org/abs/2109.10282)

### GPT-5
OpenAI. (2024). **GPT-4o System Card**. *arXiv preprint arXiv:2410.21276*. [https://arxiv.org/abs/2410.21276](https://arxiv.org/abs/2410.21276)

## Complete Code

All code examples shown in this article are available in this repository:

- `handwriting_ocr.py` - Complete TrOCR implementation
- `test_deepseek_ocr_4bit.py` - DeepSeek-OCR with quantization
- `benchmark.py` - Performance benchmarking tools (in parent directory)

For setup instructions, see [README.md](README.md).



## TODO:

things to add in appropriate section



- [DeepSeek OCR Guide](https://deepseeksguides.com/deepseek-ocr-guide/) - Memory requirements and GPU specifications
