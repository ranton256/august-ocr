#!/usr/bin/env python3
"""
Handwriting OCR pipeline using TrOCR (Transformer-based OCR)

This module provides handwriting recognition capabilities for photos of handwritten notes.
It uses Microsoft's TrOCR model which is specifically designed for handwritten text.

For tutorial purposes, this is structured to allow easy swapping of different OCR backends.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# Vision/OCR imports
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("Warning: TrOCR not available. Install with: pip install transformers torch")

# LLM imports for optional correction
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class HandwritingOCR:
    """Handwriting OCR using TrOCR or other vision models"""

    def __init__(self, model_name: str = "microsoft/trocr-large-handwritten", use_gpu: bool = True):
        """
        Initialize the handwriting OCR model

        Args:
            model_name: HuggingFace model identifier
            use_gpu: Whether to use GPU if available
        """
        if not TROCR_AVAILABLE:
            raise ImportError(
                "TrOCR dependencies not installed. "
                "Install with: pip install transformers torch torchvision"
            )

        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        print(f"Loading {model_name} on {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better handwriting recognition

        Unlike document OCR, handwriting from photos needs different preprocessing:
        - Perspective correction for angled shots
        - Lighting normalization
        - Less aggressive thresholding to preserve stroke details

        Args:
            image: Input image as numpy array (BGR format from cv2)

        Returns:
            Preprocessed image
        """
        # Convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This helps with varied lighting conditions in photos
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return image

    def extract_text(self, image_path: str, preprocess: bool = True) -> Dict:
        """
        Extract text from a handwritten image

        Args:
            image_path: Path to image file
            preprocess: Whether to apply preprocessing

        Returns:
            Dictionary with 'text' and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Preprocess if requested
        if preprocess:
            image = self.preprocess_image(image)

        # Convert to PIL Image for TrOCR
        pil_image = Image.fromarray(image)

        # Process with TrOCR
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {
            'text': text,
            'model': self.model_name,
            'device': self.device,
            'image_size': pil_image.size
        }


def perspective_correction(image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Detect and correct perspective distortion in photos of documents/notes

    This is useful when photos are taken at an angle rather than straight-on.

    Args:
        image: Input image
        debug: Whether to show intermediate steps

    Returns:
        Corrected image or None if document edges couldn't be detected
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 75, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Find the largest rectangular contour
    doc_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        if debug:
            print("Could not find document contour")
        return None

    # Get the four corners
    pts = doc_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left will have smallest sum, bottom-right will have largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right will have smallest difference, bottom-left will have largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Compute width and height of new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def process_image(
    image_path: str,
    output_dir: str,
    ocr_model: HandwritingOCR,
    apply_perspective_correction: bool = False,
    use_llm_correction: bool = False,
    openai_client: Optional[OpenAI] = None
) -> Dict:
    """
    Process a single handwritten image

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        ocr_model: Initialized HandwritingOCR model
        apply_perspective_correction: Whether to correct perspective
        use_llm_correction: Whether to use LLM for correction
        openai_client: OpenAI client (required if use_llm_correction=True)

    Returns:
        Dictionary with results
    """
    print(f"\nProcessing: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None

    # Apply perspective correction if requested
    if apply_perspective_correction:
        corrected = perspective_correction(img, debug=True)
        if corrected is not None:
            img = corrected
            print("Applied perspective correction")

    # Save preprocessed image
    image_name = Path(image_path).stem
    preprocessed_path = os.path.join(output_dir, f"{image_name}_preprocessed.jpg")
    cv2.imwrite(preprocessed_path, img)

    # Extract text with OCR
    result = ocr_model.extract_text(image_path, preprocess=True)
    extracted_text = result['text']
    print(f"Extracted: {extracted_text[:100]}...")

    # Optional LLM correction
    corrected_text = extracted_text
    if use_llm_correction and openai_client:
        print("Applying LLM correction...")
        corrected_text = correct_with_llm(openai_client, extracted_text, is_handwriting=True)

    return {
        'image_path': image_path,
        'preprocessed_path': preprocessed_path,
        'extracted': extracted_text,
        'corrected': corrected_text,
        'model': result['model']
    }


def correct_with_llm(client: OpenAI, text: str, is_handwriting: bool = True) -> str:
    """
    Use GPT-4o to correct OCR errors

    Args:
        client: OpenAI client
        text: Text to correct
        is_handwriting: Whether text is from handwriting (affects prompt)

    Returns:
        Corrected text
    """
    if is_handwriting:
        system_prompt = """You are a helpful assistant who is an expert at correcting
        OCR errors from handwritten text. Handwriting OCR often struggles with:
        - Similar looking letters (e/a, n/u, i/l, etc.)
        - Word boundaries in cursive writing
        - Incomplete or unclear characters

        Correct the text while preserving the original meaning and tone."""

        user_prompt = f"""Correct any OCR errors in this handwritten text using common sense
        reasoning. Respond only with the corrected text, no explanations:

{text}"""
    else:
        system_prompt = """You are a helpful assistant who is an expert on the English language,
        including spelling, grammar, and common OCR errors."""

        user_prompt = f"""Correct any typos caused by bad OCR in this text, using common sense
        reasoning, responding only with the corrected text:

{text}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return completion.choices[0].message.content


def process_batch(
    input_dir: str,
    output_dir: str,
    model_name: str = "microsoft/trocr-large-handwritten",
    use_gpu: bool = True,
    apply_perspective_correction: bool = False,
    use_llm_correction: bool = False,
    max_images: Optional[int] = None
) -> pd.DataFrame:
    """
    Process multiple handwritten images in batch

    Args:
        input_dir: Directory containing images
        output_dir: Directory to save results
        model_name: HuggingFace model to use
        use_gpu: Whether to use GPU
        apply_perspective_correction: Whether to correct perspective
        use_llm_correction: Whether to use LLM correction
        max_images: Maximum number of images to process (None for all)

    Returns:
        DataFrame with results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OCR model
    ocr_model = HandwritingOCR(model_name=model_name, use_gpu=use_gpu)

    # Initialize OpenAI client if needed
    openai_client = None
    if use_llm_correction:
        if not OPENAI_AVAILABLE:
            print("Warning: OpenAI not available, skipping LLM correction")
            use_llm_correction = False
        else:
            load_dotenv()
            openai_client = OpenAI()

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

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

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = os.path.join(output_dir, 'handwriting_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Save extracted and corrected text files
    extracted_path = os.path.join(output_dir, 'extracted_handwriting.txt')
    corrected_path = os.path.join(output_dir, 'corrected_handwriting.txt')

    with open(extracted_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Image: {row['image_path']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(row['extracted'] + '\n')

    with open(corrected_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Image: {row['image_path']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(row['corrected'] + '\n')

    print(f"Extracted text saved to {extracted_path}")
    print(f"Corrected text saved to {corrected_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from handwritten images using TrOCR'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='handwriting_images',
        help='Input directory containing handwritten images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/trocr-large-handwritten',
        help='HuggingFace model name (default: microsoft/trocr-large-handwritten)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    parser.add_argument(
        '--perspective-correction',
        action='store_true',
        help='Apply perspective correction for angled photos'
    )
    parser.add_argument(
        '--llm-correction',
        action='store_true',
        help='Use GPT-4o to correct OCR errors (requires OPENAI_API_KEY)'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        print("\nTo use this script:")
        print(f"1. Create a directory: mkdir {args.input}")
        print(f"2. Add handwritten note photos to: {args.input}/")
        print("3. Run this script again")
        return

    # Process images
    df = process_batch(
        input_dir=args.input,
        output_dir=args.output,
        model_name=args.model,
        use_gpu=not args.no_gpu,
        apply_perspective_correction=args.perspective_correction,
        use_llm_correction=args.llm_correction,
        max_images=args.max
    )

    print(f"\n✓ Processed {len(df)} images successfully!")
    print(f"✓ Results saved to {args.output}/")


if __name__ == '__main__':
    main()
