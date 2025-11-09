#!/usr/bin/env python3
"""
OCR Performance Benchmarking Tool

This module provides tools for comparing different OCR methods:
- Pytesseract (traditional OCR)
- DeepSeek-OCR (state-of-the-art vision-language OCR, 97% accuracy)
- With and without LLM correction

Metrics calculated:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Processing time
- Cost estimation for LLM correction
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import pandas as pd
import numpy as np
from Levenshtein import distance as levenshtein_distance

# Import OCR modules
try:
    import pytesseract
    from text_from_pdfs import preprocess_image, extract_text
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available")

try:
    from handwriting_ocr import DeepSeekOCR
    DEEPSEEK_OCR_AVAILABLE = True
except ImportError:
    DEEPSEEK_OCR_AVAILABLE = False
    print("Warning: DeepSeek-OCR not available")


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER)

    CER = (substitutions + deletions + insertions) / total characters in reference

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        CER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    if not reference:
        return 1.0 if hypothesis else 0.0

    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER)

    WER = (substitutions + deletions + insertions) / total words in reference

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison

    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation (optional)

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


class OCRBenchmark:
    """Benchmark different OCR methods"""

    def __init__(self, ground_truth_file: Optional[str] = None):
        """
        Initialize benchmark

        Args:
            ground_truth_file: JSON file with ground truth annotations
                Format: {"image_path": "ground truth text", ...}
        """
        self.ground_truth = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded {len(self.ground_truth)} ground truth annotations")

        self.results = []

    def benchmark_pytesseract(self, image_path: str) -> Dict:
        """
        Benchmark pytesseract on an image

        Args:
            image_path: Path to image

        Returns:
            Results dictionary
        """
        if not PYTESSERACT_AVAILABLE:
            return {"error": "pytesseract not available"}

        import cv2

        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        start_time = time.time()
        preprocessed = preprocess_image(img)
        extracted = extract_text(preprocessed)
        processing_time = time.time() - start_time

        return {
            'method': 'pytesseract',
            'text': extracted,
            'processing_time': processing_time
        }

    def benchmark_deepseek_ocr(
        self,
        image_path: str,
        model: Optional[DeepSeekOCR] = None
    ) -> Dict:
        """
        Benchmark DeepSeek-OCR on an image

        Args:
            image_path: Path to image
            model: Pre-initialized DeepSeekOCR model (for efficiency)

        Returns:
            Results dictionary
        """
        if not DEEPSEEK_OCR_AVAILABLE:
            return {"error": "DeepSeek-OCR not available"}

        # Initialize model if not provided
        close_model = False
        if model is None:
            model = DeepSeekOCR()
            close_model = True

        start_time = time.time()
        result = model.extract_text(image_path, preprocess=True)
        processing_time = time.time() - start_time

        return {
            'method': 'deepseek-ocr',
            'text': result['text'],
            'processing_time': processing_time,
            'model': result['model']
        }

    def benchmark_image(
        self,
        image_path: str,
        methods: List[str] = ['pytesseract', 'deepseek-ocr'],
        deepseek_model: Optional[DeepSeekOCR] = None
    ) -> List[Dict]:
        """
        Benchmark multiple OCR methods on a single image

        Args:
            image_path: Path to image
            methods: List of methods to benchmark
            deepseek_model: Pre-initialized DeepSeek-OCR model

        Returns:
            List of result dictionaries
        """
        print(f"\nBenchmarking: {image_path}")

        results = []
        ground_truth = self.ground_truth.get(image_path, None)

        for method in methods:
            if method == 'pytesseract':
                result = self.benchmark_pytesseract(image_path)
            elif method == 'deepseek-ocr':
                result = self.benchmark_deepseek_ocr(image_path, deepseek_model)
            else:
                print(f"Unknown method: {method}")
                continue

            if 'error' in result:
                print(f"  {method}: {result['error']}")
                continue

            # Calculate metrics if ground truth available
            if ground_truth:
                result['cer'] = calculate_cer(
                    normalize_text(ground_truth),
                    normalize_text(result['text'])
                )
                result['wer'] = calculate_wer(
                    normalize_text(ground_truth),
                    normalize_text(result['text'])
                )
                print(f"  {method}: CER={result['cer']:.3f}, WER={result['wer']:.3f}, "
                      f"Time={result['processing_time']:.2f}s")
            else:
                print(f"  {method}: Time={result['processing_time']:.2f}s "
                      f"(no ground truth)")

            result['image_path'] = image_path
            result['has_ground_truth'] = ground_truth is not None
            results.append(result)

        return results

    def benchmark_directory(
        self,
        input_dir: str,
        methods: List[str] = ['pytesseract', 'deepseek-ocr'],
        max_images: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Benchmark OCR methods on all images in a directory

        Args:
            input_dir: Directory containing images
            methods: List of methods to benchmark
            max_images: Maximum number of images to process

        Returns:
            DataFrame with all results
        """
        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

        image_files = sorted(image_files)[:max_images] if max_images else sorted(image_files)
        print(f"Found {len(image_files)} images")

        # Initialize DeepSeek-OCR model once for efficiency
        deepseek_model = None
        if 'deepseek-ocr' in methods and DEEPSEEK_OCR_AVAILABLE:
            deepseek_model = DeepSeekOCR()

        # Benchmark each image
        all_results = []
        for image_path in image_files:
            results = self.benchmark_image(
                str(image_path),
                methods=methods,
                deepseek_model=deepseek_model
            )
            all_results.extend(results)

        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        return df

    def generate_report(self, df: pd.DataFrame, output_file: str = 'benchmark_report.md'):
        """
        Generate a markdown report from benchmark results

        Args:
            df: DataFrame with benchmark results
            output_file: Path to save report
        """
        with open(output_file, 'w') as f:
            f.write("# OCR Benchmark Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            # Summary statistics by method
            f.write("## Summary by Method\n\n")

            for method in df['method'].unique():
                method_df = df[df['method'] == method]

                f.write(f"### {method}\n\n")
                f.write(f"- Images processed: {len(method_df)}\n")
                f.write(f"- Average processing time: {method_df['processing_time'].mean():.2f}s\n")

                if 'cer' in method_df.columns:
                    method_with_gt = method_df[method_df['has_ground_truth']]
                    if len(method_with_gt) > 0:
                        f.write(f"- Average CER: {method_with_gt['cer'].mean():.3f}\n")
                        f.write(f"- Average WER: {method_with_gt['wer'].mean():.3f}\n")
                        f.write(f"- Best CER: {method_with_gt['cer'].min():.3f}\n")
                        f.write(f"- Worst CER: {method_with_gt['cer'].max():.3f}\n")

                f.write("\n")

            # Detailed results table
            f.write("## Detailed Results\n\n")

            if 'cer' in df.columns:
                display_cols = ['image_path', 'method', 'cer', 'wer', 'processing_time']
                display_df = df[display_cols].copy()
                display_df['cer'] = display_df['cer'].round(3)
                display_df['wer'] = display_df['wer'].round(3)
                display_df['processing_time'] = display_df['processing_time'].round(2)
            else:
                display_cols = ['image_path', 'method', 'processing_time']
                display_df = df[display_cols].copy()
                display_df['processing_time'] = display_df['processing_time'].round(2)

            f.write(display_df.to_markdown(index=False))
            f.write("\n\n")

            # Cost estimation
            f.write("## Cost Estimation\n\n")
            f.write("Assuming GPT-4o pricing (~$0.005 per page for correction):\n\n")

            num_images = len(df['image_path'].unique())
            llm_cost = num_images * 0.005

            f.write(f"- Total images: {num_images}\n")
            f.write(f"- Estimated LLM correction cost: ${llm_cost:.2f}\n")
            f.write("\n")

        print(f"\nReport saved to {output_file}")


def create_ground_truth_template(image_dir: str, output_file: str = 'ground_truth.json'):
    """
    Create a template JSON file for ground truth annotations

    Args:
        image_dir: Directory containing images
        output_file: Path to save template
    """
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    # Create template
    template = {}
    for image_path in sorted(image_files):
        template[str(image_path)] = ""

    # Save
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Ground truth template saved to {output_file}")
    print(f"Fill in the correct text for each image, then use with --ground-truth flag")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark OCR methods and calculate accuracy metrics'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        default=None,
        help='JSON file with ground truth annotations'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['pytesseract', 'deepseek-ocr'],
        choices=['pytesseract', 'deepseek-ocr'],
        help='OCR methods to benchmark'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.csv',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--report',
        type=str,
        default='benchmark_report.md',
        help='Output markdown report file'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )
    parser.add_argument(
        '--create-template',
        action='store_true',
        help='Create ground truth template and exit'
    )

    args = parser.parse_args()

    # Create ground truth template if requested
    if args.create_template:
        create_ground_truth_template(args.input)
        return

    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return

    # Run benchmark
    benchmark = OCRBenchmark(ground_truth_file=args.ground_truth)
    df = benchmark.benchmark_directory(
        args.input,
        methods=args.methods,
        max_images=args.max
    )

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Generate report
    benchmark.generate_report(df, output_file=args.report)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        print(f"\n{method}:")
        print(f"  Processed: {len(method_df)} images")
        print(f"  Avg time: {method_df['processing_time'].mean():.2f}s")

        if 'cer' in method_df.columns:
            method_with_gt = method_df[method_df['has_ground_truth']]
            if len(method_with_gt) > 0:
                print(f"  Avg CER: {method_with_gt['cer'].mean():.3f}")
                print(f"  Avg WER: {method_with_gt['wer'].mean():.3f}")


if __name__ == '__main__':
    # Check for Levenshtein package
    try:
        import Levenshtein
    except ImportError:
        print("Error: python-Levenshtein package required for benchmarking")
        print("Install with: pip install python-Levenshtein")
        exit(1)

    main()
