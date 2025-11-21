#!/usr/bin/env python3
"""
OCR Performance Benchmarking Tool

This module provides tools for comparing different OCR methods:
- Pytesseract (traditional OCR)
- TrOCR (transformer-based OCR for handwriting)
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
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from handwriting_ocr import TrOCRModel
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
    from handwriting_ocr import TrOCRModel, correct_with_llm
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("Warning: TrOCR not available")

try:
    from openai import OpenAI
    from text_from_pdfs import ask_the_english_prof
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available")


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

    def __init__(self, ground_truth_file: Optional[str] = None, openai_client: Optional[OpenAI] = None):
        """
        Initialize benchmark

        Args:
            ground_truth_file: Deprecated - kept for compatibility but not used.
                Ground truth is now loaded from individual files in ground_truth/ directory.
            openai_client: OpenAI client for LLM correction (optional)
        """
        # Ground truth is loaded on-demand from files, not from JSON
        self.openai_client = openai_client
        self.results = []
        
    def _load_ground_truth(self, image_path: str) -> Optional[str]:
        """
        Load ground truth text from a file based on image path.
        
        Looks for: ground_truth/{image_basename}_ref.txt
        Example: images/IMG_8479.jpg -> ground_truth/IMG_8479_ref.txt
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Ground truth text if file exists, None otherwise
        """
        # Extract basename and convert .jpg/.png/etc to _ref.txt
        image_basename = Path(image_path).stem  # Gets "IMG_8479" from "images/IMG_8479.jpg"
        ground_truth_path = Path("ground_truth") / f"{image_basename}_ref.txt"
        
        if ground_truth_path.exists():
            try:
                with open(ground_truth_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Skip header lines that start with # (comments)
                    text_lines = [line.rstrip('\n') for line in lines if line.strip() and not line.strip().startswith('#')]
                    return '\n'.join(text_lines)
            except Exception as e:
                print(f"Warning: Could not load ground truth from {ground_truth_path}: {e}")
                return None
        return None

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

    def benchmark_trocr(
        self,
        image_path: str,
        model: Optional['TrOCRModel'] = None
    ) -> Dict:
        """
        Benchmark TrOCR on an image

        Args:
            image_path: Path to image
            model: Pre-initialized TrOCRModel (for efficiency)

        Returns:
            Results dictionary
        """
        if not TROCR_AVAILABLE:
            return {"error": "TrOCR not available"}

        # Initialize model if not provided
        close_model = False
        if model is None:
            model = TrOCRModel()
            close_model = True

        start_time = time.time()
        result = model.extract_text(image_path, preprocess=True)
        processing_time = time.time() - start_time

        return {
            'method': 'trocr',
            'text': result['text'],
            'processing_time': processing_time,
            'model': result['model']
        }

    def apply_llm_correction(self, text: str, is_handwriting: bool = False) -> Tuple[str, Dict]:
        """
        Apply LLM correction to text and return corrected text with token usage

        Args:
            text: Text to correct
            is_handwriting: Whether text is from handwriting OCR

        Returns:
            Tuple of (corrected_text, token_info)
        """
        if not OPENAI_AVAILABLE or not self.openai_client:
            return text, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}

        start_time = time.time()
        if is_handwriting:
            corrected = correct_with_llm(self.openai_client, text, is_handwriting=True)
        else:
            corrected = ask_the_english_prof(self.openai_client, text)
        llm_time = time.time() - start_time

        # Estimate token usage (rough approximation: 1 token ≈ 4 characters)
        # GPT-5 pricing: $1.25 per 1M input tokens, $10.00 per 1M output tokens
        input_tokens = len(text) // 4
        output_tokens = len(corrected) // 4
        cost = (input_tokens / 1_000_000 * 1.25) + (output_tokens / 1_000_000 * 10.00)

        return corrected, {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'llm_time': llm_time
        }

    def benchmark_pytesseract_gpt5(self, image_path: str) -> Dict:
        """Benchmark pytesseract with GPT-5 correction"""
        if not PYTESSERACT_AVAILABLE:
            return {"error": "pytesseract not available"}
        if not OPENAI_AVAILABLE or not self.openai_client:
            return {"error": "OpenAI client not available"}

        import cv2

        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        start_time = time.time()
        preprocessed = preprocess_image(img)
        extracted = extract_text(preprocessed)
        ocr_time = time.time() - start_time

        # Apply LLM correction
        corrected, token_info = self.apply_llm_correction(extracted, is_handwriting=False)
        total_time = ocr_time + token_info['llm_time']

        result = {
            'method': 'pytesseract_gpt5',
            'text': corrected,
            'processing_time': total_time,
            'ocr_time': ocr_time,
            'llm_time': token_info['llm_time'],
            'input_tokens': token_info['input_tokens'],
            'output_tokens': token_info['output_tokens'],
            'cost': token_info['cost']
        }
        return result

    def benchmark_trocr_gpt5(self, image_path: str, model: Optional['TrOCRModel'] = None) -> Dict:
        """Benchmark TrOCR with GPT-5 correction"""
        if not TROCR_AVAILABLE:
            return {"error": "TrOCR not available"}
        if not OPENAI_AVAILABLE or not self.openai_client:
            return {"error": "OpenAI client not available"}

        # Initialize model if not provided
        if model is None:
            model = TrOCRModel()

        start_time = time.time()
        result = model.extract_text(image_path, preprocess=True)
        ocr_time = time.time() - start_time

        # Apply LLM correction
        corrected, token_info = self.apply_llm_correction(result['text'], is_handwriting=True)
        total_time = ocr_time + token_info['llm_time']

        return {
            'method': 'trocr_gpt5',
            'text': corrected,
            'processing_time': total_time,
            'ocr_time': ocr_time,
            'llm_time': token_info['llm_time'],
            'input_tokens': token_info['input_tokens'],
            'output_tokens': token_info['output_tokens'],
            'cost': token_info['cost'],
            'model': result['model']
        }

    def benchmark_image(
        self,
        image_path: str,
        methods: List[str] = ['pytesseract', 'trocr'],
        trocr_model: Optional['TrOCRModel'] = None
    ) -> List[Dict]:
        """
        Benchmark multiple OCR methods on a single image

        Args:
            image_path: Path to image
            methods: List of methods to benchmark
            trocr_model: Pre-initialized TrOCR model

        Returns:
            List of result dictionaries
        """
        print(f"\nBenchmarking: {image_path}")

        results = []
        ground_truth = self._load_ground_truth(image_path)

        for method in methods:
            if method == 'pytesseract':
                result = self.benchmark_pytesseract(image_path)
            elif method == 'trocr':
                result = self.benchmark_trocr(image_path, trocr_model)
            elif method == 'pytesseract_gpt5':
                result = self.benchmark_pytesseract_gpt5(image_path)
            elif method == 'trocr_gpt5':
                result = self.benchmark_trocr_gpt5(image_path, trocr_model)
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
        methods: List[str] = ['pytesseract', 'trocr'],
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

        # Initialize TrOCR model once for efficiency
        trocr_model = None
        if 'trocr' in methods and TROCR_AVAILABLE:
            trocr_model = TrOCRModel()

        # Benchmark each image
        all_results = []
        for image_path in image_files:
            results = self.benchmark_image(
                str(image_path),
                methods=methods,
                trocr_model=trocr_model
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

            if len(df) == 0 or 'method' not in df.columns:
                f.write("No results to summarize.\n\n")
                return

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
            f.write("GPT-5 pricing: $1.25 per 1M input tokens, $10.00 per 1M output tokens\n\n")

            num_images = len(df['image_path'].unique())
            
            # Calculate actual costs from token usage if available
            if 'cost' in df.columns:
                llm_methods = df[df['method'].str.contains('gpt5', na=False)]
                if len(llm_methods) > 0:
                    total_cost = llm_methods['cost'].sum()
                    avg_cost_per_page = llm_methods['cost'].mean()
                    f.write(f"- Total images: {num_images}\n")
                    f.write(f"- Total LLM correction cost: ${total_cost:.4f}\n")
                    f.write(f"- Average cost per page: ${avg_cost_per_page:.4f}\n")
                    if 'input_tokens' in llm_methods.columns:
                        total_input = llm_methods['input_tokens'].sum()
                        total_output = llm_methods['output_tokens'].sum()
                        f.write(f"- Total input tokens: {total_input:,}\n")
                        f.write(f"- Total output tokens: {total_output:,}\n")
                else:
                    f.write(f"- Total images: {num_images}\n")
                    f.write("- No LLM correction methods in results\n")
            else:
                f.write(f"- Total images: {num_images}\n")
                f.write("- No cost data available (no LLM methods used)\n")
            f.write("\n")

        print(f"\nReport saved to {output_file}")


def create_ground_truth_template(image_dir: str, output_file: str = None):
    """
    Create template text files for ground truth annotations in ground_truth/ directory

    Args:
        image_dir: Directory containing images
        output_file: Deprecated - kept for compatibility but not used
    """
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    # Create ground_truth directory if it doesn't exist
    ground_truth_dir = Path("ground_truth")
    ground_truth_dir.mkdir(exist_ok=True)

    # Create template files
    created_count = 0
    for image_path in sorted(image_files):
        image_basename = Path(image_path).stem
        ground_truth_file = ground_truth_dir / f"{image_basename}_ref.txt"
        
        # Only create if it doesn't exist
        if not ground_truth_file.exists():
            with open(ground_truth_file, 'w', encoding='utf-8') as f:
                f.write(f"# Ground Truth for {image_path}\n")
                f.write("#\n")
                f.write("# Enter the correct text from this image below:\n")
                f.write("#\n")
                f.write("\n")
            created_count += 1

    print(f"Created {created_count} ground truth template files in {ground_truth_dir}/")
    print(f"Edit the *_ref.txt files to add the correct text for each image")


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
        help='Deprecated: Ground truth is now automatically loaded from ground_truth/*_ref.txt files'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['pytesseract', 'trocr'],
        choices=['pytesseract', 'trocr', 'pytesseract_gpt5', 'trocr_gpt5'],
        help='OCR methods to benchmark'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Enable LLM correction (requires OPENAI_API_KEY)'
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

    # Initialize OpenAI client if needed
    openai_client = None
    if args.use_llm or any('gpt5' in m for m in args.methods):
        if not OPENAI_AVAILABLE:
            print("Error: OpenAI not available. Install openai package and set OPENAI_API_KEY")
            return
        try:
            # Load environment variables from .env file
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # dotenv not available, try without it
            
            openai_client = OpenAI()
            print("OpenAI client initialized")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return

    # Run benchmark (ground truth is loaded automatically from ground_truth/*_ref.txt files)
    benchmark = OCRBenchmark(ground_truth_file=None, openai_client=openai_client)
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
