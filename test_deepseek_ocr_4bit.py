#!/usr/bin/env python3
"""
Test DeepSeek-OCR with 4-bit quantization for handwriting recognition

This script demonstrates using DeepSeek-OCR with 4-bit quantization to reduce
memory requirements while maintaining good OCR performance on handwritten text.

Based on the 4-bit quantization approach from:
https://colab.research.google.com/github/Alireza-Akhavan/LLM/blob/main/deepseek_ocr_inference_4bit.ipynb
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import traceback

try:
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"Error: Required dependencies not available: {e}")
    print("\nInstall with:")
    print("  pip install bitsandbytes")
    print("  pip install addict transformers==4.46.3 tokenizers==0.20.3")


class DeepSeekOCR4Bit:
    """DeepSeek-OCR with 4-bit quantization for memory-efficient inference"""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR"):
        """
        Initialize DeepSeek-OCR with 4-bit quantization

        Args:
            model_name: HuggingFace model identifier
        """
        if not DEPS_AVAILABLE:
            raise ImportError(
                "Required dependencies not installed. "
                "Install with: pip install bitsandbytes transformers==4.46.3 tokenizers==0.20.3 addict"
            )

        self.model_name = model_name

        print(f"Loading {model_name} with 4-bit quantization...")
        print("This allows the model to run with much less GPU memory!")

        try:
            # Configure 4-bit quantization
            # Based on the notebook settings that work with T4 GPU
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load model with quantization
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                device_map="auto",
                quantization_config=self.quantization_config,
                torch_dtype=torch.float
            )
            self.model = self.model.eval()

            print("DeepSeek-OCR loaded successfully with 4-bit quantization!")
            print(f"Model is on device: {self.model.device}")

            # Print memory usage if CUDA is available
            if torch.cuda.is_available():
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        except Exception as e:
            print(f"Error loading DeepSeek-OCR: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure you have a HuggingFace token set (HF_TOKEN environment variable)")
            print("2. Ensure you have GPU with CUDA support")
            print("3. Try: pip install --upgrade transformers bitsandbytes")
            raise

    def extract_text(
        self,
        image_path: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        output_path: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 1024,
        crop_mode: bool = False,
        save_results: bool = False
    ) -> Dict:
        """
        Extract text from an image using DeepSeek-OCR

        Args:
            image_path: Path to input image
            prompt: Prompt to use for OCR
                Options:
                - "<image>\n<|grounding|>Convert the document to markdown." (structured)
                - "<image>\n<|grounding|>OCR this image." (general OCR)
                - "<image>\nFree OCR." (without layout)
                - "<image>\nDescribe this image in detail." (general)
            output_path: Directory to save results (default: current directory)
            base_size: Base size for image processing
            image_size: Image size for processing
            crop_mode: Whether to use crop mode
            save_results: Whether to save result files

        Size configurations:
            - Tiny: base_size=512, image_size=512, crop_mode=False
            - Small: base_size=640, image_size=640, crop_mode=False
            - Base: base_size=1024, image_size=1024, crop_mode=False
            - Large: base_size=1280, image_size=1280, crop_mode=False
            - Gundam: base_size=1024, image_size=640, crop_mode=True

        Returns:
            Dictionary with extracted text and metadata
        """
        if output_path is None:
            output_path = str(Path(image_path).parent)

        print(f"\nProcessing: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Config: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")

        try:
            with torch.no_grad():
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=output_path,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=save_results,
                    test_compress=True,
                    eval_mode=False
                )

            # Extract text from result
            # The result format from the notebook shows the text in the output
            extracted_text = str(result) if result else ""

            return {
                'text': extracted_text,
                'model': self.model_name,
                'quantization': '4-bit',
                'image_path': image_path,
                'prompt': prompt,
                'config': {
                    'base_size': base_size,
                    'image_size': image_size,
                    'crop_mode': crop_mode
                }
            }

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR during OCR inference for: {image_path}")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            print(f"{'='*80}\n")
            raise


def test_single_image(
    image_path: str,
    output_dir: str = "output_deepseek_4bit",
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_size: int = 1024,
    image_size: int = 1024,
    crop_mode: bool = False
):
    """
    Test DeepSeek-OCR 4-bit on a single handwriting image

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        prompt: OCR prompt to use
        base_size: Base size for processing
        image_size: Image size for processing
        crop_mode: Whether to use crop mode
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    print("\n" + "="*80)
    print("Initializing DeepSeek-OCR with 4-bit quantization")
    print("="*80)
    ocr_model = DeepSeekOCR4Bit()

    # Process image
    print("\n" + "="*80)
    print("Processing image")
    print("="*80)
    result = ocr_model.extract_text(
        image_path=image_path,
        prompt=prompt,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=True
    )

    # Save results to JSON
    result_file = os.path.join(output_dir, "deepseek_4bit_result.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Save extracted text
    text_file = os.path.join(output_dir, "extracted_text.txt")
    with open(text_file, 'w') as f:
        f.write(result['text'])

    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"Extracted text:\n{result['text']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - JSON: {result_file}")
    print(f"  - Text: {text_file}")

    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Final GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    return result


def test_batch_images(
    input_dir: str,
    output_dir: str = "output_deepseek_4bit",
    max_images: Optional[int] = None,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_size: int = 1024,
    image_size: int = 1024,
    crop_mode: bool = False
):
    """
    Test DeepSeek-OCR 4-bit on multiple handwriting images

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        max_images: Maximum number of images to process
        prompt: OCR prompt to use
        base_size: Base size for processing
        image_size: Image size for processing
        crop_mode: Whether to use crop mode
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)[:max_images] if max_images else sorted(image_files)

    print(f"\nFound {len(image_files)} images to process")

    # Initialize model once for all images
    print("\n" + "="*80)
    print("Initializing DeepSeek-OCR with 4-bit quantization")
    print("="*80)
    ocr_model = DeepSeekOCR4Bit()

    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing image {i}/{len(image_files)}")
        print(f"{'='*80}")

        try:
            result = ocr_model.extract_text(
                image_path=str(image_path),
                prompt=prompt,
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True
            )
            results.append(result)

            print(f"Success! Extracted {len(result['text'])} characters")

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            results.append({
                'image_path': str(image_path),
                'error': str(e)
            })

    # Save all results
    results_file = os.path.join(output_dir, "all_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Batch processing complete!")
    print(f"{'='*80}")
    print(f"Processed {len(results)} images")
    print(f"Results saved to: {results_file}")

    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Final GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test DeepSeek-OCR with 4-bit quantization for handwriting recognition'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Single image to process'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='images',
        help='Input directory containing handwriting images (for batch mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_deepseek_4bit',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Maximum number of images to process in batch mode'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help='OCR prompt to use'
    )
    parser.add_argument(
        '--base-size',
        type=int,
        default=1024,
        help='Base size for image processing (default: 1024)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=1024,
        help='Image size for processing (default: 1024)'
    )
    parser.add_argument(
        '--crop-mode',
        action='store_true',
        help='Enable crop mode'
    )

    args = parser.parse_args()

    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' does not exist")
            return

        test_single_image(
            image_path=args.image,
            output_dir=args.output,
            prompt=args.prompt,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode
        )
    else:
        # Batch mode
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' does not exist")
            return

        test_batch_images(
            input_dir=args.input_dir,
            output_dir=args.output,
            max_images=args.max,
            prompt=args.prompt,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode
        )


if __name__ == '__main__':
    main()
