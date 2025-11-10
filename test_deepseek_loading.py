#!/usr/bin/env python3
"""
Test script to verify DeepSeek-OCR model loading fixes
"""

import sys
from unittest.mock import MagicMock

# Mock the LlamaFlashAttention2 class preemptively
import transformers.models.llama.modeling_llama as llama_module
if not hasattr(llama_module, 'LlamaFlashAttention2'):
    llama_module.LlamaFlashAttention2 = MagicMock

import torch
from transformers import AutoModel, AutoTokenizer

def test_model_loading():
    """Test loading DeepSeek-OCR model"""
    model_name = "deepseek-ai/DeepSeek-OCR"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Testing {model_name} loading on {device}...")
    print("Note: This will download ~5GB on first run")

    try:
        # Load model using AutoModel (not AutoModelForCausalLM)
        # Official docs: https://github.com/deepseek-ai/DeepSeek-OCR
        print("\n1. Loading model...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager"  # Use eager instead of flash_attention_2 for compatibility
        )
        print("✓ Model loaded successfully!")

        # Load tokenizer
        print("\n2. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully!")

        # Move model to device and set precision
        # Official approach: model.eval().cuda().to(torch.bfloat16)
        print(f"\n3. Setting up model for {device}...")
        model.eval()
        if device == "cuda":
            model = model.cuda().to(torch.bfloat16)
        else:
            model = model.to(torch.float32)
        print("✓ Model ready!")

        print("\n" + "="*80)
        print("SUCCESS: All fixes working correctly!")
        print("="*80)
        print("\nFixes applied:")
        print("  ✓ Changed AutoModelForCausalLM → AutoModel (DeepSeek-OCR uses custom config)")
        print("  ✓ Removed torch_dtype parameter (use .to() instead)")
        print("  ✓ Using eager attention for better compatibility")
        print("  ✓ LlamaFlashAttention2 mock in place")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
