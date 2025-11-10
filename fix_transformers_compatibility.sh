#!/bin/bash
# Fix for LlamaFlashAttention2 import error
# This script clears the HuggingFace cache and upgrades transformers

echo "================================================"
echo "Fixing DeepSeek-OCR Transformers Compatibility"
echo "================================================"

echo ""
echo "Step 1: Clearing HuggingFace transformers cache..."
rm -rf ~/.cache/huggingface/modules/transformers_modules/
echo "✓ Cache cleared"

echo ""
echo "Step 2: Clearing workspace cache..."
rm -rf /workspace/.cache/huggingface/modules/transformers_modules/
echo "✓ Workspace cache cleared"

echo ""
echo "Step 3: Upgrading transformers and dependencies..."
pip install --upgrade 'transformers>=4.48.0' 'tokenizers>=0.21.0' 'accelerate>=0.26.0'
echo "✓ Dependencies upgraded"

echo ""
echo "================================================"
echo "Fix complete! You can now run handwriting_ocr.py"
echo "================================================"
