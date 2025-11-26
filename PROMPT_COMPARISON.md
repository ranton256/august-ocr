# OCR Correction Prompt Comparison

## Summary

The improved prompt significantly outperforms the original GPT-5 prompt, achieving error rates comparable to or better than raw Pytesseract.

## Results Overview

| Method | Avg CER | Avg WER | Avg Time | Notes |
|--------|---------|---------|----------|-------|
| **pytesseract** (raw) | 0.082 | 0.196 | 3.28s | Baseline OCR |
| **pytesseract_gpt5** (original) | 1.209 | 1.302 | 118.98s | ❌ Too much editing/rewriting |
| **pytesseract_gpt5_improved** | 0.079 | 0.177 | 259.67s | ✅ Best accuracy |

## Key Findings

1. **Improved prompt reduces errors dramatically**: 
   - CER reduced from 1.209 → 0.079 (93% reduction)
   - WER reduced from 1.302 → 0.177 (86% reduction)

2. **Improved prompt outperforms raw OCR**:
   - 4% better CER (0.079 vs 0.082)
   - 10% better WER (0.177 vs 0.196)

3. **Processing time**: Improved prompt takes longer (259.67s vs 118.98s), likely due to more careful processing.

## Detailed Results for 5 Pages with Ground Truth

| Image | Raw Pytesseract | Original GPT-5 | Improved GPT-5 |
|-------|----------------|-----------------|----------------|
| IMG_8478 | CER: 0.336, WER: 0.75 | CER: 4.409, WER: 4.5 ❌ | CER: 0.336, WER: 0.75 ✅ |
| IMG_8479 | CER: 0.028, WER: 0.107 | CER: 0.525, WER: 0.587 ❌ | CER: 0.016, WER: 0.043 ✅ |
| IMG_8480 | CER: 0.008, WER: 0.022 | CER: 0.377, WER: 0.47 ❌ | CER: 0.009, WER: 0.02 ✅ |
| IMG_8481 | CER: 0.026, WER: 0.061 | CER: 0.271, WER: 0.39 ❌ | CER: 0.025, WER: 0.049 ✅ |
| IMG_8482 | CER: 0.011, WER: 0.041 | CER: 0.465, WER: 0.564 ❌ | CER: 0.008, WER: 0.025 ✅ |

## Prompt Comparison

### Original Prompt
```
System: "You are a helpful assistant who is an expert on the English language, skilled in vocabulary, pronunciation, and grammar."

User: "Correct any typos caused by bad OCR in this text, using common sense reasoning, responding only with the corrected text: [text]"
```

**Issues:**
- Too vague about preserving structure
- No explicit instruction to avoid rewriting
- Model interprets "common sense reasoning" as license to edit/improve text

### Improved Prompt
```
System: "You are an expert at correcting OCR errors in scanned documents. Your task is to fix OCR mistakes while preserving the original text structure, formatting, and meaning exactly as written."

User: "The following text was extracted from a scanned document using OCR. It contains OCR errors that need to be corrected.

IMPORTANT INSTRUCTIONS:
- Fix ONLY OCR errors (misspellings, character misrecognitions, punctuation mistakes)
- Preserve the EXACT original structure, line breaks, spacing, and formatting
- Do NOT rewrite, reformat, or improve the text
- Do NOT add explanations, suggestions, or commentary
- Do NOT change the writing style or voice
- Return ONLY the corrected text, nothing else

OCR text to correct:
[text]"
```

**Improvements:**
- Explicit instructions to preserve structure and formatting
- Clear prohibition against rewriting or improving
- Specific focus on OCR errors only
- Multiple emphases on preservation

## Cost Analysis

- **Total cost for 30 images**: $0.5560
- **Average cost per page**: $0.0093
- **Total input tokens**: 48,890
- **Total output tokens**: 49,484

The improved prompt generates similar token counts, so cost is comparable.

## Recommendation

**Use the improved prompt** for OCR correction tasks. It:
- ✅ Achieves better accuracy than raw OCR
- ✅ Preserves document structure and formatting
- ✅ Avoids unwanted editing/rewriting
- ✅ Provides consistent, reliable corrections

The original prompt should be avoided as it causes GPT-5 to over-edit, resulting in higher error rates than raw OCR.

