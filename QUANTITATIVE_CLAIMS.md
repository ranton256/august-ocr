# Quantitative Claims and Metrics in Documentation

This document splits quantitative claims into two categories:

1. **Web-Verifiable Facts** - Can be checked via web search (model specs, API pricing, etc.)
2. **Project-Specific Evaluations** - Need to be measured on our data/hardware

---

## Part 1: Web-Verifiable Facts

These claims can be verified through web searches, official documentation, or published specifications.

### API Pricing (Need to verify for GPT-5)

**Current Claims (GPT-4o):**

- **Lines 287-289 (article_draft.md)**:
  - Input tokens: $2.50 per 1M tokens
  - Output tokens: $10.00 per 1M tokens
- **Action**: Search for GPT-5 API pricing on OpenAI website

**Derived Cost Claims (Need recalculation with GPT-5 pricing):**

- **Line 288**: Typical OCR correction: ~$0.02-0.03 per page
- **Line 289**: 30-page document: ~$0.60-0.90 total
- **Line 925**: Cost per page: $0.005 (TrOCR + GPT-4o)
- **Line 926**: Cost for 100 pages: $0.50
- **Line 941**: 1,000 pages costs only $5
- **Line 1367**: Pytesseract + GPT-4o: $0.005/page
- **Line 1369**: TrOCR + GPT-4o: $0.005/page
- **Line 1558**: Vision API costs: ~$0.01/page vs. $0.005/page
- **Action**: Recalculate based on GPT-5 pricing and token usage

### Model Specifications

**TrOCR Model Sizes (Lines 499-500, article_draft.md):**

- **trocr-base-handwritten**: 334M parameters, ~1GB download
- **trocr-large-handwritten**: 558M parameters, ~2GB download
- **Action**: Verify on HuggingFace model cards
- **Source**: <https://huggingface.co/microsoft/trocr-base-handwritten>
- **Source**: <https://huggingface.co/microsoft/trocr-large-handwritten>

**DeepSeek Model Memory Requirements (DEEPSEEK_4BIT_TEST.md, Lines 11-12):**

- **Full DeepSeek-OCR**: ~24GB+ GPU memory required
- **4-bit Quantized**: ~6-8GB GPU memory (can run on T4 GPU)
- **Action**: Verify on DeepSeek model documentation

### Standard Benchmarks & Research Metrics

**Dataset Split Recommendations (HANDWRITING_DATASETS.md, Lines 444-446):**

- Training: 70-80%
- Validation: 10-15%
- Test: 10-15%
- **Action**: Verify against ML best practices (this is standard, likely correct)

**GPU Speedup Factors (Multiple locations):**

- **Line 45**: GPU Benefit: 5-10x speedup for TrOCR
- **Line 1375**: GPU acceleration: 5-10x speedup for TrOCR
- **Action**: Check TrOCR research papers or benchmarks for typical GPU speedup
- **Note**: This is hardware-dependent but general ranges can be verified

### Confidence Threshold Standards

**Confidence Level Definitions (Lines 1044-1046, article_draft.md):**

- Green: High confidence (>90%)
- Yellow: Medium confidence (70-90%)
- Red: Low confidence (<70%)
- **Action**: Verify if these are standard OCR confidence thresholds or project-specific
- **Note**: These may be arbitrary thresholds we set

---

## Part 2: Project-Specific Evaluations

These claims need to be measured on your specific data, hardware, and use case. They cannot be verified via web search.

### Error Rates on August Anton Dataset

**General Accuracy Claims (Lines 46-47, article_draft.md):**

- Document OCR (Pytesseract): 85-95% (raw OCR), 95-98% (with AI correction)
- Handwriting OCR (TrOCR): 90-95% (raw OCR), 97-99% (with AI correction)
- **Action**: Re-run benchmarks on your dataset with GPT-5
- **Test Script**: `benchmark.py`

**TrOCR vs TrOCR + LLM Error Rates (Lines 922-923, article_draft.md):**

- **Character Error Rate**:
  - TrOCR Alone: 8-15%
  - TrOCR + GPT-4o: 2-6%
  - Improvement: 6-9% reduction
- **Word Error Rate**:
  - TrOCR Alone: 10-20%
  - TrOCR + GPT-4o: 4-10%
  - Improvement: 6-10% reduction
- **Action**: Re-run `benchmark.py` with GPT-5 on same dataset
- **Test Script**: `benchmark.py --ground-truth ground_truth.json`

**Performance Comparison Table (Lines 1364-1369, article_draft.md):**

- **Pytesseract**:
  - CER: 0.15-0.25
  - WER: 0.20-0.35
- **Pytesseract + GPT-4o**:
  - CER: 0.03-0.08
  - WER: 0.05-0.12
- **TrOCR (base)**:
  - CER: 0.08-0.15
  - WER: 0.10-0.20
- **TrOCR + GPT-4o**:
  - CER: 0.02-0.06
  - WER: 0.04-0.10
- **Action**: Re-run full benchmark suite with GPT-5
- **Test Script**: `benchmark.py` with all methods

### Processing Time on Your Hardware

**Document OCR Processing (Line 44, article_draft.md):**

- ~1-2s per page
- **Action**: Time `text_from_pdfs.py` on your hardware

**Handwriting OCR Processing (Line 44, article_draft.md):**

- ~3-5s per page (CPU)
- **Action**: Time `handwriting_ocr.py` on your hardware

**LLM Correction Overhead (Lines 925, 942, article_draft.md):**

- +1-2 seconds per page for LLM correction
- **Action**: Measure API call latency with GPT-5 on your network

**Detailed Processing Times (Lines 1366-1369, article_draft.md):**

- Pytesseract: ~1s/page
- Pytesseract + GPT-4o: ~2s/page
- TrOCR (base): ~3-5s/page (CPU), ~0.5s/page (GPU)
- TrOCR + GPT-4o: ~4-6s/page (CPU), ~1.5s/page (GPU)
- **Action**: Re-measure all processing times with GPT-5
- **Test Method**: Add timing code or use profiling tools

### Preprocessing Impact on Your Data

**Preprocessing Improvement (Line 1738, article_draft.md):**

- Preprocessing impact varies by document quality
- **Action**: Run A/B test with/without preprocessing on your dataset
- **Test Method**: Modify preprocessing pipeline and compare CER/WER
- **Note**: Evaluation on 5 clean typed documents showed minimal improvement (0.07-0.25pp), not 10-20%

### LLM Correction Impact on Your Data

**LLM Correction Improvement (Line 1742, article_draft.md):**

- GPT-4o can reduce error rates by 6-10% at minimal cost
- **Action**: Re-measure with GPT-5 on your dataset
- **Test Method**: Compare CER/WER before/after LLM correction

**Confidence Threshold Validation (Line 858, article_draft.md):**

- Handwriting is very clear (>95% confidence)
- **Action**: Validate if this threshold makes sense for your data
- **Test Method**: Review confidence scores vs. actual accuracy

---

## Validation Checklist

### Web Verification Tasks

- [x] Verify GPT-5 API pricing (input/output token costs)
- [x] Verify TrOCR model sizes on HuggingFace
- [x] Verify DeepSeek model memory requirements
- [x] Check if confidence thresholds (90%, 70%) are standard or arbitrary
- [x] Verify GPU speedup claims in TrOCR literature

### Benchmark Re-run Tasks

- [x] Re-run `benchmark.py` with GPT-5 for all methods
- [x] Measure CER/WER for:
  - [x] Pytesseract alone (CER: 0.015, WER: 0.056 on 5 pages)
  - [x] Pytesseract + GPT-5 (CER: 1.698, WER: 1.722 - note: GPT-5 does editing, not just correction)
  - [x] TrOCR alone (CER: 0.983, WER: 0.999 on typed text - poor performance expected)
  - [x] TrOCR + GPT-5 (CER: 0.983, WER: 0.998 - no improvement due to poor TrOCR output)
- [x] Measure processing times for all methods
- [x] Calculate new cost per page with GPT-5 pricing ($0.0045/page average)
- [x] Validate preprocessing impact (found minimal: 0.07-0.25pp, not 10-20%)
- [x] Measure LLM correction improvement with GPT-5 (note: high CER/WER due to GPT-5 editing/rewriting)

### Documentation Update Tasks

- [x] Update all cost claims with GPT-5 pricing
- [x] Update performance comparison table (Lines 1364-1369) - Updated with actual 5-page results
- [x] Update cost-benefit analysis (Lines 916-946) - Updated with actual results (changed to Pytesseract focus)
- [x] Update accuracy percentages if they change - Updated (Pytesseract 98.5% accuracy, CER 0.015)
- [x] Update processing time claims if they change
- [x] Update error rate improvements if they change - Updated to clarify GPT-5 editing/rewriting behavior

---

## Quick Reference: What to Search vs. What to Measure

| Claim Type | Verification Method | Example |
|------------|---------------------|---------|
| API Pricing | Web search | GPT-5 token costs |
| Model Sizes | Official docs | TrOCR parameter counts |
| Error Rates | Run benchmarks | CER/WER on your data |
| Processing Time | Measure locally | Seconds per page on your hardware |
| Cost per Page | Calculate from pricing | Token usage × API price |
| Accuracy % | Run benchmarks | Accuracy on your dataset |
| Speedup Factors | Research papers | GPU vs CPU performance |
