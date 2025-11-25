# Preprocessing Impact Evaluation

## Summary

**Finding:** Preprocessing provides **minimal improvement** (0.07-0.25 percentage points) for these typed documents, not the claimed 10-20% improvement.

## Methodology

Compared Pytesseract OCR with and without preprocessing on 5 pages with ground truth:

- **With preprocessing**: Grayscale conversion → noise removal (median blur) → thresholding (OTSU) → region-based extraction
- **Without preprocessing**: Grayscale for region detection only → region-based extraction (no thresholding/noise removal)

Both methods use the same region-based extraction approach for fair comparison.

## Results

### Average Performance (5 pages)

| Metric | With Preprocessing | Without Preprocessing | Difference |
|--------|-------------------|----------------------|------------|
| **CER** | 0.0149 (1.49%) | 0.0156 (1.56%) | **-0.07pp** (slightly better) |
| **WER** | 0.0562 (5.62%) | 0.0586 (5.86%) | **-0.25pp** (slightly better) |
| **Time** | 3.51s/page | 3.15s/page | +0.37s (slightly slower) |

### Per-Page Results

| Page | CER (with) | CER (without) | CER Diff | WER (with) | WER (without) | WER Diff |
|------|------------|---------------|----------|------------|---------------|----------|
| IMG_8478 | 0.0219 | 0.0219 | 0.00pp | 0.0833 | 0.0833 | 0.00pp |
| IMG_8479 | 0.0218 | 0.0230 | **-0.12pp** | 0.0951 | 0.1049 | **-0.98pp** |
| IMG_8480 | 0.0036 | 0.0026 | +0.10pp | 0.0162 | 0.0162 | 0.00pp |
| IMG_8481 | 0.0162 | 0.0198 | **-0.36pp** | 0.0433 | 0.0493 | **-0.60pp** |
| IMG_8482 | 0.0112 | 0.0109 | +0.03pp | 0.0429 | 0.0393 | +0.36pp |

## Analysis

1. **Minimal overall impact**: Preprocessing improves CER by only 0.07 percentage points and WER by 0.25 percentage points on average.

2. **Inconsistent per-page results**:
   - Preprocessing helps on pages 2 and 4 (IMG_8479, IMG_8481)
   - Preprocessing slightly hurts on pages 3 and 5 (IMG_8480, IMG_8482)
   - No difference on page 1 (IMG_8478)

3. **Time cost**: Preprocessing adds ~0.37s per page (10.5% slower).

4. **Document quality matters**: These are relatively clean typed documents. Preprocessing may provide more benefit for:
   - Scanned documents with noise
   - Aged/faded documents
   - Documents with poor contrast
   - Handwritten documents (different preprocessing pipeline)

## Conclusion

**For these typed documents, preprocessing provides minimal benefit** (0.07-0.25pp improvement), not the claimed 10-20%. The documents are already clean enough that preprocessing doesn't significantly improve OCR accuracy.

**Recommendation**:

- For clean typed documents: Preprocessing may not be worth the extra processing time
- For noisy/scanned/aged documents: Preprocessing likely provides more benefit (needs testing)
- The 10-20% improvement claim should be qualified or removed for clean typed documents

## Files

- Benchmark results: `benchmark_preprocessing_comparison.csv`
- Report: `benchmark_preprocessing_comparison.md`
- Ground truth: 5 pages (IMG_8478 through IMG_8482)
