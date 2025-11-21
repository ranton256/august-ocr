# OCR Benchmark Report

Generated: 2025-11-20 15:57:36.815409

## Summary by Method

### pytesseract

- Images processed: 3
- Average processing time: 3.55s
- Average CER: 0.016
- Average WER: 0.065
- Best CER: 0.004
- Worst CER: 0.022

### trocr

- Images processed: 3
- Average processing time: 0.56s
- Average CER: 0.975
- Average WER: 1.000
- Best CER: 0.927
- Worst CER: 1.000

### pytesseract_gpt5

- Images processed: 3
- Average processing time: 48.79s
- Average CER: 2.805
- Average WER: 2.754
- Best CER: 0.280
- Worst CER: 7.701

### trocr_gpt5

- Images processed: 3
- Average processing time: 21.61s
- Average CER: 0.975
- Average WER: 0.999
- Best CER: 0.927
- Worst CER: 1.000

## Detailed Results

| image_path          | method           |   cer |   wer |   processing_time |
|:--------------------|:-----------------|------:|------:|------------------:|
| images/IMG_8478.jpg | pytesseract      | 0.022 | 0.083 |              2.84 |
| images/IMG_8478.jpg | trocr            | 0.927 | 1     |              0.71 |
| images/IMG_8478.jpg | pytesseract_gpt5 | 7.701 | 7.375 |             39.96 |
| images/IMG_8478.jpg | trocr_gpt5       | 0.927 | 1     |              9.36 |
| images/IMG_8479.jpg | pytesseract      | 0.022 | 0.095 |              3.84 |
| images/IMG_8479.jpg | trocr            | 1     | 1     |              0.5  |
| images/IMG_8479.jpg | pytesseract_gpt5 | 0.436 | 0.49  |             62.2  |
| images/IMG_8479.jpg | trocr_gpt5       | 0.999 | 0.998 |             25.55 |
| images/IMG_8480.jpg | pytesseract      | 0.004 | 0.016 |              3.98 |
| images/IMG_8480.jpg | trocr            | 1     | 1     |              0.49 |
| images/IMG_8480.jpg | pytesseract_gpt5 | 0.28  | 0.396 |             44.2  |
| images/IMG_8480.jpg | trocr_gpt5       | 1     | 1     |             29.91 |

## Cost Estimation

GPT-5 pricing: $1.25 per 1M input tokens, $10.00 per 1M output tokens

- Total images: 3
- Total LLM correction cost: $0.0228
- Average cost per page: $0.0038
- Total input tokens: 1,620.0
- Total output tokens: 2,077.0

