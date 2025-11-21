# OCR Benchmark Report

Generated: 2025-11-20 19:53:51.695376

## Summary by Method

### pytesseract

- Images processed: 5
- Average processing time: 3.51s
- Average CER: 0.015
- Average WER: 0.056
- Best CER: 0.004
- Worst CER: 0.022

### pytesseract_no_preprocess

- Images processed: 5
- Average processing time: 3.15s
- Average CER: 0.016
- Average WER: 0.059
- Best CER: 0.003
- Worst CER: 0.023

## Detailed Results

| image_path          | method                    |   cer |   wer |   processing_time |
|:--------------------|:--------------------------|------:|------:|------------------:|
| images/IMG_8478.jpg | pytesseract               | 0.022 | 0.083 |              2.44 |
| images/IMG_8478.jpg | pytesseract_no_preprocess | 0.022 | 0.083 |              1.04 |
| images/IMG_8479.jpg | pytesseract               | 0.022 | 0.095 |              3.64 |
| images/IMG_8479.jpg | pytesseract_no_preprocess | 0.023 | 0.105 |              3.83 |
| images/IMG_8480.jpg | pytesseract               | 0.004 | 0.016 |              3.68 |
| images/IMG_8480.jpg | pytesseract_no_preprocess | 0.003 | 0.016 |              3.34 |
| images/IMG_8481.jpg | pytesseract               | 0.016 | 0.043 |              4.11 |
| images/IMG_8481.jpg | pytesseract_no_preprocess | 0.02  | 0.049 |              3.97 |
| images/IMG_8482.jpg | pytesseract               | 0.011 | 0.043 |              3.69 |
| images/IMG_8482.jpg | pytesseract_no_preprocess | 0.011 | 0.039 |              3.54 |

## Cost Estimation

GPT-5 pricing: $1.25 per 1M input tokens, $10.00 per 1M output tokens

- Total images: 5
- No cost data available (no LLM methods used)

