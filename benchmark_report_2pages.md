# OCR Benchmark Report

Generated: 2025-11-20 15:49:55.938715

## Summary by Method

### pytesseract

- Images processed: 2
- Average processing time: 3.34s
- Average CER: 0.022
- Average WER: 0.089
- Best CER: 0.022
- Worst CER: 0.022

### trocr

- Images processed: 2
- Average processing time: 0.63s
- Average CER: 0.963
- Average WER: 1.000
- Best CER: 0.927
- Worst CER: 1.000

### pytesseract_gpt5

- Images processed: 2
- Average processing time: 47.70s
- Average CER: 3.560
- Average WER: 3.419
- Best CER: 0.389
- Worst CER: 6.730

### trocr_gpt5

- Images processed: 2
- Average processing time: 19.06s
- Average CER: 0.963
- Average WER: 1.000
- Best CER: 0.927
- Worst CER: 0.999

## Detailed Results

| image_path          | method           |   cer |   wer |   processing_time |
|:--------------------|:-----------------|------:|------:|------------------:|
| images/IMG_8478.jpg | pytesseract      | 0.022 | 0.083 |              2.84 |
| images/IMG_8478.jpg | trocr            | 0.927 | 1     |              0.74 |
| images/IMG_8478.jpg | pytesseract_gpt5 | 6.73  | 6.375 |             25.49 |
| images/IMG_8478.jpg | trocr_gpt5       | 0.927 | 1     |              8.57 |
| images/IMG_8479.jpg | pytesseract      | 0.022 | 0.095 |              3.83 |
| images/IMG_8479.jpg | trocr            | 1     | 1     |              0.53 |
| images/IMG_8479.jpg | pytesseract_gpt5 | 0.389 | 0.464 |             69.9  |
| images/IMG_8479.jpg | trocr_gpt5       | 0.999 | 1     |             29.54 |

## Cost Estimation

GPT-5 pricing: $1.25 per 1M input tokens, $10.00 per 1M output tokens

- Total images: 2
- Total LLM correction cost: $0.0132
- Average cost per page: $0.0033
- Total input tokens: 852.0
- Total output tokens: 1,211.0

