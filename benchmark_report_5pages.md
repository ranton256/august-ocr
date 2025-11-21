# OCR Benchmark Report

Generated: 2025-11-20 19:50:12.252785

## Summary by Method

### pytesseract

- Images processed: 5
- Average processing time: 3.79s
- Average CER: 0.015
- Average WER: 0.056
- Best CER: 0.004
- Worst CER: 0.022

### trocr

- Images processed: 5
- Average processing time: 0.65s
- Average CER: 0.983
- Average WER: 0.999
- Best CER: 0.927
- Worst CER: 1.000

### pytesseract_gpt5

- Images processed: 5
- Average processing time: 75.91s
- Average CER: 1.698
- Average WER: 1.722
- Best CER: 0.276
- Worst CER: 7.029

### trocr_gpt5

- Images processed: 5
- Average processing time: 13.99s
- Average CER: 0.983
- Average WER: 0.998
- Best CER: 0.927
- Worst CER: 1.000

## Detailed Results

| image_path          | method           |   cer |   wer |   processing_time |
|:--------------------|:-----------------|------:|------:|------------------:|
| images/IMG_8478.jpg | pytesseract      | 0.022 | 0.083 |              2.86 |
| images/IMG_8478.jpg | trocr            | 0.927 | 1     |              0.76 |
| images/IMG_8478.jpg | pytesseract_gpt5 | 7.029 | 6.833 |             29.44 |
| images/IMG_8478.jpg | trocr_gpt5       | 0.927 | 1     |              5.21 |
| images/IMG_8479.jpg | pytesseract      | 0.022 | 0.095 |              3.81 |
| images/IMG_8479.jpg | trocr            | 1     | 1     |              0.51 |
| images/IMG_8479.jpg | pytesseract_gpt5 | 0.455 | 0.513 |             80.15 |
| images/IMG_8479.jpg | trocr_gpt5       | 0.999 | 1     |             31.69 |
| images/IMG_8480.jpg | pytesseract      | 0.004 | 0.016 |              3.94 |
| images/IMG_8480.jpg | trocr            | 1     | 1     |              0.52 |
| images/IMG_8480.jpg | pytesseract_gpt5 | 0.451 | 0.506 |             95.93 |
| images/IMG_8480.jpg | trocr_gpt5       | 0.999 | 0.998 |             12.54 |
| images/IMG_8481.jpg | pytesseract      | 0.016 | 0.043 |              4.38 |
| images/IMG_8481.jpg | trocr            | 0.989 | 0.993 |              0.94 |
| images/IMG_8481.jpg | pytesseract_gpt5 | 0.278 | 0.355 |             77.56 |
| images/IMG_8481.jpg | trocr_gpt5       | 0.989 | 0.993 |              6.34 |
| images/IMG_8482.jpg | pytesseract      | 0.011 | 0.043 |              3.97 |
| images/IMG_8482.jpg | trocr            | 1     | 1     |              0.52 |
| images/IMG_8482.jpg | pytesseract_gpt5 | 0.276 | 0.402 |             96.45 |
| images/IMG_8482.jpg | trocr_gpt5       | 1     | 1     |             14.18 |

## Cost Estimation

GPT-5 pricing: $1.25 per 1M input tokens, $10.00 per 1M output tokens

- Total images: 5
- Total LLM correction cost: $0.0448
- Average cost per page: $0.0045
- Total input tokens: 3,320.0
- Total output tokens: 4,066.0

