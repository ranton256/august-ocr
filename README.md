# Python OCR using Pytesseract, Pillow, and pdf2image

## Setup

brew install tesseract
brew install poppler


conda env create -p ./env  -f environment.yml

pip install -r requirements.txt

## Credits

## Credits

I took inspiration and some code snippets from various articles online:

This article has a good overview of image preprocessing methods for OCR.

- <https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7>

The article provided a great example of identifying rectangles in the image to process separately.

- <https://medium.com/@siromermer/extracting-text-from-images-ocr-using-opencv-pytesseract-aa5e2f7ad513>

## Future Improvements

- Use the dewarping algorithm discussed in <https://mzucker.github.io/2016/08/15/page-dewarping.html> and implemented in <https://github.com/tachylatus/page_dewarp>.

