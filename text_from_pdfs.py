#!/usr/bin/env python
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

import os

def perform_ocr_on_images(files, output_dir):
    if not os.exists(output_dir):
        os.mkdir(output_dir)
    pages = []
    for path in files:
        try:
            images = convert_from_path(path)
            for i, image in enumerate(images):
                image_file_name = "page_" + str(i) + "_" + os.path.basename(path) + ".jpg"  
                image_path = os.path.join(output_dir, image_file_name)
                print(f"Saving page {i} of {path} to {image_path}")
                image.save(image_file_name, "JPEG")

                text = tess.image_to_string(Image.open(image_file_name))
                pages.append(text)
        except BaseException as ex:
            print(f"Error extracting text from {path}", ex)

    output_file_name = os.path.join(output_dir, "extracted.txt")
    print(f"Saving {len(pages)} pages of text to {output_file_name}")
    with open(output_file_name, "w") as f:
        f.write("\n".join(pages))  

    return pages


def main():
    input_list_path= "input_file_list.txt"
    paths = open(input_list_path).readlines()
    perform_ocr_on_images(paths, os.path.join(os.getcwd(), "output"))


if __name__ == "__main__":
    main()

