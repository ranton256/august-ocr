#!/usr/bin/env python
import argparse
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

import os
import traceback

from openai import OpenAI
from dotenv import load_dotenv


MODEL = "gpt-4o"

SYSTEM_PROMPT = "You are a helpful assistant who is an expert on the English language, skilled in vocabulary, " \
                "pronunciation, and grammar. "
USER_PROMPT = "Correct any typos caused by bad OCR in this text, using common sense reasoning, responding only with " \
              "the corrected text: "
RECT_SIZE = (50, 40)


def process_pdf(path, output_dir):
    pages = []
    images = convert_from_path(path)
    for i, image in enumerate(images):
        image_file_name = "page_" + str(i) + "_" + os.path.basename(path) + ".jpg"
        image_path = os.path.join(output_dir, image_file_name)
        print(f"Saving page {i} of {path} to {image_path}")
        image.save(image_file_name, "JPEG")
        text = process_image(image_file_name, output_dir)
        pages.append(text)
    return pages



def get_preproc_path(path, output_dir):
    path_base, path_ext = os.path.splitext(os.path.basename(path))
    preproc_file_name = path_base + "_proc.jpg"
    return os.path.join(output_dir, preproc_file_name)


def process_image(path, output_dir):
    #image = Image.open(path)
    #image = cv2.read(path)
    img = cv2.imread(
      filename=path,
      flags=cv2.IMREAD_COLOR,
    )

    h, w, c = img.shape
    print(f'Image shape: {h}H x {w}W x {c}C')
        
    img = cv2.cvtColor(
      src=img,    
      code=cv2.COLOR_BGR2RGB,
    )
    img = preprocess_image(img)
    # save the preprocessed image
    preproc_path = get_preproc_path(path, output_dir)
    pil_image = Image.fromarray(img) 
    pil_image.save(preproc_path, "JPEG")

    text = extract_text(img)

    return text

def extract_text(img):
    # Find bounding boxes of interest
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RECT_SIZE)
    dilation = cv2.dilate(img, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()

    # For each box, extract subimage with crop and extract text.
    cnt_list=[]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"Extracting text from bbox at {x}, {y} size {w} by {h}", end="")
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(im2,(x,y),8,(255,255,0),8)
        cropped = im2[y:y + h, x:x + w]

        text = pytesseract.image_to_string(cropped)
        text = text.strip()
        if text:
            print(f" --> {text}")
            cnt_list.append((x,y,text))
        else:
            print(" --> (nil)")
    # Sort by y position to keep text in correct order.
    sorted_list = sorted(cnt_list, key=lambda c: c[1])
    # Return as text paragraphs
    all_text = []
    last_y = 0
    for x, y, txt in sorted_list:
        div = "\n"
        if y - last_y > 1:
            # assume this is a new paragraph is y is different.
            div = "\n\n"
        all_text.append(div)
        all_text.append(txt)

    result = "".join(all_text)
    print(f"Result is length: {len(result)}")
    return result


def ask_the_english_prof(client, user_query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    completion = client.chat.completions.create(model=MODEL, messages=messages)
    return completion.choices[0].message.content


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image,5)


def thresholding(image):
    # create a binary (two-level) image from grayscale input.
    cutoff, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    print(f"thresholding at {cutoff}")
    return image


def preprocess_image(image):
    image = get_grayscale(image)
    image = remove_noise(image)
    image = thresholding(image)
    
    return image


def perform_ocr_on_images(files, client, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pages = []
    n_files = len(files)
    for pi, path in enumerate(files):
        print(f"Processing {path}, {pi+1}/{n_files}")
        try:
            path_base, path_ext = os.path.splitext(path)
            if path_ext.lower() == ".pdf":
                pages.extend(process_pdf(path, output_dir))
            else:
                pages.append(process_image(path, output_dir))
        except BaseException as ex:
            print(f"Error extracting text from {path}", ex)
            traceback.print_exc()
            raise ex

    print(f"Extracted {len(pages)} pages")
    output_file_name = os.path.join(output_dir, "extracted.txt")
    write_pages(output_file_name, pages)

    print("Generating corrections")
    corrected_file_name = os.path.join(output_dir, "corrected.txt")
    corrected_pages = []
    for pi, page in enumerate(pages):
        print(f"Correcting {pi+1}/{len(pages)}")
        prompt = USER_PROMPT + page
        corrected = ask_the_english_prof(client, prompt)
        corrected_pages.append(corrected)

    print(f"counts: files: {len(files)}, pages:{len(pages)}, corrected:{len(corrected_pages)}")
    write_pages(corrected_file_name, corrected_pages)

    preproc_paths = [get_preproc_path(p, output_dir) for p in files]

    df = pd.DataFrame({'image_path': files, 'preprocessed': preproc_paths, 'extracted': pages, 'corrected': corrected_pages})
    return df


def write_pages(output_file_name, pages):
    print(f"Saving {len(pages)} pages of text to {output_file_name}")
    with open(output_file_name, "w") as f:
        f.write("\n".join(pages))


def main():
    load_dotenv()
    # parser.add_argument("-l", "--long", action="store_true")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max", help="Maximum number of documents to process", type=int, default=0)

    args = parser.parse_args()

    input_list_path = "input_file_list.txt"
    paths = open(input_list_path).readlines()
    paths = [p.strip() for p in paths]

    
    if args.max > 0:
        paths = paths[:args.max]

    client = OpenAI()

    output_dir = os.path.join(os.getcwd(), "output")
    df = perform_ocr_on_images(paths, client, output_dir)
    csv_path = os.path.join(output_dir, "results.csv")
    print(f"Outputting results to {csv_path}")
    df.to_csv(csv_path, index=False)  
    


if __name__ == "__main__":
    main()
