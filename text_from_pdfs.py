#!/usr/bin/env python
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

import os

from openai import OpenAI
from dotenv import load_dotenv


MODEL = "gpt-4o"

SYSTEM_PROMPT = "You are a helpful assistant who is an expert on the English language, skilled in vocabulary, " \
                "pronunciation, and grammar. "
USER_PROMPT = "Correct any typos caused by bad OCR in this text, using common sense reasoning, responding only with " \
              "the corrected text: "


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


def process_image(path, output_dir):
    text = pytesseract.image_to_string(Image.open(path))
    return text


def ask_the_english_prof(client, user_query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    completion = client.chat.completions.create(model=MODEL, messages=messages)
    return completion.choices[0].message.content


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

    output_file_name = os.path.join(output_dir, "extracted.txt")
    write_pages(output_file_name, pages)

    corrected_file_name = os.path.join(output_dir, "corrected.txt")
    corrected_pages = []
    for page in pages:
        prompt = USER_PROMPT + page
        corrected = ask_the_english_prof(client, prompt)
        corrected_pages.append(corrected)

    write_pages(corrected_file_name, corrected_pages)

    df = pd.DataFrame({'image_path': files, 'extracted': pages, 'corrected': corrected_pages})
    return df


def write_pages(output_file_name, pages):
    print(f"Saving {len(pages)} pages of text to {output_file_name}")
    with open(output_file_name, "w") as f:
        f.write("\n".join(pages))


def main():
    load_dotenv()
    input_list_path = "input_file_list.txt"
    paths = open(input_list_path).readlines()
    paths = [p.strip() for p in paths]

    # TODO: testing
    paths = paths[:3]

    client = OpenAI()

    output_dir = os.path.join(os.getcwd(), "output")
    df = perform_ocr_on_images(paths, client, output_dir)
    csv_path = os.path.join(output_dir, "results.csv")
    print(f"Outputting results to {csv_path}")
    df.to_csv(csv_path, index=False)  
    


if __name__ == "__main__":
    main()
