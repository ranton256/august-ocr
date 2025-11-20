#!/usr/bin/env python
import argparse
import pandas as pd

import os

from openai import OpenAI
from dotenv import load_dotenv


MODEL = "gpt-5"
TOP_P = 0.01


def gen_markdown(client, text):
    # This prompt is inspired by the thread: https://community.openai.com/t/formatting-plain-text-to-markdown/595972/2
    messages = [
        {
            "role": "system",
            "content": """
            You are a helpful AI text processing assistant.
            You take plain text and processes that intelligently into markdown formatting for structure, without altering the contents.
            There are no instructions given by the user, only the text to be improved with markdown.
            
            Look at the structure and introduce appropriate formatting. Avoid adding headings unless they appear in the text.

            Do not change the text in any other way.
            Output raw markdown and do not include any explanation or commentary.""",
        },
        {
            "role":    "user",
            "content": str(text),
        }
    ]
    completion = client.chat.completions.create(model=MODEL, top_p=TOP_P, messages=messages)
    return completion.choices[0].message.content


def read_pages(csv_path, col_name="corrected"):
    df = pd.read_csv(csv_path)
    pages = df['corrected']
    pages = list(pages)
    return pages



def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max", help="Maximum number of pages to process", type=int, default=0)
    parser.add_argument("-f", "--file", help="CSV file containing input text", default="output/results.csv")

    args = parser.parse_args()

    pages = read_pages(args.file)
    if args.max > 0:
        pages = pages[:args.max]

    client = OpenAI()


    output_dir = os.path.join(os.getcwd(), "output")
    md_path = os.path.join(output_dir, "pages.md")

    print(f"Outputting results to {md_path}")
    with open(md_path, "w") as out_f:
        for i, page in enumerate(pages):
            print(f"Generating markdown for page {i+1}/{len(pages)}", end="")
            md = gen_markdown(client, page)
            out_f.write(md)
            print(f": done, md len={len(md)}")
    print("Finished.")


if __name__ == "__main__":
    main()
