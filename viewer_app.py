import time
import os

import streamlit as st
import pandas as pd
from PIL import Image
from common import get_preproc_path

st.set_page_config(
    page_title="August OCR",
    page_icon="📖",
    layout="wide",
)


def main():
    st.title("OCR Comparison App")

    # Mode selection
    mode = st.radio(
        "Select OCR Mode:",
        ["Document OCR (Pytesseract)", "Handwriting OCR (TrOCR)"],
        horizontal=True,
        help="Choose between traditional document OCR or handwriting recognition"
    )

    # Load appropriate results file
    if mode == "Document OCR (Pytesseract)":
        results_file = "output/results.csv"
        description = """This shows traditional OCR using PyTesseract, Pillow, and opencv-python.
It performs preprocessing steps to improve results, then uses OpenAI's GPT-4o to correct the OCR output.
This works best for typed or printed documents."""
    else:
        results_file = "output/handwriting_results.csv"
        description = """This shows handwriting recognition using Microsoft's TrOCR, a transformer-based OCR model.
It's specifically trained on handwritten text and works well on both cursive and printed handwriting.
This works best for handwritten notes captured with a phone camera."""

    st.write(description)
    st.write("For more information see the [README](https://github.com/ranton256/august-ocr/blob/main/README.md).")

    # Check if results file exists
    if not os.path.exists(results_file):
        st.warning(f"Results file not found: {results_file}")
        if mode == "Handwriting OCR (TrOCR)":
            st.info("Run `python handwriting_ocr.py --input handwriting_images/` to generate handwriting results.")
        else:
            st.info("Run `python text_from_pdfs.py` to generate document OCR results.")
        return

    df = pd.read_csv(results_file)

    n_pages = len(df)
    if n_pages == 0:
        st.write("No pages to show")
        return

    if 'page' not in st.query_params:
        page = 1
        st.query_params['page'] = page
    else:

        try:
            page = int(st.query_params['page'])
        except ValueError:
            st.write("Invalid page number")
            page = 1

    old_page = page

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("First Page", disabled=page == 1):
            if page > 1:
                page = 1
                st.query_params['page'] = page
    with col2:
        if st.button("Previous Page", disabled=page == 1):
            if page > 1:
                page -= 1
                st.query_params['page'] = page
    with col3:
        st.write(f"Showing page {page} of {n_pages}")

    with col4:
        if st.button("Next Page", disabled=page == n_pages):
            if page < n_pages:
                page += 1
                st.query_params['page'] = page
    with col5:
        if st.button("Last Page", disabled=page == n_pages):
            if page < n_pages:
                page = n_pages
                st.query_params['page'] = page

    if n_pages > 1:
        page = st.slider('Select Page', 1, n_pages, page)
        if old_page != page:
            st.query_params['page'] = page

    if old_page != page:
        # NOTE: This is a terrible hack due to a sync. bug between st.rerun() and st.query_params
        time.sleep(0.1)
        st.rerun()

    image_path = df.loc[page - 1, 'image_path']
    extracted_text = df.loc[page - 1, 'extracted']
    corrected_text = df.loc[page - 1, 'corrected']

    output_dir = "output"

    # Load images
    image = Image.open(image_path)

    # For document OCR, use the standard preprocessing path
    # For handwriting OCR, look for preprocessed_path column
    if mode == "Document OCR (Pytesseract)":
        pre_path = get_preproc_path(image_path, output_dir)
        pre_caption = f'Preprocessed Page {page}'
    else:
        # Handwriting results have preprocessed_path column
        if 'preprocessed_path' in df.columns:
            pre_path = df.loc[page - 1, 'preprocessed_path']
            pre_caption = f'Preprocessed Photo {page}'
        else:
            pre_path = image_path  # Fallback to original
            pre_caption = f'Original Photo {page}'

    # Check if preprocessed image exists
    if os.path.exists(pre_path):
        pre_image = Image.open(pre_path)
    else:
        pre_image = image  # Fallback to original image

    # Display images
    col1, col2 = st.columns(2)

    with col1:
        caption = f'Page {page}' if mode == "Document OCR (Pytesseract)" else f'Photo {page}'
        st.image(image, caption=caption, use_column_width=True)

    with col2:
        st.image(pre_image, caption=pre_caption, use_column_width=True)

    # Display text results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extracted Text")
        st.write(extracted_text)

        # Show model info for handwriting mode
        if mode == "Handwriting OCR (TrOCR)" and 'model' in df.columns:
            st.caption(f"Model: {df.loc[page - 1, 'model']}")

    with col2:
        st.subheader("Corrected Text")
        st.write(corrected_text)

        # Show character/word count
        if corrected_text and isinstance(corrected_text, str):
            char_count = len(corrected_text)
            word_count = len(corrected_text.split())
            st.caption(f"{word_count} words, {char_count} characters")


if __name__ == "__main__":
    main()
