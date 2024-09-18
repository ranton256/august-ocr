import time

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
    df = pd.read_csv("output/results.csv")
    st.title("OCR Comparison App")

    st.write(
        """This project shows how to extract text from images or PDFs using PyTesseract, Pillow, and opencv-python.
It performs a number of preprocessing steps to improve the results, and then uses OpenAI's GPT-4o to correct the OCR output.
""")

    st.write("For more information see the the [README](https://github.com/ranton256/august-ocr/blob/main/README.md).")

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
        if st.button("First Page"):
            if page > 1:
                page = 1
                st.query_params['page'] = page
    with col2:
        if st.button("Previous Page"):
            if page > 1:
                page -= 1
                st.query_params['page'] = page
    with col3:
        st.write(f"Showing page {page} of {n_pages}")

    with col4:
        if st.button("Next Page"):
            if page < n_pages:
                page += 1
                st.query_params['page'] = page
    with col5:
        if st.button("Last Page"):
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

    image = Image.open(image_path)
    pre_path = get_preproc_path(image_path, output_dir)
    pre_image = Image.open(pre_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption=f'Page {page}', use_column_width=True)

    with col2:
        st.image(pre_image, caption=f'Preprocessed Page {page}', use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extracted Text")
        st.write(extracted_text)

    with col2:
        st.subheader("Corrected Text")
        st.write(corrected_text)


if __name__ == "__main__":
    main()
