import streamlit as st
import pandas as pd
from PIL import Image
import difflib
from pathlib import Path
from common import get_preproc_path

st.set_page_config(
    page_title="August OCR",
    page_icon="📖",
    layout="wide",
    #initial_sidebar_state="expanded",
)

def main():
    df = pd.read_csv("output/results.csv")
    st.title("OCR Comparison App")

    if len(df) > 1:
        page = st.slider('Select Page', 1, len(df), 1)
    else:
        page = 1

    st.write(f"Showing page {page}")

    image_path = df.loc[page - 1, 'image_path']
    extracted_text = df.loc[page - 1, 'extracted']
    corrected_text = df.loc[page - 1, 'corrected']

    output_dir = "output"

    image = Image.open(image_path)
    pre_path = get_preproc_path(image_path, output_dir)
    pre_image = Image.open(pre_path)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(image, caption=f'Page {page}', use_column_width=True)

    with col2:
        st.image(pre_image, caption=f'Preprocessed Page {page}', use_column_width=True)

    with col3:
        st.subheader("Extracted Text")
        st.write(extracted_text)

    with col4:
        st.subheader("Corrected Text")
        st.write(corrected_text)

if __name__ == "__main__":
    main()

