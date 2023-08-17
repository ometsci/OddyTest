import streamlit as st
import numpy as np
import tempfile

# Make a temporary directory
numpy_files = tempfile.mkdtemp()
result_directory = tempfile.mkdtemp()

st.title('Oddy Test Coupon Rating')

# Drag and drop upload box
uploaded_files = st.file_uploader("Choose an image file", type=['png', 'jpg', 'tiff'], accept_multiple_files=True)

# Show uploaded images
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.image(uploaded_file)

if len(uploaded_files) > 0:
    st.sidebar.button('Rate coupons')
    # Currently only downloads last image
    st.sidebar.download_button("Download processed images", data=uploaded_file, file_name="Processed images.jpg")
