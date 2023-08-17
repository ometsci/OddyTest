import streamlit as st
import numpy as np
st.title('Oddy Test Coupon Rating')

# Drag and drop upload box
uploaded_files = st.file_uploader("Choose an image file", type=['png', 'jpg', 'tiff'], accept_multiple_files=True)

# Show uploaded images
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.image(uploaded_file)

if len(uploaded_files) > 0:
    # Folder for numpy files
    !rm -rf numpy_files
    !mkdir numpy_files
    # Make directory for outputted images
    !rm -rf result_directory
    !mkdir result_directory
    st.sidebar.button('Rate coupons')
    # Currently only downloads last image
    st.sidebar.download_button("Download processed images", data=uploaded_file, file_name="Processed images.jpg")
