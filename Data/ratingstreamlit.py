import streamlit as st
st.title('Oddy Test Coupon Rating')

# Drag and drop upload box
uploaded_files = st.file_uploader("Choose an image file", type=['png', 'jpg', 'tiff'], accept_multiple_files=True)

# Show uploaded images
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.image(uploaded_file)

if len(uploaded_files) > 0:
    st.button('Rate coupons')
    # Currently only downloads last image
    st.download_button("Download processed images", data=uploaded_file, file_name="Processed images.jpg")
