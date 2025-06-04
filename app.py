import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Image Augmentation Tool", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📸 Image Augmentation Tool</h1>",
    unsafe_allow_html=True
)

st.markdown("Enhance your images with transformations like rotation, flip, grayscale, crop, and more!")

# Sidebar
with st.sidebar:
    st.header("🛠️ Transformation Options")
    gray = st.checkbox("🖤 Convert to Grayscale")
    rotate = st.slider("🔄 Rotate Image", -180, 180, 0)
    flip_h = st.checkbox("↔️ Flip Horizontally")
    flip_v = st.checkbox("↕️ Flip Vertically")
    shear_x = st.slider("📐 Shear X", -0.5, 0.5, 0.0, step=0.01)
    translate_x = st.slider("➡️ Translate X", -100, 100, 0)
    translate_y = st.slider("⬇️ Translate Y", -100, 100, 0)
    crop = st.checkbox("✂️ Crop Center (50%)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        orig = image.copy()

        # Transformations
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if rotate != 0:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), rotate, 1)
            image = cv2.warpAffine(image, M, (w, h))

        if flip_h:
            image = cv2.flip(image, 1)
        if flip_v:
            image = cv2.flip(image, 0)

        if shear_x != 0.0:
            h, w = image.shape[:2]
            M = np.float32([[1, shear_x, 0], [0, 1, 0]])
            image = cv2.warpAffine(image, M, (w, h))

        if translate_x != 0 or translate_y != 0:
            M = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        if crop:
            h, w = image.shape[:2]
            x1, y1 = w // 4, h // 4
            x2, y2 = x1 + w // 2, y1 + h // 2
            image = image[y1:y2, x1:x2]

        # Display in columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Original Image")
            st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("🎨 Transformed Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(result)
        buf = io.BytesIO()
        im_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("📥 Download Transformed Image", byte_im, file_name="transformed.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please upload an image to begin.")
