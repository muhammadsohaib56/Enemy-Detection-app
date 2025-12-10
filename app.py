import streamlit as st
import os
from detector import detect_enemies

st.set_page_config(page_title="Enemy Detection App", layout="centered")

st.title("🎮 Enemy Detection")
st.write(
    "Rules:\n"
    "- FF00FF pink horizontal bar above head: Enemy (red box)\n"
    "- Blue horizontal bar above head: Teammate (black box)\n"
    "- No bar: Bot (treated as enemy, red box)\n"
    "Shadows are ignored."
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = os.path.join("uploads", uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(input_path, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Enemies"):
        # Show spinner while processing
        with st.spinner("Detecting enemies, teammates, and bots..."):
            output_img, output_path = detect_enemies(input_path)

        st.success("Detection Complete!")
        st.image(output_path, caption="Enemy Detection Output", use_container_width=True)
        st.write(f"Saved output: {output_path}")
