import streamlit as st
import os
from detector import detect_enemies

st.set_page_config(page_title="Enemy Detection App", layout="centered")

st.title("🎮 Enemy / Bot Detection")

st.write(
    "Detection rules (as per client):\n"
    "- Blue horizontal bar above head: Teammate (ignored, no box).\n"
    "- FF00FF pink horizontal bar above head: Enemy (red box).\n"
    "- No bar: Bot or mannequin. Bots are treated as enemies (red box); mannequins are filtered using a pretrained image classifier.\n"
    "- Teammates and mannequins must not be detected."
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = os.path.join("uploads", uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(input_path, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Enemies / Bots"):
        with st.spinner("Detecting enemies and bots (teammates and mannequins ignored)..."):
            output_img, output_path = detect_enemies(input_path)

        st.success("Detection Complete!")
        st.image(output_path, caption="Detection Output", use_container_width=True)
        st.write(f"Saved output: {output_path}")
