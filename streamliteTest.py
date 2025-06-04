from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

import streamlit as st
from PIL import Image

st.title("Text-to-Image Generator (Streamlit)")

prompt = st.text_input("Enter a prompt")
if st.button("Generate"):
    with st.spinner("Generating..."):
        import time
        start = time.time()
        image = pipe(prompt).images[0]
        print("Generation time:", time.time() - start)

        st.image(image, caption=prompt)
