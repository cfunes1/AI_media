from flask import Flask, request, send_file
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

# Initialize model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "a scenic landscape")

    # Generate image
    import time
    start = time.time()
    image = pipe(prompt).images[0]
    print("Generation time:", time.time() - start)


    # Convert image to byte stream
    img_io = BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
