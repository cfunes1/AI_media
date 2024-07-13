from carlos_tools_image import generate_DALLE3_image, save_image_from_b64data, generate_SD3_image, generate_SDXLt_image

"""Generate an image from text using choose model."""
prompt = input("Enter the text prompt for the image: ")
print("Models available: \n1) DALL-E 3, \n2) Stable Diffusion 3, \n3) Stable Difussion XL turbo, \n4) All")
i: str = input("Choose a model: ")
if i not in {"1", "2", "3", "4"}:
    print("Invalid model choice.")
    exit()
if i == "1" or i == "4":
    image_data = generate_DALLE3_image(prompt, "b64_json")
    save_image_from_b64data(image_data, "media", "DALLE3.png")
if i == "2" or i == "4":
    generate_SD3_image(prompt, "media", "SD3.png")
if i == "3" or i == "4":
    generate_SDXLt_image(prompt, "media", "SDXLt.png")


