from diffusers import StableDiffusionInpaintPipeline, DiffusionPipeline, AutoencoderKL, AutoPipelineForInpainting
import torch
from PIL import Image

# This script is for doing inpainting on the sd/sdxl inpainting model with lora weights

pipe = AutoPipelineForInpainting.from_pretrained("./models/sdxl-inpainting-1.0", 
                                                torch_dtype=torch.float16, use_safetensors=True)

pipe.load_lora_weights("./lora-weights/ms-sofa-sdxl-from-inpainting", weight_name="pytorch_lora_weights.safetensors", adapter_name="ms")
pipe.to("cuda")

# For inpainting
img_url = "./images/home1.jpg"
mask_url = "./images/home1-cft-mask.png"

init_image = Image.open(img_url).resize((1024, 1024)).convert("RGB")
mask_image = Image.open(mask_url).resize((1024, 1024)).convert("RGB")

prompt = ["a photo of a mct table in a living room"] * 4
# generator = torch.Generator(device="cuda").manual_seed(9)

# Inpainting
images = pipe(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=8.0, num_inference_steps=30, strength=0.99).images

# Save the inpainted image
images[0].save("./results/home-sdxl-lora-1.png")
images[1].save("./results/home-sdxl-lora-2.png")
images[2].save("./results/home-sdxl-lora-3.png")
images[3].save("./results/home-sdxl-lora-4.png")
