# Step 1: Import required libraries
from diffusers import StableDiffusionPipeline
import torch

# Step 2: Load pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Step 3: Text prompt (you can change this)
prompt = "A smart city with AI robots and green energy"

# Step 4: Generate image
image = pipe(prompt).images[0]

# Step 5: Save generated image
image.save("generated_image.png")

print("Image generated and saved as generated_image.png")
