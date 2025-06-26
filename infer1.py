import random
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from pipeline.mixture_tiling_sdxl import StableDiffusionXLTilingPipeline
from pipeline.util import MAX_SEED, create_hdr_effect, quantize_8bit, select_scheduler

device = "cuda"

model_id="stabilityai/stable-diffusion-xl-base-1.0"
vae_subfolder = "vae"

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to(device)

# vae = AutoencoderKL.from_pretrained(
#     model_id,
#     subfolder=vae_subfolder,
#     # torch_dtype=torch.float16
# ).to(device)

pipe = StableDiffusionXLTilingPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    # vae=vae,
    # use_safetensors=False, #for yammermix   
    #variant="fp16",
).to(device)

unet = UNet2DConditionModel.from_pretrained(model_id, 
                                            subfolder="unet", 
                                            # use_safetensors=False
                                            #variant="fp16",
                                            ).to(device)
# quantize_8bit(unet)  # << Enable this if you have limited VRAM
pipe.unet = unet

# pipe.enable_model_cpu_offload() #<< Enable this if you have limited VRAM
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

generation_seed = 30
scheduler="UniPC" #<-- See **SAMPLERS** variable keys list on [util.py](pipeline/util.py#L24) file. 
# scheduler="DPM++ 2M Karras" #<-- See **SAMPLERS** variable keys list on [util.py](pipeline/util.py#L24) file. 
# scheduler="Euler Ancestral" #<-- See **SAMPLERS** variable keys list on [util.py](pipeline/util.py#L24) file. 

pipe.scheduler = select_scheduler(pipe, scheduler)

# Set seed
generator = torch.Generator(device).manual_seed(generation_seed)

target_height = 2048
target_width = 3048
tile_height = 1024
tile_width = 1024

left_prompt = "a satellite view of comic style stardew valley style city."
left_gs = 5.0 #-> cfg 

# center_prompt = "a satellite view of comic style stardew valley style city."
# center_prompt = "a satellite view of comic style style jungle."
# center_prompt = "Where's Waldo style art of a sprawling ancient Indian temple city. A river flows through the bustling cityscape with towering gopuram temples, intricate details."
# center_prompt = "Psychedelic: autumn forest landscape, psychedelic style, vibrant colors, swirling patterns, abstract forms, surreal, trippy, colorful"
# center_prompt = "abstract expressionist painting artstyle abstract expressionism jungle. energetic brushwork, bold colors, abstract forms, expressive, emotional."
# center_prompt = "cubist artwork artstyle cubist woman. geometric shapes, abstract, innovative, revolutionary"
# center_prompt = "graffiti style random text. street art, vibrant, urban, detailed, tag, mural"
# center_prompt = "grunge style grid style roses. textured, distressed, vintage, edgy, punk rock vibe, dirty, noisy"
center_prompt = "collage style random. mixed media, layered, textural, detailed, artistic"


center_gs = 5.0 #-> cfg 

right_prompt = "a satellite view of comic style stardew valley style city."
right_gs = 5.0 #-> cfg  

# negative_prompt = "nsfw, lowres, bad anatomy, bad hands, duplicate, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry"
# negative_prompt = "realistic, photorealistic, low contrast, plain, simple, monochrome"
# negative_prompt = "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
# negative_prompt = "walls, bricks, floor, ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
# negative_prompt = "smooth, clean, minimalist, sleek, modern, photorealistic"
negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"


num_inference_steps = 30

tile_overlap_pixels = 896
hdr_effect = 0.0 # set this to enable it.

g_scale = 10
# num_h_tiles = 255
# num_v_tiles = 55

num_h_tiles = 12
num_v_tiles = 2

all_h_prompts = [center_prompt for i in range(num_h_tiles)]
g_h_sclaes = [g_scale for i in range(num_h_tiles)]

all_prompts = [all_h_prompts for i in range(num_v_tiles)]
all_g_sclaes = [g_h_sclaes for i in range(num_v_tiles)]


# Mixture of Diffusers generation
image = pipe(
    prompt=all_prompts,
    negative_prompt=negative_prompt,
    tile_height=tile_height,
    tile_width=tile_width,
    tile_row_overlap=tile_overlap_pixels,
    tile_col_overlap=tile_overlap_pixels,    
    guidance_scale_tiles=all_g_sclaes,     
    height=target_height,
    width=target_width,               
    generator=generator,
    num_inference_steps=num_inference_steps
)["images"][0]

image = create_hdr_effect(image, hdr_effect)

image.save("result.png")
print(image.size)