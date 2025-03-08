import random
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from pipeline.mixture_tiling_sdxl import StableDiffusionXLTilingPipeline
from pipeline.util import MAX_SEED, create_hdr_effect, quantize_8bit, select_scheduler

device = "cuda"

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to(device)

model_id="stablediffusionapi/yamermix-v8-vae"
pipe = StableDiffusionXLTilingPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    vae=vae,
    use_safetensors=False, #for yammermix   
    #variant="fp16",
).to(device)

unet = UNet2DConditionModel.from_pretrained(model_id, 
                                            subfolder="unet", 
                                            use_safetensors=False
                                            #variant="fp16",
                                            )
quantize_8bit(unet)  # << Enable this if you have limited VRAM
pipe.unet = unet

pipe.enable_model_cpu_offload() #<< Enable this if you have limited VRAM
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

generation_seed = random.randint(0, MAX_SEED)
scheduler="UniPC" #<-- See **SAMPLERS** variable keys list on [util.py](pipeline/util.py#L24) file. 
pipe.scheduler = select_scheduler(pipe, scheduler)

# Set seed
generator = torch.Generator(device).manual_seed(generation_seed)

target_height = 1024
target_width = 3048
tile_height = 1024
tile_width = 1024

left_prompt = "a waterfall"
left_gs = 5.0 #-> cfg 

center_prompt = "a plain with cattle"
center_gs = 5.0 #-> cfg 

right_prompt = "a rocky mountain"
right_gs = 5.0 #-> cfg  

negative_prompt = "nsfw, lowres, bad anatomy, bad hands, duplicate, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry"
num_inference_steps = 30

tile_overlap_pixels = 256
hdr_effect = 0.0 # set this to enable it.

# Mixture of Diffusers generation
image = pipe(
    prompt=[
        [
            left_prompt,
            center_prompt,
            right_prompt,            
        ]
    ],
    negative_prompt=negative_prompt,
    tile_height=tile_height,
    tile_width=tile_width,
    tile_row_overlap=0,
    tile_col_overlap=tile_overlap_pixels,    
    guidance_scale_tiles=[[left_gs, center_gs, right_gs]],     
    height=target_height,
    width=target_width,               
    generator=generator,
    num_inference_steps=num_inference_steps
)["images"][0]

image = create_hdr_effect(image, hdr_effect)

image.save("result.png")