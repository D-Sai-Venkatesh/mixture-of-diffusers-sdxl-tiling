<h1 align="center">Mixture-of-Diffusers for SDXL Tiling Pipeline🤗</h1>           
<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; overflow:hidden;">
    <span>This project implements the <a href="https://arxiv.org/pdf/2302.02412">📜 MoD (Mixture-of-Diffusers)</a> tiled diffusion technique for SDXL pipeline and is based on the original project at <a href='https://github.com/albarji/mixture-of-diffusers'>Mixture-of-Diffusers</a>.    
</div>

If you like the project, please give me a star! ⭐

[![GitHub](https://img.shields.io/github/stars/DEVAIEXP/mixture-of-diffusers-sdxl-tiling?style=socia)](https://github.com/DEVAIEXP/mixture-of-diffusers-sdxl-tiling)
<a href='https://huggingface.co/spaces/elismasilva/mixture-of-diffusers-sdxl-tiling'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a><br>
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6S71ACXMR)

<div style="text-align: center;">
  <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/mixture_tiling_app.PNG" width="1536">
</div>

## About
The **Mixture-of-Diffusers for SDXL Tiling Pipeline** strives to provide a better tool for image composition by using several diffusion processes in parallel, each configured with a specific prompt and settings, and focused on a particular region of the image. The mixture of diffusion processes is done in a way that harmonizes the generation process, preventing "seam" effects in the generated image. Using several diffusion processes in parallel has also practical advantages when generating very large images, as the GPU memory requirements are similar to that of generating an image of the size of a single tile. For more information see original project here: <a href='https://github.com/albarji/mixture-of-diffusers'>Mixture-of-Diffusers</a>


## Key Features
**Multi-Area Prompt Support**
Easily input distinct descriptions for the left, center, and right regions (e.g., "a waterfall" for the left, "a plain with cattle" for the center, and "a rocky mountain" for the right). This innovative feature allows the system to seamlessly blend multiple scenes into one breathtaking panoramic image.

**Advanced Tiling Technology**
The project leverages a sophisticated tiling approach that intelligently manages overlapping regions, ensuring natural transitions and high-resolution panoramic outputs. This isn’t just a simple image merge—it’s a refined process designed to deliver exceptional quality and intricate detail.

## Installation

Use Python version 3.10.* and have the Python virtual environment installed.

Then run the following commands in the terminal:

**Clone repository:**
```bash
git clone https://github.com/DEVAIEXP/mixture-of-diffusers-sdxl-tiling.git
cd mixture-of-diffusers-sdxl-tiling
```

**Prepare environment:**
```bash
python -m venv venv
(for windows) .\venv\Scripts\activate
(for linux) source /venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
pip install -r requirements.txt
```

## How to Run
**Gradio app:**
To launch the Gradio app on your local machine, execute the following command in your terminal:
```bash
python app.py
```

The following code👇 comes from [infer.py](infer.py). If you want to do quickly inference, please refer to the code in [infer.py](infer.py). 

````python
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
````

To save VRAM, you can enable FP8 Quantization on UNET:
````python
unet = UNet2DConditionModel.from_pretrained(model_id, 
                                            subfolder="unet", 
                                            use_safetensors=False
                                            #variant="fp16",
                                            )
quantize_8bit(unet)
pipe.unet = unet
````

To save VRAM, you can enable CPU offloading, vae tiling and vae slicing
````python
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()
````

Set the scheduler. See **SAMPLERS** variable keys list on [util.py](pipeline/util.py#L24) file. 
````python
# Set selected scheduler
scheduler="UniPC" #<--set the key name here
pipe.scheduler = select_scheduler(pipe, scheduler)
````
....
````python
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
````

## Acknowledgements
- Our work is highly inspired by [Mixture-of-Diffusers](https://github.com/albarji/mixture-of-diffusers). Thanks for their great works!
- We borrowed some ideias like **hdr effect** from [TileUpscalerV2](https://huggingface.co/spaces/gokaygokay/TileUpscalerV2). Thanks for your work!

## Other DEVAIXP works
* [MoD ControlNet Tile Upscaler for SDXL](https://github.com/DEVAIEXP/mod-control-tile-upscaler-sdxl) - SDXL Image-to-Image pipeline that leverages **ControlNet Tile** and **Mixture-of-Diffusers techniques**, integrating tile diffusion directly into the latent space denoising process. Designed to overcome the limitations of conventional pixel-space tile processing, this pipeline delivers **Super Resolution (SR)** upscaling for **higher-quality images, reduced processing time**, and **greater adaptability**.and settings, and focused on a particular region of the image.
* [Image Interrogator](https://github.com/DEVAIEXP/image-interrogator) - Tool for image captioning with support for large models like LLaVa, CogVml and others.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEVAIEXP/mixture-of-diffusers-sdxl-tiling&type=Date)](https://star-history.com/#DEVAIEXP/mixture-of-diffusers-sdxl-tiling&Date)

## License
This project is released under the [MIT](LICENSE).

## Contact
If you have any questions, please contact: contact@devaiexp.com