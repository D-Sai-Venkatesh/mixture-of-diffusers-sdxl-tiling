import random
import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import AutoencoderKL
from mixture_tiling_sdxl import StableDiffusionXLTilingPipeline

MAX_SEED = np.iinfo(np.int32).max
SCHEDULERS = [
                "LMSDiscreteScheduler",
                "DEISMultistepScheduler",
                "HeunDiscreteScheduler",
                "EulerAncestralDiscreteScheduler",
                "EulerDiscreteScheduler",
                "DPMSolverMultistepScheduler",
                "DPMSolverMultistepScheduler-Karras",
                "DPMSolverMultistepScheduler-Karras-SDE",
                "UniPCMultistepScheduler"
]

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")

model_id="stablediffusionapi/yamermix-v8-vae"
pipe = StableDiffusionXLTilingPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    vae=vae,
    use_safetensors=False, #for yammermix   
    #variant="fp16",
).to("cuda")

#pipe.enable_model_cpu_offload() #<< Enable this if you have limited VRAM
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

#region functions
def select_scheduler(scheduler_name):
    scheduler = scheduler_name.split("-")
    scheduler_class_name = scheduler[0]
    add_kwargs = {"beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear", "num_train_timesteps": 1000}
    if len(scheduler) > 1:
        add_kwargs["use_karras_sigmas"] = True
    if len(scheduler) > 2:
        add_kwargs["algorithm_type"] = "sde-dpmsolver++"
    import diffusers
    scheduler = getattr(diffusers, scheduler_class_name)    
    scheduler = scheduler.from_config(pipe.scheduler.config, **add_kwargs) 
    return scheduler

@spaces.GPU
def predict(left_prompt, center_prompt, right_prompt, negative_prompt, left_gs, center_gs, right_gs, overlap_pixels, steps, generation_seed, scheduler, tile_height, tile_width, target_height, target_width):
    global pipe
    
    # Set selected scheduler
    print(f"Using scheduler: {scheduler}...")
    pipe.scheduler = select_scheduler(scheduler)

    # Set seed
    generator = torch.Generator("cuda").manual_seed(generation_seed)
    
    target_height = int(target_height)
    target_width = int(target_width)
    tile_height = int(tile_height)
    tile_width = int(tile_width)
    
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
        tile_col_overlap=overlap_pixels,        
        guidance_scale_tiles=[[left_gs, center_gs, right_gs]],
        height=target_height,
        width=target_width,
        target_size=(target_height, target_width),
        generator=generator,
        num_inference_steps=steps,
    )["images"][0]

    return image

def calc_tile_size(target_height, target_width, overlap_pixels, max_tile_width_size=1280):
    num_cols=3
    num_rows=1    
    min_tile_dimension=8
    reduction_step=8
    max_tile_height_size=1024
    best_tile_width = 0
    best_tile_height = 0
    best_adjusted_target_width = 0
    best_adjusted_target_height = 0
    found_valid_solution = False

    # Adjust Tile Width
    tile_width = max_tile_width_size
    tile_height = max_tile_height_size

    while tile_width >= min_tile_dimension:
        horizontal_borders = num_cols - 1
        total_horizontal_overlap_pixels = (overlap_pixels * horizontal_borders)
        adjusted_target_width = tile_width * num_cols - total_horizontal_overlap_pixels

        vertical_borders = num_rows - 1
        total_vertical_overlap_pixels = (overlap_pixels * vertical_borders)
        adjusted_target_height = tile_height * num_rows - total_vertical_overlap_pixels

        if tile_width <= max_tile_width_size and adjusted_target_width <= target_width:
            if adjusted_target_width > best_adjusted_target_width:
                best_tile_width = tile_width
                best_adjusted_target_width = adjusted_target_width
                found_valid_solution = True

        tile_width -= reduction_step

    # Adjust Tile Height
    if found_valid_solution:
        tile_width = best_tile_width
        tile_height = max_tile_height_size

        while tile_height >= min_tile_dimension:
            horizontal_borders = num_cols - 1
            total_horizontal_overlap_pixels = (overlap_pixels * horizontal_borders)
            adjusted_target_width = tile_width * num_cols - total_horizontal_overlap_pixels

            vertical_borders = num_rows - 1
            total_vertical_overlap_pixels = (overlap_pixels * vertical_borders)
            adjusted_target_height = tile_height * num_rows - total_vertical_overlap_pixels
        
            if tile_height <= max_tile_height_size and adjusted_target_height <= target_height:
                 if adjusted_target_height > best_adjusted_target_height:
                    best_tile_height = tile_height
                    best_adjusted_target_height = adjusted_target_height

            tile_height -= reduction_step

    new_target_height = best_adjusted_target_height
    new_target_width = best_adjusted_target_width
    tile_width = best_tile_width
    tile_height = best_tile_height

    print("--- TILE SIZE CALCULATED VALUES ---")    
    print(f"Overlap pixels (requested): {overlap_pixels}")
    print(f"Tile Height (divisible by 8, max {max_tile_height_size}): {tile_height}")
    print(f"Tile Width (divisible by 8, max {max_tile_width_size}): {tile_width}")
    print(f"Number of Columns (horizontal tiles): {num_cols}")
    print(f"Number of Rows (vertical tiles): {num_rows}")
    print(f"Original Target Height: {target_height}")
    print(f"Original Target Width: {target_width}")
    print(f"New Target Height (total covered height): {new_target_height}")
    print(f"New Target Width (total covered width): {new_target_width}\n")

    return new_target_height, new_target_width, tile_height, tile_width

def do_calc_tile(target_height, target_width, overlap_pixels, max_tile_size):    
    new_target_height, new_target_width, tile_height, tile_width = calc_tile_size(target_height, target_width, overlap_pixels, max_tile_size)    
    return gr.update(value=tile_height), gr.update(value=tile_width), gr.update(value=new_target_height), gr.update(value=new_target_width)

def clear_result():
    return gr.update(value=None)

def run_for_examples(left_prompt, center_prompt, right_prompt, negative_prompt, left_gs, center_gs, right_gs, overlap_pixels, steps, generation_seed, scheduler, tile_height, tile_width, target_height, target_width, max_tile_width):
    return predict(left_prompt, center_prompt, right_prompt, negative_prompt, left_gs, center_gs, right_gs, overlap_pixels, steps, generation_seed, scheduler, tile_height, tile_width, target_height, target_width)

def randomize_seed_fn(generation_seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        generation_seed = random.randint(0, MAX_SEED)
    return generation_seed

css = """
.gradio-container .fillable { 
    width: 95% !important;
    max-width: unset !important;
}
"""
title = """<h1 align="center">Mixture-of-Diffusers for SDXL Tiling Pipeline🤗</h1>           
           <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; overflow:hidden;">
                <span>This project implements an SDXL pipeline based on the original project, <a href='https://github.com/albarji/mixture-of-diffusers'>Mixture-of-Diffusers</a>. For more information, see the:                
                <a href="https://arxiv.org/pdf/2408.06072">📜 paper </a>
           </div>           
           """

tips = """
### Method
The method proposed here strives to provide a better tool for image composition by using several diffusion processes in parallel, each configured with a specific prompt and settings, and focused on a particular region of the image. The mixture of diffusion processes is done in a way that harmonizes the generation process, preventing "seam" effects in the generated image.
Using several diffusion processes in parallel has also practical advantages when generating very large images, as the GPU memory requirements are similar to that of generating an image of the size of a single tile.
For practical demonstration purposes, this demo only covers image generation using 1x3 tiles. However, in the pipeline, you can freely increase the number of rows and columns as well as specify a row overlap.

### Tips
1. Describe the same environment for all image elements in your prompt. This helps to better harmonize the final image.
2. Keep the same stylization in both prompts.
3. Test different overlap sizes.
4. Test fews increments on seed.
5. This may take a while.
6. Enjoy!
"""

about = """
📧 **Contact**
<br>
If you have any questions or suggestions, feel free to send your question to <b>contact@devaiexp.com</b>.
"""

with gr.Blocks(css=css) as app:
    gr.Markdown(title)    
    with gr.Row():
        with gr.Column(scale=7):
            generate_button = gr.Button("Generate")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Left region")
                    left_prompt = gr.Textbox(lines=4, 
                                            label="Prompt for left side of the image")
                    left_gs = gr.Slider(minimum=0, 
                                        maximum=15, 
                                        value=7, 
                                        step=1, 
                                        label="Left CFG scale")
                with gr.Column(scale=1):
                    gr.Markdown("### Center region")
                    center_prompt = gr.Textbox(lines=4, 
                                            label="Prompt for the center of the image")
                    center_gs = gr.Slider(minimum=0, 
                                        maximum=15, 
                                        value=7, 
                                        step=1, 
                                        label="Center CFG scale")
                with gr.Column(scale=1):
                    gr.Markdown("### Right region")
                    right_prompt = gr.Textbox(lines=4, 
                                            label="Prompt for the right side of the image")
                    right_gs = gr.Slider(minimum=0, 
                                        maximum=15, 
                                        value=7, 
                                        step=1, 
                                        label="Right CFG scale")
            with gr.Row():
                negative_prompt = gr.Textbox(lines=2, 
                                            label="Negative prompt for the image",
                                            value="nsfw, lowres, bad anatomy, bad hands, duplicate, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry")
            with gr.Row():
                result = gr.Image(
                    label="Generated Image",
                    show_label=True, 
                    format="png",
                    interactive=False,
                    # allow_preview=True,
                    # preview=True,
                    scale=1,

                )
        with gr.Column():
            gr.Markdown(tips)
    with gr.Sidebar(label="Parameters", open=True):
        gr.Markdown("### General parameters")
        with gr.Row():
            height = gr.Slider(label="Height", 
                            value=1024,
                            step=8, 
                            visible=True,
                            minimum=512,
                            maximum=1024)            
            width = gr.Slider(label="Width",
                            value=1280,
                            step=8, 
                            visible=True,
                            minimum=512,
                            maximum=3840)
            overlap = gr.Slider(minimum=0, 
                                maximum=512, 
                                value=128, 
                                step=8, 
                                label="Tile Overlap")
            max_tile_size = gr.Dropdown(label="Max. Tile Size", choices=[1024, 1280], value=1280)
            calc_tile = gr.Button("Calculate Tile Size") 
        with gr.Row():                       
            tile_height = gr.Textbox(label="Tile height", value=1024, interactive=False)            
            tile_width = gr.Textbox(label="Tile width", value=1024, interactive=False)
        with gr.Row():
            new_target_height = gr.Textbox(label="New image height", value=1024, interactive=False)
            new_target_width = gr.Textbox(label="New image width", value=1024, interactive=False)
        with gr.Row():
            steps = gr.Slider(minimum=1,
                            maximum=50, 
                            value=30, 
                            step=1, 
                            label="Inference steps")
            
            generation_seed = gr.Slider(label="Seed",
                                        minimum=0,
                                        maximum=MAX_SEED,
                                        step=1,
                                        value=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)        
        with gr.Row():
            scheduler = gr.Dropdown(
                label="Schedulers",
                choices=SCHEDULERS,
                value=SCHEDULERS[0],
            )
    with gr.Row():
        gr.Examples(
            examples=[
                [
                    "Iron Man, repulsor rays blasting enemies in destroyed cityscape, sparks, energy trails, crumbling skyscrapers, smoke, debris, cinematic lighting, photorealistic, intense action. Focus: Iron Man.",
                    "Captain America charging forward, vibranium shield deflecting energy blasts in destroyed cityscape, collapsing buildings, rubble streets, battle-damaged suit, determined expression, distant explosions, cinematic composition, realistic rendering. Focus: Captain America.",
                    "Thor wielding Stormbreaker in destroyed cityscape, lightning crackling, powerful strike downwards, shattered buildings, burning debris, ground trembling, Asgardian armor, cinematic photography, realistic details. Focus: Thor.",
                    negative_prompt.value,
                    5, 5, 5,
                    160,
                    30,
                    1328797844,
                    "UniPCMultistepScheduler",
                    1024,
                    1280,
                    1024,                       
                    3840,
                    1024
                ],
                [
                    "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                    "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                    "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                    negative_prompt.value,
                    7, 7, 7,
                    256,
                    30,
                    297984183,
                    "DPMSolverMultistepScheduler-Karras-SDE",
                    1024,
                    1280,
                    1024,                       
                    3840,
                    1280
                ],
                [
                    "Abstract decorative illustration, by joan miro and gustav klimt and marlina vera and loish, elegant, intricate, highly detailed, smooth, sharp focus, vibrant colors, artstation, stunning masterpiece",
                    "Abstract decorative illustration, by joan miro and gustav klimt and marlina vera and loish, elegant, intricate, highly detailed, smooth, sharp focus, vibrant colors, artstation, stunning masterpiece",
                    "Abstract decorative illustration, by joan miro and gustav klimt and marlina vera and loish, elegant, intricate, highly detailed, smooth, sharp focus, vibrant colors, artstation, stunning masterpiece",
                    negative_prompt.value,
                    7, 7, 7,
                    128,
                    30,
                    580541206,
                    "LMSDiscreteScheduler",
                    1024,
                    768,
                    1024,     
                    2048,
                    1280
                ],
                [
                    "Magical diagrams and runes written with chalk on a blackboard, elegant, intricate, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                    "Magical diagrams and runes written with chalk on a blackboard, elegant, intricate, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                    "Magical diagrams and runes written with chalk on a blackboard, elegant, intricate, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                    negative_prompt.value,
                    9, 9, 9,
                    128,
                    30,
                    12591765619,
                    "LMSDiscreteScheduler",
                    1024,
                    768,
                    1024,                                        
                    2048,
                    1280
                ]
            ],
            inputs=[left_prompt, center_prompt, right_prompt, negative_prompt, left_gs, center_gs, right_gs, overlap, steps, generation_seed, scheduler, tile_height, tile_width, height, width, max_tile_size],
            fn=run_for_examples,
            outputs=result,
            cache_examples=True
        )
       
    event_calc_tile_size={"fn": do_calc_tile, "inputs":[height, width, overlap, max_tile_size], "outputs":[tile_height, tile_width, new_target_height, new_target_width]}
    calc_tile.click(**event_calc_tile_size)
    
    generate_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(**event_calc_tile_size
    ).then(                        
        fn=randomize_seed_fn,
        inputs=[generation_seed, randomize_seed],
        outputs=generation_seed,
        queue=False,
        api_name=False,
    ).then(
        fn=predict,
        inputs=[left_prompt, center_prompt, right_prompt, negative_prompt, left_gs, center_gs, right_gs, overlap, steps, generation_seed, scheduler, tile_height, tile_width, new_target_height, new_target_width],
        outputs=result,
    )
    gr.Markdown(about)
app.launch(share=False)
