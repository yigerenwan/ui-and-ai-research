# MV-Adapter - forked from huanngzh/MV-Adapter

## Installation

Clone the repo first:

```Bash
git clone https://github.com/huanngzh/MV-Adapter.git
cd MV-Adapter
```

(Optional) Create a fresh conda env:

```Bash
conda create -n mvadapter python=3.10
conda activate mvadapter
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install -r requirements.txt
```

## Notes

### System Requirements

In the model zoo of MV-Adapter, running image-to-multiview generation has the highest system requirements, which requires about 14G GPU memory.

## Usage: Multiview Generation

### Launch Demo

#### Text to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.gradio_demo_t2mv --base_model "stabilityai/stable-diffusion-xl-base-1.0"
```

> Reminder: When switching the demo to another base model, delete the `gradio_cached_examples` directory, otherwise it will affect the examples results of the next demo.

**With anime-themed <a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1" target="_blank">Animagine XL 3.1</a>:**

```Bash
python -m scripts.gradio_demo_t2mv --base_model "cagliostrolab/animagine-xl-3.1"
```

**With general <a href="https://huggingface.co/Lykon/dreamshaper-xl-1-0" target="_blank">Dreamshaper</a>:**

```Bash
python -m scripts.gradio_demo_t2mv --base_model "Lykon/dreamshaper-xl-1-0" --scheduler ddpm
```


You can also specify a new diffusers-format text-to-image diffusion model using `--base_model`. Note that it should be the model name in huggingface, such as `stabilityai/stable-diffusion-xl-base-1.0`, or a local path refer to a text-to-image pipeline directory. Note that if you specify `latent-consistency/lcm-sdxl` to use latent consistency models, please add `--scheduler lcm` to the command.

#### Image to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.gradio_demo_i2mv
```

### Inference Scripts

We recommend that experienced users check the files in the scripts directory to adjust the parameters appropriately to try the best "card drawing" results.

#### Text to Multiview Generation

Note that you can specify a diffusers-format text-to-image diffusion model as the base model using `--base_model xxx`. It should be the model name in huggingface, such as `stabilityai/stable-diffusion-xl-base-1.0`, or a local path refer to a text-to-image pipeline directory.

**With SDXL:**

```Bash
python -m scripts.inference_t2mv_sdxl --text "an astronaut riding a horse" \
--seed 42 \
--output output.png
```

**With personalized models:**

anime-themed <a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1" target="_blank">Animagine XL 3.1</a>

```Bash
python -m scripts.inference_t2mv_sdxl --base_model "cagliostrolab/animagine-xl-3.1" \
--text "1girl, izayoi sakuya, touhou, solo, maid headdress, maid, apron, short sleeves, dress, closed mouth, white apron, serious face, upper body, masterpiece, best quality, very aesthetic, absurdres" \
--seed 0 \
--output output.png
```

general <a href="https://huggingface.co/Lykon/dreamshaper-xl-1-0" target="_blank">Dreamshaper</a>

```Bash
python -m scripts.inference_t2mv_sdxl --base_model "Lykon/dreamshaper-xl-1-0" \
--scheduler ddpm \
--text "the warrior Aragorn from Lord of the Rings, film grain, 8k hd" \
--seed 0 \
--output output.png
```

realistic <a href="https://huggingface.co/stablediffusionapi/real-dream-sdxl" target="_blank">real-dream-sdxl</a>

```Bash
python -m scripts.inference_t2mv_sdxl --base_model "stablediffusionapi/real-dream-sdxl" \
--scheduler ddpm \
--text "macro shot, parrot, colorful, dark shot, film grain, extremely detailed" \
--seed 42 \
--output output.png
```

**With <a href="https://huggingface.co/latent-consistency/lcm-sdxl" target="_blank">LCM</a>:**

```Bash
python -m scripts.inference_t2mv_sdxl --unet_model "latent-consistency/lcm-sdxl" \
--scheduler lcm \
--text "Samurai koala bear" \
--num_inference_steps 8 \
--seed 42 \
--output output.png
```

**With LoRA:**

stylized lora <a href="https://huggingface.co/goofyai/3d_render_style_xl" target="_blank">3d_render_style_xl</a>

```Bash
python -m scripts.inference_t2mv_sdxl --lora_model "goofyai/3d_render_style_xl/3d_render_style_xl.safetensors" \
--text "3d style, a fox with flowers around it" \
--seed 20 \
--lora_scale 1.0 \
--output output.png
```

**With ControlNet:**

Scribble to Multiview with <a href="https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0" target="_blank">controlnet-scribble-sdxl-1.0</a>

```Bash
python -m scripts.inference_scribble2mv_sdxl --text "A 3D model of Finn the Human from the animated television series Adventure Time. He is wearing his iconic blue shirt and green backpack and has a neutral expression on his face. He is standing in a relaxed pose with his left foot slightly forward and his right foot back. His arms are at his sides and his head is turned slightly to the right. The model is made up of simple shapes and has a stylized, cartoon-like appearance. It is textured to resemble the character's appearance in the show." \
--seed 0 \
--output output.png \
--guidance_scale 5.0 \
--controlnet_images "assets/demo/scribble2mv/color_0000.webp" "assets/demo/scribble2mv/color_0001.webp" "assets/demo/scribble2mv/color_0002.webp" "assets/demo/scribble2mv/color_0003.webp" "assets/demo/scribble2mv/color_0004.webp" "assets/demo/scribble2mv/color_0005.webp" \
--controlnet_conditioning_scale 0.7
```

**With SD2.1:**

> SD2.1 has lower demand for computing resources and higher inference speed, but a bit lower performance than SDXL.
> In our tests, ddpm scheduler works better than other schedulers here.

```Bash
python -m scripts.inference_t2mv_sd --text "a corgi puppy" \
--seed 42 --scheduler ddpm \
--output output.png
```

#### Image to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.inference_i2mv_sdxl \
--image assets/demo/i2mv/A_decorative_figurine_of_a_young_anime-style_girl.png \
--text "A decorative figurine of a young anime-style girl" \
--seed 21 --output output.png --remove_bg
```

**With LCM:**

```Bash
python -m scripts.inference_i2mv_sdxl \
--unet_model "latent-consistency/lcm-sdxl" \
--scheduler lcm \
--image assets/demo/i2mv/A_juvenile_emperor_penguin_chick.png \
--text "A juvenile emperor penguin chick" \
--num_inference_steps 8 \
--seed 0 --output output.png --remove_bg
```

**With SD2.1:** (lower demand for computing resources and higher inference speed)

> In our tests, ddpm scheduler works better than other schedulers here.

```Bash
python -m scripts.inference_i2mv_sd \
--image assets/demo/i2mv/A_decorative_figurine_of_a_young_anime-style_girl.png \
--text "A decorative figurine of a young anime-style girl" \
--output output.png --remove_bg --scheduler ddpm
```

#### Text-Geometry to Multiview Generation

**Importantly**, when using geometry-condition generation, please make sure that the orientation of the mesh you provide is consistent with the following example. Otherwise, you need to adjust the angles in the scripts when rendering the view.

**With SDXL:**

```Bash
python -m scripts.inference_tg2mv_sdxl \
--mesh assets/demo/tg2mv/ac9d4e4f44f34775ad46878ba8fbfd86.glb \
--text "Mater, a rusty and beat-up tow truck from the 2006 Disney/Pixar animated film 'Cars', with a rusty brown exterior, big blue eyes."
```


```Bash
python -m scripts.inference_tg2mv_sdxl \
--mesh assets/demo/tg2mv/b5f0f0f33e3644d1ba73576ceb486d42.glb \
--text "Optimus Prime, a character from Transformers, with blue, red and gray colors, and has a flame-like pattern on the body"
```

#### Image-Geometry to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.inference_ig2mv_sdxl \
--image assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.jpeg \
--mesh assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.glb \
--output output.png --remove_bg
```

![example_ig2mv_out](assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c_mv.png)

#### Partial Image + Geometry to Multiview

**With SDXL:**

```Bash
python -m scripts.inference_ig2mv_partial_sdxl \
--image assets/demo/ig2mv/cartoon_style_table.png \
--mesh assets/demo/ig2mv/cartoon_style_table.glb \
--output output.png
```

