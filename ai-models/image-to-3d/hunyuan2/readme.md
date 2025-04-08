# Hunyuan2 - Forked from https://github.com/Tencent/Hunyuan3D-2

### Install Requirements

Please install Pytorch via the [official](https://pytorch.org/) site. Then install the other requirements via

```bash
pip install -r requirements.txt
pip install -e .
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

### Code Usage

We designed a diffusers-like API to use our shape generation model - Hunyuan3D-DiT and texture synthesis model -
Hunyuan3D-Paint.

You could assess **Hunyuan3D-DiT** via:

```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]
```

The output mesh is a [trimesh object](https://trimesh.org/trimesh.html), which you could save to glb/obj (or other
format) file.

For **Hunyuan3D-Paint**, do the following:

```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')
```

Please visit [examples](examples) folder for more advanced usage, such as **multiview image to 3D generation** and *
*texture generation
for handcrafted mesh**.

### Gradio App

You could also host a [Gradio](https://www.gradio.app/) App in your own computer via:

Standard Version

```bash
# Hunyuan3D-2mini
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mini --subfolder hunyuan3d-dit-v2-mini --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode
# Hunyuan3D-2mv
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode
# Hunyuan3D-2
python3 gradio_app.py --model_path tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0 --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode
```

Turbo Version

```bash
# Hunyuan3D-2mini
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mini --subfolder hunyuan3d-dit-v2-mini-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
# Hunyuan3D-2mv
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
# Hunyuan3D-2
python3 gradio_app.py --model_path tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
```