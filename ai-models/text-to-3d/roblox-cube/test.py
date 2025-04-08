import torch
import trimesh
from cube3d.inference.engine import Engine, EngineFast

# load ckpt
config_path = "cube3d/configs/open_model.yaml"
gpt_ckpt_path = "model_weights/shape_gpt.safetensors"
shape_ckpt_path = "model_weights/shape_tokenizer.safetensors"
engine_fast = EngineFast( # only supported on CUDA devices, replace with Engine otherwise
    config_path, 
    gpt_ckpt_path, 
    shape_ckpt_path, 
    device=torch.device("cuda"),
)

# inference
input_prompt = "A pair of noise-canceling headphones"
# NOTE: Reduce `resolution_base` for faster inference and lower VRAM usage
# The `top_p` parameter controls randomness between inferences:
#   Float < 1: Keep smallest set of tokens with cumulative probability ≥ top_p. Default None: deterministic generation.
mesh_v_f = engine_fast.t2s([input_prompt], use_kv_cache=True, resolution_base=8.0, top_p=0.9)

# save output
vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
_ = trimesh.Trimesh(vertices=vertices, faces=faces).export("output.obj")