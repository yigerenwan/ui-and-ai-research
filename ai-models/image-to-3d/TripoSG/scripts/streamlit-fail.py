import argparse
import os
import sys
from glob import glob
from typing import Any, Union, Optional

# Fix for numpy binary incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import torch
# Import trimesh later after warnings are suppressed
from huggingface_hub import snapshot_download
from PIL import Image
import streamlit as st
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import trimesh after warnings are suppressed
import trimesh

# Change this line to use a relative import or create the module inline
# from image_process import prepare_image
# from briarmbg import BriaRMBG

# Define prepare_image function directly in this file
def prepare_image(
    image_input: Union[str, Image.Image],
    bg_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
    rmbg_net: Optional[Any] = None,
    target_size: int = 512,
) -> Image.Image:
    """
    Prepare an image for processing by TripoSG.
    """
    # Load image if path is provided
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")
    
    # Resize image while preserving aspect ratio
    width, height = img.size
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new image with the target size and background color
    bg_color_uint8 = (np.clip(bg_color, 0, 1) * 255).astype(np.uint8)
    new_img = Image.new("RGB", (target_size, target_size), tuple(bg_color_uint8))
    
    # Paste the resized image onto the center of the new image
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    # Remove background if rmbg_net is provided
    if rmbg_net is not None:
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(new_img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(rmbg_net.device)
        
        # Get alpha mask
        with torch.no_grad():
            alpha = rmbg_net(img_tensor)
        
        # Convert back to numpy
        alpha = alpha[0, 0].cpu().numpy()
        
        # Apply alpha mask
        img_array = np.array(new_img)
        bg_color_expanded = bg_color.reshape(1, 1, 3) * 255
        img_array = img_array * alpha[:, :, np.newaxis] + bg_color_expanded * (1 - alpha[:, :, np.newaxis])
        new_img = Image.fromarray(img_array.astype(np.uint8))
    
    return new_img

# Define BriaRMBG class directly in this file
class BriaRMBG(torch.nn.Module):
    """
    Background removal model based on BriaAI's RMBG model.
    """
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    
    @classmethod
    def from_pretrained(cls, model_path, model_filename="model.jit"):
        """Load model from pretrained weights"""
        instance = cls()
        model_file_path = os.path.join(model_path, model_filename)
        instance.model = torch.jit.load(model_file_path, map_location=instance.device)
        return instance
    
    def to(self, device):
        """Move model to specified device"""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def forward(self, x):
        """Forward pass through the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call from_pretrained first.")
        return self.model(x)

from triposg.pipelines.pipeline_triposg import TripoSGPipeline


@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
) -> trimesh.Scene:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))    
    return mesh


def main():
    st.title("Image to 3D Model Converter")
    st.write("Upload an image to convert it to a 3D model using TripoSG")
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    seed = st.sidebar.number_input("Seed", value=42, min_value=0, max_value=1000000)
    num_inference_steps = st.sidebar.slider("Number of Inference Steps", min_value=10, max_value=100, value=50)
    guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.0, step=0.5)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Generate 3D Model"):
            with st.spinner("Processing... This may take a while."):
                try:
                    # Set device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    
                    # Download pretrained weights if not already downloaded
                    triposg_weights_dir = "pretrained_weights/TripoSG"
                    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
                    
                    if not os.path.exists(triposg_weights_dir):
                        st.text("Downloading TripoSG weights...")
                        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
                    
                    # Check for RMBG model file
                    rmbg_model_path = os.path.join(rmbg_weights_dir, "model.jit")
                    if not os.path.exists(rmbg_model_path):
                        st.text("Downloading RMBG weights...")
                        # Force re-download by removing the directory if it exists
                        if os.path.exists(rmbg_weights_dir):
                            import shutil
                            shutil.rmtree(rmbg_weights_dir)
                        
                        # Download with allow_patterns to ensure we get all necessary files
                        snapshot_download(
                            repo_id="briaai/RMBG-1.4", 
                            local_dir=rmbg_weights_dir,
                        )
                        
                        # Look for model.jit in the downloaded directory and its subdirectories
                        model_files = []
                        for root, dirs, files in os.walk(rmbg_weights_dir):
                            for file in files:
                                if file.endswith(".jit") or file.endswith(".pt"):
                                    model_files.append(os.path.join(root, file))
                        
                        if model_files:
                            # Use the first found model file
                            rmbg_model_path = model_files[0]
                            st.text(f"Found model file: {rmbg_model_path}")
                        else:
                            st.error(f"Failed to find model file. Please check the repository structure at https://huggingface.co/briaai/RMBG-1.4")
                            # List files in the directory to debug
                            st.text(f"Files in {rmbg_weights_dir}:")
                            for root, dirs, files in os.walk(rmbg_weights_dir):
                                for file in files:
                                    st.text(f"  - {os.path.join(root, file)}")
                            return
                    
                    # Initialize models
                    st.text("Initializing models...")
                    # Update BriaRMBG.from_pretrained to use the exact model path
                    rmbg_net = BriaRMBG.from_pretrained(os.path.dirname(rmbg_model_path), model_filename=os.path.basename(rmbg_model_path)).to(device)
                    rmbg_net.eval()
                    
                    pipe = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)
                    
                    # Run inference
                    st.text("Generating 3D model...")
                    mesh = run_triposg(
                        pipe,
                        image_input=image,
                        rmbg_net=rmbg_net,
                        seed=seed,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    )
                    
                    # Save the mesh to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
                        mesh.export(tmp_file.name)
                        
                        # Provide download link
                        with open(tmp_file.name, "rb") as f:
                            st.download_button(
                                label="Download 3D Model (GLB)",
                                data=f,
                                file_name="triposg_model.glb",
                                mime="model/gltf-binary"
                            )
                        
                        # Clean up the temporary file
                        os.unlink(tmp_file.name)
                    
                    st.success("3D model generated successfully!")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
