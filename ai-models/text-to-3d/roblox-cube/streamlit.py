import torch
import trimesh
import streamlit as st
from cube3d.inference.engine import Engine, EngineFast
import os
import matplotlib.pyplot as plt

st.title("Text to 3D Model Generator")
st.write("Generate 3D models from text descriptions using Roblox's Cube3D model")

# Sidebar for configuration
st.sidebar.header("Configuration")
resolution = st.sidebar.slider("Resolution Base", min_value=4.0, max_value=16.0, value=8.0, step=1.0, 
                              help="Higher values give better quality but use more VRAM and take longer")
top_p_value = st.sidebar.slider("Top P (Randomness)", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                               help="Controls randomness: lower is more deterministic, higher is more creative")
use_cuda = st.sidebar.checkbox("Use CUDA (if available)", value=True)

# Model loading (with loading indicator)
@st.cache_resource
def load_model(use_cuda):
    with st.spinner("Loading model... This may take a minute."):
        config_path = "cube3d/configs/open_model.yaml"
        gpt_ckpt_path = "model_weights/shape_gpt.safetensors"
        shape_ckpt_path = "model_weights/shape_tokenizer.safetensors"
        
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            engine = EngineFast(
                config_path, 
                gpt_ckpt_path, 
                shape_ckpt_path, 
                device=device,
            )
        else:
            st.warning("CUDA not available, using CPU with standard Engine (slower)")
            engine = Engine(
                config_path, 
                gpt_ckpt_path, 
                shape_ckpt_path, 
                device=device,
            )
        return engine

# Function to export mesh as GLB
def export_as_glb(vertices, faces, output_path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Export as GLB
    mesh.export(output_path, file_type='glb')
    return output_path

# Load the model
engine = load_model(use_cuda)

# Input for text prompt
input_prompt = st.text_area("Enter your description", 
                           placeholder="Describe what 3D model you want to generate...",
                           height=100)

# Generate button
if st.button("Generate 3D Model"):
    if not input_prompt:
        st.error("Please enter a description")
    else:
        with st.spinner(f"Generating 3D model for: '{input_prompt}'"):
            try:
                # Inference
                mesh_v_f = engine.t2s(
                    [input_prompt], 
                    use_kv_cache=True, 
                    resolution_base=float(resolution), 
                    top_p=None if top_p_value == 0 else float(top_p_value)
                )
                
                # Save output
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
                
                # Save as GLB for download
                glb_path = os.path.join(output_dir, f"{input_prompt.replace(' ', '_')[:20]}.glb")
                export_as_glb(vertices, faces, glb_path)
                
                st.success(f"3D model generated successfully!")
                
                # Display information about the model
                st.subheader("3D Model Information")
                st.write(f"Vertices: {len(vertices)}")
                st.write(f"Faces: {len(faces)}")
                
                # Create a simple visualization of the model (top view)
                st.subheader("Model Preview (simplified)")
                fig, ax = plt.subplots()
                ax.scatter(vertices[:, 0], vertices[:, 1], s=1, alpha=0.5)
                ax.set_aspect('equal')
                ax.set_title("Top view (X-Y projection)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                st.pyplot(fig)
                
                # Provide download link for GLB
                with open(glb_path, "rb") as file:
                    st.download_button(
                        label="Download GLB file",
                        data=file,
                        file_name=f"{input_prompt.replace(' ', '_')[:20]}.glb",
                        mime="application/octet-stream"
                    )
                
                st.info("To view the 3D model, download the GLB file and open it with a 3D viewer like Windows 3D Viewer, Blender, or online viewers.")
                    
            except Exception as e:
                st.error(f"Error generating model: {str(e)}")

st.markdown("---")
st.markdown("Powered by Roblox's Cube3D model")