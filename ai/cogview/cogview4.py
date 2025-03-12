from diffusers import CogView4Pipeline
import torch
import streamlit as st
from PIL import Image
import io
import os
from datetime import datetime

st.set_page_config(page_title="CogView4 Image Generator", layout="wide")
st.title("CogView4 Image Generator")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16, device=device)
    
    # Open it for reduce GPU memory usage
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    return pipe

pipe = load_model()

# Sidebar for parameters
st.sidebar.header("Generation Parameters")

prompt = st.sidebar.text_area(
    "Prompt",
    value="A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection.",
    height=200
)

# Add aspect ratio selector before width/height
aspect_ratio = st.sidebar.selectbox(
    "Aspect Ratio",
    options=[
        "1:1 (Square)",
        "4:3 (Standard)",
        "16:9 (Widescreen)",
        "9:16 (Portrait)",
        "3:2 (Photo)",
        "2:3 (Portrait Photo)",
        "Custom"
    ],
    index=0
)

# Define base resolution and calculate dimensions based on aspect ratio
base_resolution = 1024
if aspect_ratio != "Custom":
    if aspect_ratio == "1:1 (Square)":
        width = height = base_resolution
    elif aspect_ratio == "4:3 (Standard)":
        width = base_resolution
        height = int(base_resolution * (3/4))
    elif aspect_ratio == "16:9 (Widescreen)":
        width = base_resolution
        height = int(base_resolution * (9/16))
    elif aspect_ratio == "9:16 (Portrait)":
        width = int(base_resolution * (9/16))
        height = base_resolution
    elif aspect_ratio == "3:2 (Photo)":
        width = base_resolution
        height = int(base_resolution * (2/3))
    elif aspect_ratio == "2:3 (Portrait Photo)":
        width = int(base_resolution * (2/3))
        height = base_resolution

# Show custom width/height sliders only if Custom is selected
if aspect_ratio == "Custom":
    width = st.sidebar.select_slider("Width", options=[512, 768, 1024, 1280], value=1024)
    height = st.sidebar.select_slider("Height", options=[512, 768, 1024, 1280], value=1024)
else:
    # Display the calculated dimensions (read-only)
    st.sidebar.text(f"Width: {width}px")
    st.sidebar.text(f"Height: {height}px")

guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
num_inference_steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=100, value=10, step=5)
num_images = st.sidebar.slider("Number of Images", min_value=1, max_value=4, value=1, step=1)

# Create two columns for main content and examples (moved up before image generation)
main_col, example_col = st.columns([2, 1])

# Generate button
if st.sidebar.button("Generate Image"):
    with st.spinner("🎨 Preparing to generate your masterpiece..."):
        try:
            # Add progress bar
            progress_bar = st.progress(0)
            
            # Update progress for model preparation
            progress_bar.progress(20)
            st.markdown("🔄 Loading model settings...")
            
            # Generate images
            progress_bar.progress(40)
            st.markdown("✨ Generating your image...")
            
            images = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
            ).images
            
            # Update progress for post-processing
            progress_bar.progress(80)
            st.markdown("🎉 Adding final touches...")
            
            # Display images in main column with fixed width
            with main_col:
                for i, image in enumerate(images):
                    # Calculate fixed width based on number of images
                    fixed_width = min(width, 600)  # Max width of 400px
                    st.image(image, caption=f"Generated Image {i+1}", width=fixed_width)
                    
                    # Save image button
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"cogview4_{timestamp}_{i+1}.png"
                    
                    st.download_button(
                        label="Download Image",
                        data=img_byte_arr,
                        file_name=filename,
                        mime="image/png"
                    )
            
            # Complete progress
            progress_bar.progress(100)
            st.success("✨ Generation complete! Your images are ready.")
            
        except Exception as e:
            st.error(f"❌ Error generating image: {e}")
            
# Example prompts in right column (moved up)
with example_col:
    st.markdown("### 📝 Prompt Examples")
    example_prompts = [
        "A young woman in a flowing white dress walking through a field of lavender at sunset, her hair gently moving in the breeze.",
        "A distinguished gentleman in a tailored suit standing in front of a modern art gallery, examining a striking abstract painting.",
        "A fashion-forward person sitting at a Parisian café, wearing oversized sunglasses and a chic summer outfit.",
        "A majestic dragon soaring through a sunset sky, scales gleaming with iridescent colors, casting long shadows over a medieval castle below.",
        "A cozy cafe interior with warm lighting, steam rising from coffee cups, and rain drops on window panes creating a peaceful atmosphere.",
        "A futuristic cityscape at night with neon lights, flying vehicles, and towering holographic advertisements reflecting in the wet streets.",
        "A skilled craftsman working in their workshop, carefully shaping wood with traditional tools under warm lighting.",
        "A cosmic scene showing the birth of a new galaxy, with swirling nebulae, stardust, and celestial bodies in vibrant colors.",
        "An adventurous mountain climber reaching the summit at dawn, silhouetted against a dramatic sunrise."
    ]
    
    # Display prompts
    for prompt in example_prompts:
        st.code(prompt, language="")

# Display instructions
st.sidebar.markdown("---")
st.sidebar.markdown("## Instructions")
st.sidebar.markdown("1. Enter your prompt in the text area")
st.sidebar.markdown("2. Adjust the parameters as needed")
st.sidebar.markdown("3. Click 'Generate Image' to create your image")
st.sidebar.markdown("4. Download the generated image using the button below each image")