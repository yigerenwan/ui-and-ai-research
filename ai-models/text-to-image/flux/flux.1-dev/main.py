import streamlit as st
import torch
from diffusers import FluxPipeline
from PIL import Image
import io
from datetime import datetime

st.set_page_config(page_title="FLUX.1 Image Generator", layout="wide")
st.title("FLUX.1 Image Generator")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device=device)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe

pipe = load_model()

# Sidebar for parameters
st.sidebar.header("Generation Parameters")

prompt = st.sidebar.text_area(
    "Prompt",
    value="A cat holding a sign that says hello world",
    height=200
)

# Add aspect ratio selector
aspect_ratio = st.sidebar.selectbox(
    "Aspect Ratio",
    options=[
        "1:1 (1024x1024)",
        "9:16 (768x1344)",
        "3:4 (864x1152)",
        "16:9 (1344x768)",
        "4:3 (1152x864)",
        "2:1 (1440x720)",
        "1:2 (720x1440)",
        "Custom"
    ],
    index=0
)

# Define dimensions based on aspect ratio
if aspect_ratio != "Custom":
    if aspect_ratio == "1:1 (1024x1024)":
        width = height = 1024
    elif aspect_ratio == "9:16 (768x1344)":
        width, height = 768, 1344
    elif aspect_ratio == "3:4 (864x1152)":
        width, height = 864, 1152
    elif aspect_ratio == "16:9 (1344x768)":
        width, height = 1344, 768
    elif aspect_ratio == "4:3 (1152x864)":
        width, height = 1152, 864
    elif aspect_ratio == "2:1 (1440x720)":
        width, height = 1440, 720
    elif aspect_ratio == "1:2 (720x1440)":
        width, height = 720, 1440

if aspect_ratio == "Custom":
    width = st.sidebar.select_slider("Width", options=[256, 512, 768, 1024, 1280, 1344, 1440], value=1024)
    height = st.sidebar.select_slider("Height", options=[256, 512, 720, 768, 864, 1024, 1152, 1280, 1344, 1440], value=1024)
else:
    st.sidebar.text(f"Width: {width}px")
    st.sidebar.text(f"Height: {height}px")

guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=3.5, step=0.1)
num_inference_steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=100, value=50, step=5)
num_images = st.sidebar.slider("Number of Images", min_value=1, max_value=4, value=1, step=1)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=0)

# Create two columns for main content and examples
main_col, example_col = st.columns([2, 1])

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# Generate button
if st.sidebar.button("Generate Image"):
    with st.spinner("🎨 Preparing to generate your masterpiece..."):
        try:
            progress_bar = st.progress(0)
            progress_bar.progress(20)
            st.markdown("🔄 Loading model settings...")
            
            progress_bar.progress(40)
            st.markdown("✨ Generating your image...")
            
            images = []
            
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            for i in range(num_images):
                # Create a new generator for each image with an incremented seed
                current_seed = seed + i
                generator = torch.Generator("cpu").manual_seed(current_seed)
                
                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=512,
                    generator=generator
                ).images[0]
                images.append(image)
                
                # Clear CUDA cache after each image generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            st.session_state.generated_images = images
            
            progress_bar.progress(80)
            st.markdown("🎉 Adding final touches...")
            
            progress_bar.progress(100)
            st.success("✨ Generation complete! Your images are ready.")
            
        except Exception as e:
            st.error(f"❌ Error generating image: {str(e)}")

# Display images in main column
with main_col:
    if st.session_state.generated_images:
        for i, image in enumerate(st.session_state.generated_images):
            fixed_width = min(width, 600)
            st.image(image, caption=f"Generated Image {i+1}", width=fixed_width)
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_{timestamp}_{i+1}.png"
            
            st.download_button(
                label="Download Image",
                data=img_byte_arr,
                file_name=filename,
                mime="image/png"
            )

# Example prompts in right column
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
    
    for prompt in example_prompts:
        st.code(prompt, language="")

# Display instructions
st.sidebar.markdown("---")
st.sidebar.markdown("## Instructions")
st.sidebar.markdown("1. Enter your prompt in the text area")
st.sidebar.markdown("2. Adjust the parameters as needed")
st.sidebar.markdown("3. Click 'Generate Image' to create your image")
st.sidebar.markdown("4. Download the generated image using the button below each image")