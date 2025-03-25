import streamlit as st
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
from PIL import Image
import torch
from threading import Thread
import tempfile
import os

# Try to import moviepy, show installation instructions if not available
try:
    # Try different import paths for moviepy
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        # Alternative import path
        from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Page config
st.set_page_config(page_title="Gemma-3 Image Chat", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to maintain chat history limit
def maintain_chat_history(messages, max_history=10):
    if len(messages) > max_history:
        return messages[-max_history:]
    return messages

# Initialize model and processor FIRST
@st.cache_resource
def load_model():
    model_id = "google/gemma-3-12b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# Load model immediately after definition
model, processor = load_model()

# Set up the Streamlit page
st.title("Gemma-3 Visual Chat")

# Update the CSS to better position the chat input
st.markdown("""
    <style>
        /* Main container adjustments */
        .main .block-container {
            padding-bottom: 100px !important;
        }

        /* Fixed chat input container */
        div[data-testid="stChatInput"] {
            position: fixed !important;
            bottom: 0 !important;
            left: 33.33% !important;  /* Accounts for the left column width */
            right: 0 !important;
            padding: 1rem !important;
            background-color: white !important;
            z-index: 100 !important;
            border-top: 1px solid #ddd !important;
        }

        /* Ensure chat messages don't get hidden */
        [data-testid="stChatMessageContainer"] {
            margin-bottom: 80px !important;
            overflow-y: auto !important;
            max-height: calc(100vh - 200px) !important;
        }

        /* Responsive adjustments */
        @media (max-width: 992px) {
            div[data-testid="stChatInput"] {
                left: 40% !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Create a main container for proper spacing
main_container = st.container()

with main_container:
    # Create two columns with fixed ratio
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # Left column content
        media_type = st.radio("Select media type:", ["Image", "Video"])
        
        if media_type == "Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                media_for_model = image
        else:  # Video
            if not MOVIEPY_AVAILABLE:
                st.error("Video processing requires the moviepy package. Please install it with: `pip install moviepy`")
                st.info("After installation, restart the application.")
            else:
                uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
                if uploaded_file:
                    # Save the uploaded video to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        video_path = tmp_file.name
                    
                    # Display the video
                    st.video(uploaded_file)
                    
                    # Extract frames from the video
                    try:
                        with VideoFileClip(video_path) as video:
                            # Get video duration
                            duration = video.duration
                            
                            # Extract 9 frames from the video
                            num_frames = 30
                            
                            # Extract frames at different points in the video
                            frames = []
                            frame_times = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]
                            
                            st.write(f"Extracting {num_frames} frames for analysis")
                            
                            # Create a 3x3 grid for displaying frames
                            cols = st.columns(3)
                            
                            for i, frame_time in enumerate(frame_times):
                                frame = video.get_frame(frame_time)
                                frame_image = Image.fromarray(frame)
                                frames.append(frame_image)
                                
                                # Display in a 3x3 grid
                                col_idx = i % 3
                                with cols[col_idx]:
                                    st.image(frame_image, caption=f"Frame {i+1} ({frame_time:.1f}s)", use_container_width=True)
                            
                            # Use the middle frame as the primary frame for the model
                            media_for_model = frames[len(frames)//2]
                            
                            # Store all frames for potential use
                            st.session_state.video_frames = frames
                            
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                    
                    # Clean up the temporary file
                    try:
                        os.unlink(video_path)
                    except:
                        pass
        
        context = st.text_area(
            "Context (optional, max 10000 words)",
            height=100,
            max_chars=10000,
            help="Add additional context for the conversation"
        )

    with col2:
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display all messages from the session state
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Chat input (will be positioned by CSS)
        if prompt := st.chat_input("What would you like to discuss?"):
            # Add user message to chat and display immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Show assistant response with loading indicator
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    # Prepare messages for model
                    messages = [
                        {
                            "role": "user",
                            "content": []
                        }
                    ]

                    # Add media to message content if uploaded
                    if uploaded_file and 'media_for_model' in locals():
                        # For now, we can only send one image to the model
                        # Most multimodal models only accept a single image per prompt
                        messages[-1]["content"].append({"type": "image", "image": media_for_model})
                        
                        # Add note about multiple frames if this is a video
                        if media_type == "Video" and hasattr(st.session_state, 'video_frames') and len(st.session_state.video_frames) > 1:
                            frame_note = f"This is a video with {len(st.session_state.video_frames)} key frames extracted. I'm analyzing the main frame (frame {len(st.session_state.video_frames)//2 + 1})."
                            messages[-1]["content"].append({"type": "text", "text": frame_note})
                    
                    # Add context if provided
                    if context:
                        messages[-1]["content"].append({"type": "text", "text": context})
                    
                    # Add text prompt to message content
                    messages[-1]["content"].append({"type": "text", "text": prompt})

                    # Process input and generate response
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"
                    ).to(model.device, dtype=torch.bfloat16)

                    streamer = TextIteratorStreamer(processor, skip_special_tokens=True)

                    generation_kwargs = dict(
                        inputs,
                        streamer=streamer,
                        max_new_tokens=1024,
                        do_sample=False,
                    )

                    # Start the generation in a separate thread
                    thread = Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()

                    # Update the output in real-time
                    for new_text in streamer:
                        full_response += new_text
                        # Clean the response to remove any prefixes
                        cleaned_response = full_response
                        # Remove any "user [text] model" pattern
                        if "user" in cleaned_response and "model" in cleaned_response:
                            cleaned_response = cleaned_response.split("model", 1)[1].strip()
                        # If only "model" prefix exists
                        elif "model" in cleaned_response:
                            cleaned_response = cleaned_response.split("model", 1)[1].strip()
                        message_placeholder.markdown(cleaned_response + "▌")
                    
                    # Final cleaning of the complete response
                    if "user" in full_response and "model" in full_response:
                        full_response = full_response.split("model", 1)[1].strip()
                    elif "model" in full_response:
                        full_response = full_response.split("model", 1)[1].strip()
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add cleaned assistant response to session state
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Maintain history limit after adding new messages
                    st.session_state.messages = maintain_chat_history(st.session_state.messages)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")