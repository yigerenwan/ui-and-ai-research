import streamlit as st
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
from PIL import Image
import torch
from threading import Thread

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
    model_id = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# Load model immediately after definition
model, processor = load_model()

# Set up the Streamlit page
st.title("Gemma-3 Image Chat")

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
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
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

                    # Add image to message content if uploaded
                    if uploaded_file:
                        messages[-1]["content"].append({"type": "image", "image": image})
                    
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