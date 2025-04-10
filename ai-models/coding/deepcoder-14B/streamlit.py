import streamlit as st
from llama_cpp import Llama

# Page configuration
st.set_page_config(page_title="DeepCoder Chat", layout="wide")

# Custom CSS for a modern, clean look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 5px solid #8bc34a;
    }
    .code-block {
        background-color: #263238;
        color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.3rem;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    h1 {
        color: #333;
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("💻 DeepCoder Chat")
st.markdown("Chat with DeepCoder-14B, an AI assistant that can help with coding tasks.")

# Define model options
model_options = {
    "agentica-org_DeepCoder-14B-Preview-bf16.gguf": {"size": "29.55GB", "description": "Full BF16 weights."},
    "agentica-org_DeepCoder-14B-Preview-Q8_0.gguf": {"size": "15.70GB", "description": "Extremely high quality, generally unneeded but max available quant."},
    "agentica-org_DeepCoder-14B-Preview-Q6_K_L.gguf": {"size": "12.50GB", "description": "Uses Q8_0 for embed and output weights. Very high quality, near perfect, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q6_K.gguf": {"size": "12.12GB", "description": "Very high quality, near perfect, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q5_K_L.gguf": {"size": "10.99GB", "description": "Uses Q8_0 for embed and output weights. High quality, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q5_K_M.gguf": {"size": "10.51GB", "description": "High quality, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q5_K_S.gguf": {"size": "10.27GB", "description": "High quality, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q4_K_L.gguf": {"size": "9.57GB", "description": "Uses Q8_0 for embed and output weights. Good quality, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q4_1.gguf": {"size": "9.39GB", "description": "Legacy format, similar performance to Q4_K_S but with improved tokens/watt on Apple silicon."},
    "agentica-org_DeepCoder-14B-Preview-Q4_K_M.gguf": {"size": "8.99GB", "description": "Good quality, default size for most use cases, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q3_K_XL.gguf": {"size": "8.61GB", "description": "Uses Q8_0 for embed and output weights. Lower quality but usable, good for low RAM availability."},
    "agentica-org_DeepCoder-14B-Preview-Q4_K_S.gguf": {"size": "8.57GB", "description": "Slightly lower quality with more space savings, recommended."},
    "agentica-org_DeepCoder-14B-Preview-IQ4_NL.gguf": {"size": "8.55GB", "description": "Similar to IQ4_XS, but slightly larger. Offers online repacking for ARM CPU inference."},
    "agentica-org_DeepCoder-14B-Preview-Q4_0.gguf": {"size": "8.54GB", "description": "Legacy format, offers online repacking for ARM and AVX CPU inference."},
    "agentica-org_DeepCoder-14B-Preview-IQ4_XS.gguf": {"size": "8.12GB", "description": "Decent quality, smaller than Q4_K_S with similar performance, recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q3_K_L.gguf": {"size": "7.92GB", "description": "Lower quality but usable, good for low RAM availability."},
    "agentica-org_DeepCoder-14B-Preview-Q3_K_M.gguf": {"size": "7.34GB", "description": "Low quality."},
    "agentica-org_DeepCoder-14B-Preview-IQ3_M.gguf": {"size": "6.92GB", "description": "Medium-low quality, new method with decent performance comparable to Q3_K_M."},
    "agentica-org_DeepCoder-14B-Preview-Q3_K_S.gguf": {"size": "6.66GB", "description": "Low quality, not recommended."},
    "agentica-org_DeepCoder-14B-Preview-Q2_K_L.gguf": {"size": "6.53GB", "description": "Uses Q8_0 for embed and output weights. Very low quality but surprisingly usable."},
    "agentica-org_DeepCoder-14B-Preview-IQ3_XS.gguf": {"size": "6.38GB", "description": "Lower quality, new method with decent performance, slightly better than Q3_K_S."},
    "agentica-org_DeepCoder-14B-Preview-IQ3_XXS.gguf": {"size": "5.95GB", "description": "Lower quality, new method with decent performance, comparable to Q3 quants."},
    "agentica-org_DeepCoder-14B-Preview-Q2_K.gguf": {"size": "5.77GB", "description": "Very low quality but surprisingly usable."},
    "agentica-org_DeepCoder-14B-Preview-IQ2_M.gguf": {"size": "5.36GB", "description": "Relatively low quality, uses SOTA techniques to be surprisingly usable."},
    "agentica-org_DeepCoder-14B-Preview-IQ2_S.gguf": {"size": "5.00GB", "description": "Low quality, uses SOTA techniques to be usable."},
    "agentica-org_DeepCoder-14B-Preview-IQ2_XS.gguf": {"size": "4.70GB", "description": "Low quality, uses SOTA techniques to be usable."},
}

# Add model selection to sidebar
with st.sidebar:
    st.header("Model Selection")
    
    # Create a list of options with short names, size and description
    model_display_options = []
    model_name_to_filename = {}
    
    for filename, info in model_options.items():
        # Extract the short name (e.g., "Q6_K" from "agentica-org_DeepCoder-14B-Preview-Q6_K.gguf")
        short_name = filename.split("agentica-org_DeepCoder-14B-Preview-")[1].split(".gguf")[0]
        display_option = f"{short_name} ({info['size']}) - {info['description']}"
        model_display_options.append(display_option)
        model_name_to_filename[display_option] = filename
    
    # Default to Q6_K as it's a good balance of quality and size
    default_model = "Q6_K ("
    default_index = next((i for i, option in enumerate(model_display_options) if option.startswith(default_model)), 0)
    
    selected_model_display = st.selectbox(
        "Select model version:",
        model_display_options,
        index=default_index,
        help="Choose a model version based on your available RAM and performance needs."
    )
    
    # Get the full filename from the selection
    selected_model_filename = model_name_to_filename[selected_model_display]
    
    st.info(f"Selected: {selected_model_filename}\nSize: {model_options[selected_model_filename]['size']}")

# Initialize the model
@st.cache_resource(show_spinner=False)
def load_model(model_filename):
    try:
        model = Llama.from_pretrained(
            repo_id="bartowski/agentica-org_DeepCoder-14B-Preview-GGUF",
            filename=model_filename,
            n_ctx=64000,  # Context window size
            n_gpu_layers=-1,  # Use all available GPU layers
            verbose=True  # Add verbose output for debugging
        )
        st.sidebar.success("✅ Model loaded successfully with GPU acceleration")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.sidebar.warning("⚠️ Falling back to CPU mode")
        return Llama.from_pretrained(
            repo_id="bartowski/agentica-org_DeepCoder-14B-Preview-GGUF",
            filename=model_filename,
            n_ctx=64000,
            n_gpu_layers=0,  # CPU only as fallback
            verbose=True
        )

# Load the model with the selected filename
with st.spinner(f"Loading {selected_model_filename}... This may take a moment."):
    model = load_model(selected_model_filename)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div><strong>You:</strong></div>
            <div>{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Use Streamlit's native markdown rendering for code blocks
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div><strong>DeepCoder:</strong></div>
        </div>
        """, unsafe_allow_html=True)
        # Render the content with proper markdown support
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask DeepCoder something...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.container():
        st.markdown(f"""
        <div class="chat-message user-message">
            <div><strong>You:</strong></div>
            <div>{user_input}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get model response
    with st.spinner("DeepCoder is thinking..."):
        # Format messages for llama.cpp
        prompt = ""
        for message in st.session_state.messages:
            if message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            else:
                prompt += f"Assistant: {message['content']}\n"
        
        prompt += "Assistant: "
        
        # Create a placeholder for streaming output
        response_placeholder = st.empty()
        assistant_container = st.container()
        
        with assistant_container:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div><strong>DeepCoder:</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize streaming response
            streaming_response = ""
            response_area = st.empty()
            
            # Generate response with llama.cpp in streaming mode
            for chunk in model(
                prompt=prompt,
                max_tokens=100000,
                temperature=0.7,
                stop=["User:"],
                stream=True
            ):
                # Extract the token from the chunk
                token = chunk["choices"][0]["text"]
                
                # Add token to the streaming response
                streaming_response += token
                
                # Update the display with the current response
                response_area.markdown(streaming_response)
                
            # Final response is the complete streaming response
            assistant_response = streaming_response.strip()
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Add a sidebar with information
with st.sidebar:
    st.header("About DeepCoder-14B")
    st.markdown("""
    DeepCoder-14B is a specialized AI model designed to assist with coding tasks.
    
    **Features:**
    - Code generation
    - Debugging assistance
    - Explaining code concepts
    - Answering programming questions
    
    This interface allows you to chat with DeepCoder and get coding help in real-time.
    """)
    
    st.divider()
    st.caption("Powered by agentica-org/DeepCoder-14B-Preview")

# Add system info to sidebar
with st.sidebar:
    st.header("System Information")
    try:    
        # Display additional model info if available
        if hasattr(model, 'model_path'):
            st.code(f"Model path: {model.model_path}")
    except Exception as e:
        st.warning(f"Could not retrieve system information: {str(e)}")