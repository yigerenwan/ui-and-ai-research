# To use GPU, you need to install llama-cpp-python with CUDA support.
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
# Also install streamlit: pip install streamlit

import streamlit as st
from llama_cpp import Llama
from typing import List, Dict

class QwenChatbot:
    def __init__(self, repo_id: str, model_filename: str, n_gpu_layers: int = -1):
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=model_filename, 
            n_gpu_layers=n_gpu_layers,
            n_ctx=20000  # Increased context window size
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10  # 5 pairs of messages
        
    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            
    def get_model_response(self, temperature: float = 0.6, max_tokens: int = 4096) -> str:
        with st.spinner('Thinking...'):
            response = self.llm.create_chat_completion(
                messages=self.conversation_history,
                temperature=temperature,
                top_p=0.95,
                top_k=10,
                min_p=0.05,
                typical_p=1,
                stream=True,
                max_tokens=max_tokens,  # Now using the parameter value
            )
            
            # Handle streaming response
            full_response = ""
            message_placeholder = st.empty()
            for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            return full_response

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = QwenChatbot(
            repo_id="bartowski/Qwen_QwQ-32B-GGUF",
            model_filename="Qwen_QwQ-32B-Q6_K.gguf"
        )
    if "conversation_name" not in st.session_state:
        st.session_state.conversation_name = "New Chat"

def clear_conversation():
    st.session_state.messages = []
    st.session_state.chatbot.conversation_history = []
    st.session_state.conversation_name = "New Chat"
    st.rerun()

def main():
    st.set_page_config(
        page_title="Qwen QwQ-32B Chatbot", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Add custom CSS for fixed chat input at bottom
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #262730;
        color: white;
    }
    
    /* Fixed chat input container at bottom */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 20.625rem;  /* Adjust for sidebar width (250px) */
        right: 0;
        padding: 1rem;
        background-color: white;
        z-index: 100;
        border-top: 1px solid #ddd;
    }
    
    /* Add padding to bottom of chat messages to prevent overlap with fixed input */
    .main .block-container {
        padding-bottom: 100px;
    }

    /* Adjust for collapsed sidebar */
    @media (max-width: 992px) {
        div[data-testid="stChatInput"] {
            left: 3rem;  /* Adjust when sidebar is collapsed */
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Left sidebar - Dark theme
    with st.sidebar:
        st.title("Qwen QwQ-32B")
        
        if st.button("New Chat", use_container_width=True):
            clear_conversation()
            
        st.text_input("Conversation Name", key="conversation_name")
        
        st.divider()
        
        # Model settings
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=256, max_value=10240, value=4096, step=256)
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Qwen QwQ-32B Chatbot")
        
        # Display chat messages from history first
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Place chat input at the bottom
        prompt = st.chat_input("What would you like to talk about?")
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Add user message to chatbot history
            st.session_state.chatbot.add_message("user", prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                assistant_response = st.session_state.chatbot.get_model_response(
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.session_state.chatbot.add_message("assistant", assistant_response)
            
            # Rerun to update the chat history display
            st.rerun()
    
    # Right sidebar - Example prompts
    with col2:
        st.subheader("Example Prompts")
        
        example_prompts = {
            "Math Problem": "Solve the equation: 3x² + 6x - 2 = 0",
            "Coding Challenge": "Write a Python function to check if a string is a palindrome",
            "Reasoning Question": "If a train travels at 120 km/h, how long will it take to travel 450 km?",
            "Logic Puzzle": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
        }
        
        for title, prompt in example_prompts.items():
            st.code(prompt, language="text")

if __name__ == "__main__":
    main()