# To use GPU, you need to install llama-cpp-python with CUDA support.
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

from llama_cpp import Llama
from typing import List, Dict

class QwenChatbot:
    def __init__(self, repo_id: str, model_filename: str, n_gpu_layers: int = -1):
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=model_filename, 
            n_gpu_layers=n_gpu_layers,
            n_ctx=20000  # Increased context window size to 131072
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10  # 5 pairs of messages
        
    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            
    def get_model_response(self) -> str:
        response = self.llm.create_chat_completion(
            messages=self.conversation_history,
            temperature=0.6,
            top_p=0.95,
            top_k=10,
            min_p=0.05,
            typical_p=1,
            stream=True,
            max_tokens=10240,  # Reduced max tokens to stay within context window
        )
        # Handle streaming response
        full_response = ""
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                content = chunk["choices"][0]["delta"]["content"]
                full_response += content
                print(content, end="", flush=True)
        print()  # New line after response
        return full_response
    
    def chat(self) -> None:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break
                
            self.add_message("user", user_input)
            
            assistant_message = self.get_model_response()
            
            self.add_message("assistant", assistant_message)
            print("\n" + "-"*50 + "\n")

def main():
    chatbot = QwenChatbot(
        repo_id="bartowski/Qwen_QwQ-32B-GGUF",
        model_filename="Qwen_QwQ-32B-Q6_K.gguf"
    )
    chatbot.chat()

if __name__ == "__main__":
    main()