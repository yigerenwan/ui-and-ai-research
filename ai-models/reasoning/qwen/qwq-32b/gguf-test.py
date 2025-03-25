from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Qwen_QwQ-32B-GGUF",
	filename="Qwen_QwQ-32B-IQ2_M.gguf",
	n_gpu_layers=-1,
	# n_ctx=20000,
)

resp = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)

print(resp)