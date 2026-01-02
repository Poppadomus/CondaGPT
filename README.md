# Local AI Chatbot with Pygame UI

A fully local, open-source chatbot that runs large language models (LLMs) on your own hardware using Hugging Face Transformers. No internet connection or API key required after the initial model download. Features a simple graphical interface built with Pygame, streaming responses with a typing effect, conversation history, and support for multiple models.

<img width="850" height="545" alt="image" src="https://github.com/user-attachments/assets/a81d3f59-95d1-4d00-8c33-ff9061327115" />


## Features

- Runs entirely offline after model download
- Streaming responses with realistic typing animation
- Color-coded chat window (user vs assistant)
- Scrolling chat history with mouse wheel / arrow keys support
- "Thinking..." indicator during generation
- Send button and input box
- Commands: `/reset` to clear history, `/quit` to exit
- Supports chat templates for compatible models
- Automatic stopping at common conversation delimiters
- 8-bit quantization fallback for lower VRAM usage

## Recommended Models

| Model                                      | Quality   | VRAM Needed | Speed (approx.)      |
|--------------------------------------------|-----------|-------------|----------------------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0`       | Good      | ~2GB        | 1-2 sec/response     |
| `mistralai/Mistral-7B-Instruct-v0.2`      | Excellent | ~16GB       | 5-10 sec/response    |
| `meta-llama/Llama-2-13b-chat-hf`           | Excellent | ~24GB       | Slower               |

Change the model in code: `bot = LocalChatbot(model_name="your/model-here")`

## Requirements

- Python 3.8+
- GPU with CUDA recommended (CPU works but very slow for larger models)
- Libraries:
  ```bash
  pip install torch transformers pygame bitsandbytes accelerate
