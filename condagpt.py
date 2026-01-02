import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import pygame
import threading
from queue import Queue
from pygame.locals import *

class LocalChatbot:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize chatbot with a capable local model.
        
        Recommended models (fastest to slowest):
        - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (fast, 2GB VRAM, ~1-2 sec per response)
        - mistralai/Mistral-7B-Instruct-v0.2 (best quality, 16GB VRAM, ~5-10 sec per response)
        - meta-llama/Llama-2-13b-chat-hf (best quality, 24GB VRAM)
        """
        print(f"Loading model: {model_name}")
        print("This may take a few minutes on first run...\n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load in 8-bit to reduce memory usage (requires bitsandbytes)
        # If you get an error, remove quantization_config
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        except:
            # Fallback without 8-bit quantization
            print("8-bit quantization not available, using fp16 instead...\n")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.conversation_history = []
    
    def chat(self, user_input: str, temperature: float = 0.7, max_tokens: int = 512, token_queue: Optional[Queue] = None) -> str:
        """Generate a response to user input, streaming tokens to queue if provided."""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Format as conversation
        messages = self.conversation_history
        
        # Use chat template if available (Mistral supports this)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # Fallback formatting
            prompt = self._format_messages(messages)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[-1]
        
        # Generate with streaming
        full_response = ""
        stop_phrases = ["USER:", "User:", "user:", "YOU:", "You:", "ASSISTANT:", "Assistant:"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=False
            )
        
        # Decode the full response first
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_response = full_text[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        
        # Stop at common delimiters to prevent generating user input
        for phrase in stop_phrases:
            if phrase in full_response:
                full_response = full_response[:full_response.index(phrase)].strip()
                break
        
        # Stream characters to queue if provided
        if token_queue:
            for char in full_response:
                token_queue.put(char)
                import time
                time.sleep(0.01)  # Small delay for typing effect
            token_queue.put(None)  # Signal end of response
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })
        
        return full_response
    
    def _format_messages(self, messages: list) -> str:
        """Fallback message formatting."""
        formatted = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted += f"{role}: {content}\n"
        formatted += "ASSISTANT:"
        return formatted
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        torch.cuda.empty_cache()

    def run_pygame_ui(self):
        """Run the chatbot with a Pygame graphical interface."""
        pygame.init()
        screen_width, screen_height = 900, 700
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Local Chatbot - Pygame UI")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 26)
        small_font = pygame.font.Font(None, 20)

        chat_messages = []  # Store messages as (role, content) tuples
        input_text = ""
        generating = False
        response_queue = Queue()
        scroll_offset = 0  # For scrolling
        current_response = ""  # For streaming response

        line_height = 32
        history_height = screen_height - 120
        max_visible_lines = history_height // line_height
        
        # Send button
        send_button_rect = pygame.Rect(screen_width - 100, screen_height - 80, 80, 50)

        BG_COLOR = (30, 30, 40)
        INPUT_BG = (50, 50, 60)
        USER_COLOR = (100, 180, 255)
        ASSISTANT_COLOR = (255, 200, 100)
        SYSTEM_COLOR = (180, 180, 180)
        THINKING_COLOR = (200, 200, 0)
        TEXT_COLOR = (255, 255, 255)

        def wrap_text(text, max_width, use_small=False):
            """Wrap text to fit within max_width."""
            current_font = small_font if use_small else font
            words = text.split(' ')
            lines = []
            current = []
            for word in words:
                test = ' '.join(current + [word]) if current else word
                if current_font.size(test)[0] <= max_width:
                    current.append(word)
                else:
                    if current:
                        lines.append(' '.join(current))
                        current = [word]
                    else:
                        # Word is too long, add it anyway and start new line
                        lines.append(word)
            if current:
                lines.append(' '.join(current))
            return lines

        def add_message(role, content):
            """Add a message to the chat display."""
            chat_messages.append((role, content))

        def render_chat_lines():
            """Convert chat messages to display lines with proper formatting."""
            display_lines = []
            for role, content in chat_messages:
                if role == "system":
                    wrapped = wrap_text(content, screen_width - 60, use_small=True)
                    for i, line in enumerate(wrapped):
                        if i == 0:
                            display_lines.append(("system", f"System: {line}"))
                        else:
                            display_lines.append(("system", line))
                else:
                    prefix = "You: " if role == "user" else "Assistant: "
                    wrapped = wrap_text(content, screen_width - 80, use_small=False)
                    for i, line in enumerate(wrapped):
                        if i == 0:
                            display_lines.append((role, prefix + line))
                        else:
                            display_lines.append((role, line))
            return display_lines

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif generating:
                        continue  # disable input while generating
                    elif event.key == K_RETURN:
                        msg = input_text.strip()
                        input_text = ""
                        if msg.lower() == "/quit":
                            running = False
                        elif msg.lower() == "/reset":
                            self.reset_conversation()
                            chat_messages.clear()
                            scroll_offset = 0
                            add_message("system", "Conversation history cleared.")
                        elif msg:
                            add_message("user", msg)
                            chat_messages.append(("assistant", ""))  # Add empty assistant message
                            scroll_offset = 0
                            generating = True
                            current_response = ""
                            user_msg = msg

                            def generation_thread():
                                try:
                                    self.chat(user_msg, token_queue=response_queue)
                                except Exception as e:
                                    response_queue.put(f"Error: {str(e)}")
                                    response_queue.put(None)

                            threading.Thread(target=generation_thread, daemon=True).start()
                    elif event.key == K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == K_UP:
                        # Scroll up
                        scroll_offset = min(scroll_offset + 3, len(render_chat_lines()) - max_visible_lines)
                    elif event.key == K_DOWN:
                        # Scroll down
                        scroll_offset = max(scroll_offset - 3, 0)
                    else:
                        input_text += event.unicode
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll wheel up
                        scroll_offset = min(scroll_offset + 3, len(render_chat_lines()) - max_visible_lines)
                    elif event.button == 5:  # Scroll wheel down
                        scroll_offset = max(scroll_offset - 3, 0)
                    elif send_button_rect.collidepoint(event.pos) and not generating:
                        # Send button clicked
                        msg = input_text.strip()
                        input_text = ""
                        if msg:
                            add_message("user", msg)
                            chat_messages.append(("assistant", ""))  # Add empty assistant message
                            scroll_offset = 0
                            generating = True
                            current_response = ""
                            user_msg = msg

                            def generation_thread():
                                try:
                                    self.chat(user_msg, token_queue=response_queue)
                                except Exception as e:
                                    response_queue.put(f"Error: {str(e)}")
                                    response_queue.put(None)

                            threading.Thread(target=generation_thread, daemon=True).start()

            # Check for streaming response
            if generating:
                while not response_queue.empty():
                    char = response_queue.get()
                    if char is None:  # End of response
                        generating = False
                        break
                    current_response += char
                
                # Update the last assistant message in chat_messages
                if chat_messages and chat_messages[-1][0] == "assistant":
                    chat_messages[-1] = ("assistant", current_response)

            # Rendering
            screen.fill(BG_COLOR)

            # Chat history
            display_lines = render_chat_lines()
            visible_lines = display_lines[-max_visible_lines - scroll_offset:-scroll_offset if scroll_offset > 0 else None] if display_lines else []
            
            y = 20
            for role, line in visible_lines:
                if role == "user":
                    color = USER_COLOR
                    current_font = font
                elif role == "assistant":
                    color = ASSISTANT_COLOR
                    current_font = font
                else:  # system
                    color = SYSTEM_COLOR
                    current_font = small_font
                
                surf = current_font.render(line, True, color)
                screen.blit(surf, (30, y))
                y += line_height

            # Thinking indicator
            if generating:
                thinking_surf = font.render("Assistant is thinking...", True, THINKING_COLOR)
                screen.blit(thinking_surf, (30, y))

            # Input box
            input_rect = pygame.Rect(20, screen_height - 80, screen_width - 120, 50)
            pygame.draw.rect(screen, INPUT_BG, input_rect, border_radius=12)
            cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else " "
            input_surf = font.render(input_text + cursor, True, TEXT_COLOR)
            screen.blit(input_surf, (input_rect.x + 15, input_rect.y + 12))

            # Hint when empty
            if not input_text:
                hint_surf = small_font.render("Type your message here and press Enter...", True, (100, 100, 100))
                screen.blit(hint_surf, (input_rect.x + 15, input_rect.y + 12))

            # Send button
            pygame.draw.rect(screen, (70, 120, 180) if not generating else (50, 80, 120), send_button_rect, border_radius=8)
            send_text = small_font.render("Send" if not generating else "...", True, TEXT_COLOR)
            text_rect = send_text.get_rect(center=send_button_rect.center)
            screen.blit(send_text, text_rect)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

# Usage
if __name__ == "__main__":
    bot = LocalChatbot()
    bot.run_pygame_ui()
