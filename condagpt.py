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
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        except:
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
        
        messages = self.conversation_history
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = self._format_messages(messages)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[-1]
        
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
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_response = full_text[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        
        for phrase in stop_phrases:
            if phrase in full_response:
                full_response = full_response[:full_response.index(phrase)].strip()
                break
        
        if token_queue:
            for char in full_response:
                token_queue.put(char)
                import time
                time.sleep(0.01)
            token_queue.put(None)
        
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
        font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 18)

        chat_messages = []  # Store messages as {"role": str, "content": str}
        input_text = ""
        generating = False
        response_queue = Queue()
        scroll_offset = 0
        current_response = ""

        line_height = 28
        history_height = screen_height - 120
        max_visible_lines = history_height // line_height
        
        send_button_rect = pygame.Rect(screen_width - 100, screen_height - 80, 80, 50)

        BG_COLOR = (30, 30, 40)
        INPUT_BG = (50, 50, 60)
        USER_COLOR = (100, 180, 255)
        ASSISTANT_COLOR = (200, 220, 100)
        SYSTEM_COLOR = (180, 180, 180)
        THINKING_COLOR = (150, 200, 255)
        TEXT_COLOR = (255, 255, 255)

        def wrap_text(text, max_width, current_font):
            """Wrap text to fit within max_width, respecting existing line breaks."""
            lines = []
            # First split by existing newlines
            for paragraph in text.split('\n'):
                if not paragraph.strip():
                    lines.append("")
                    continue
                
                words = paragraph.split(' ')
                current_line = []
                for word in words:
                    test_line = ' '.join(current_line + [word]) if current_line else word
                    if current_font.size(test_line)[0] <= max_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)
                if current_line:
                    lines.append(' '.join(current_line))
            return lines

        def add_message(role, content):
            """Add a message to the chat display."""
            chat_messages.append({"role": role, "content": content})

        def render_chat_lines():
            """Convert chat messages to display lines with proper formatting."""
            display_lines = []
            for msg in chat_messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    wrapped = wrap_text(content, screen_width - 60, small_font)
                    for i, line in enumerate(wrapped):
                        if i == 0:
                            display_lines.append({
                                "text": f"[System] {line}",
                                "role": "system",
                                "font": small_font
                            })
                        else:
                            display_lines.append({
                                "text": line,
                                "role": "system",
                                "font": small_font
                            })
                else:
                    prefix = "You: " if role == "user" else "Assistant: "
                    wrapped = wrap_text(content, screen_width - 100, font)
                    for i, line in enumerate(wrapped):
                        if i == 0:
                            display_lines.append({
                                "text": prefix + line,
                                "role": role,
                                "font": font
                            })
                        else:
                            display_lines.append({
                                "text": line,
                                "role": role,
                                "font": font
                            })
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
                        continue
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
                            scroll_offset = 0
                            generating = True
                            current_response = ""
                            user_msg = msg

                            def generation_thread():
                                try:
                                    self.chat(user_msg, token_queue=response_queue)
                                except Exception as e:
                                    response_queue.put(str(e))
                                    response_queue.put(None)

                            threading.Thread(target=generation_thread, daemon=True).start()
                    elif event.key == K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == K_UP:
                        display_lines = render_chat_lines()
                        scroll_offset = min(scroll_offset + 3, max(0, len(display_lines) - max_visible_lines))
                    elif event.key == K_DOWN:
                        scroll_offset = max(scroll_offset - 3, 0)
                    else:
                        input_text += event.unicode
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll wheel up
                        display_lines = render_chat_lines()
                        scroll_offset = min(scroll_offset + 3, max(0, len(display_lines) - max_visible_lines))
                    elif event.button == 5:  # Scroll wheel down
                        scroll_offset = max(scroll_offset - 3, 0)
                    elif send_button_rect.collidepoint(event.pos) and not generating:
                        msg = input_text.strip()
                        input_text = ""
                        if msg:
                            add_message("user", msg)
                            scroll_offset = 0
                            generating = True
                            current_response = ""
                            user_msg = msg

                            def generation_thread():
                                try:
                                    self.chat(user_msg, token_queue=response_queue)
                                except Exception as e:
                                    response_queue.put(str(e))
                                    response_queue.put(None)

                            threading.Thread(target=generation_thread, daemon=True).start()

            # Process streaming response
            if generating:
                while not response_queue.empty():
                    char = response_queue.get()
                    if char is None:
                        generating = False
                        break
                    current_response += char
                
                # Update or add assistant message
                if not chat_messages or chat_messages[-1]["role"] != "assistant":
                    add_message("assistant", current_response)
                else:
                    chat_messages[-1]["content"] = current_response

            # Rendering
            screen.fill(BG_COLOR)

            # Chat history
            display_lines = render_chat_lines()
            start_idx = max(0, len(display_lines) - max_visible_lines - scroll_offset)
            end_idx = max(0, len(display_lines) - scroll_offset)
            visible_lines = display_lines[start_idx:end_idx] if display_lines else []
            
            y = 20
            for line_data in visible_lines:
                color = (USER_COLOR if line_data["role"] == "user" 
                        else ASSISTANT_COLOR if line_data["role"] == "assistant"
                        else SYSTEM_COLOR)
                
                surf = line_data["font"].render(line_data["text"], True, color)
                screen.blit(surf, (30, y))
                y += line_height

            # Thinking indicator
            if generating:
                thinking_surf = font.render("⟳ Thinking...", True, THINKING_COLOR)
                screen.blit(thinking_surf, (30, y))

            # Input box
            input_rect = pygame.Rect(20, screen_height - 80, screen_width - 120, 50)
            pygame.draw.rect(screen, INPUT_BG, input_rect, border_radius=12)
            cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else " "
            input_surf = font.render(input_text + cursor, True, TEXT_COLOR)
            screen.blit(input_surf, (input_rect.x + 15, input_rect.y + 12))

            if not input_text:
                hint_surf = small_font.render("Type message and press Enter...", True, (100, 100, 100))
                screen.blit(hint_surf, (input_rect.x + 15, input_rect.y + 12))

            # Send button
            button_color = (70, 120, 180) if not generating else (50, 80, 120)
            pygame.draw.rect(screen, button_color, send_button_rect, border_radius=8)
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
