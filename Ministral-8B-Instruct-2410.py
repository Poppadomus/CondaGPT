import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import pygame
import threading
from queue import Queue
from pygame.locals import *
import pyperclip
import time

class LocalChatbot:
    def __init__(self, model_name: str = "mistralai/Ministral-8B-Instruct-2410"):
        print(f"Loading model: {model_name}")
        print("This may take a few minutes on first run (8B model)...\n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        except Exception as e:
            print("8-bit loading failed, falling back to fp16...\n")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_name = "Ministral-8B"
        
        self.mode_list = [
            "Ada", "Grace", "Hedy", "Rosalind", "Marie", "Irène", "Lise", "Katherine", "Vera", "Barbara",
            "Jocelyn", "Caroline", "Dorothy", "Cecilia", "Emmy", "Sofia", "Hypatia", "Mary", "Rachel", "Jane",
            "Jennifer", "Stephanie", "Tim", "Sergey", "Larry", "Mark", "Jack", "Jeff", "Sundar", "Satya",
            "Linus", "Richard", "John", "Dennis", "Ken", "Bjarne", "Guido", "Yukihiro", "Brendan", "James",
            "Nikola", "Thomas", "Michael", "Max", "Erwin", "Werner", "Niels", "Enrico", "Stephen", "Roger",
            "Carl", "Charles", "René", "Blaise", "Evangelista", "Robert", "Christiaan", "Johannes", "Tycho", "Galileo",
            "Archimedes", "Euclid", "Pythagoras", "Hipparchus", "Eratosthenes", "Heron", "Ptolemy", "Aristotle", "Plato", "Socrates",
            "Democritus", "Anaximander", "Thales", "Pythagoras", "Heraclitus", "Parmenides", "Empedocles", "Anaxagoras", "Zeno", "Epicurus",
            "Lucretius", "Cicero", "Seneca", "Marcus", "Augustine", "Thomas", "William", "Roger", "Francis", "Leonardo",
            "Nicolaus", "Georg", "Leonhard", "Daniel", "Joseph", "Antoine", "Alessandro", "Humphry", "John", "James"
        ]

        self.modes = {}
        num_modes = len(self.mode_list)
        FIXED_MAX_TOKENS = 32

        for i, mode_name in enumerate(self.mode_list):
            progress = i / (num_modes - 1)
            temp = 1.8 - progress * 1.6
            rep = 1.0 + progress * 0.5
            top_p = 1.0 - progress * 0.6
            delay = 0.001 + progress * 0.039

            self.modes[mode_name] = {
                "temp": round(temp, 2),
                "max": FIXED_MAX_TOKENS,
                "rep": round(rep, 2),
                "top_p": round(top_p, 2),
                "delay": round(delay, 4)
            }
        
        self.current_mode = "Leonhard"
        self.stop_generation = False
        self.stop_phrases = ["<|im_end|>", "</s>", "USER:", "User:", "ASSISTANT:"]

    def get_mode_params(self, mode_name: str):
        return self.modes.get(mode_name, self.modes[self.current_mode])

    def chat(self, user_input: str, token_queue: Optional[Queue] = None, mode_override: Optional[str] = None) -> str:
        messages = [{"role": "user", "content": user_input}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        params = self.get_mode_params(mode_override) if mode_override else self.get_mode_params(self.current_mode)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params["max"],
                temperature=params["temp"],
                top_p=params["top_p"],
                top_k=40,
                do_sample=True,
                repetition_penalty=params["rep"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        if token_queue:
            delay = params["delay"]
            for char in response:
                if self.stop_generation:
                    token_queue.put(None)
                    return response
                token_queue.put(char)
                time.sleep(delay)
            token_queue.put(None)
        
        return response

    def run_dual_chat(self, name1: str, name2: str, response_queue: Queue, topic: str = ""):
        if name1 not in self.modes or name2 not in self.modes:
            response_queue.put(None)
            return

        response_queue.put(("header", f"{name1} ↔ {name2} Conversation"))
        if topic:
            response_queue.put(("header", topic))

        context = topic or "You are having an interesting conversation."
        current_speaker = name1

        for _ in range(20):
            if self.stop_generation:
                response_queue.put(("stop", ""))
                break

            response_queue.put(("speaker", current_speaker))

            temp_queue = Queue()
            threading.Thread(
                target=lambda s=current_speaker, c=context: self.chat(
                    f"You are {s}. Continue the conversation:\n\n{c}",
                    token_queue=temp_queue,
                    mode_override=s
                ),
                daemon=True
            ).start()

            full_response = ""
            while True:
                char = temp_queue.get()
                if char is None:
                    break
                full_response += char
                response_queue.put(("char", char))

            context += f"\n{current_speaker}: {full_response.strip()}"
            response_queue.put(("end", full_response.strip()))

            current_speaker = name2 if current_speaker == name1 else name1

        response_queue.put(None)

    def run_benchmark(self, prompt: str, response_queue: Queue, start_mode: Optional[str] = None):
        response_queue.put("Running benchmark across all modes...\n\n")

        modes_to_run = self.mode_list
        if start_mode and start_mode in self.modes:
            start_idx = self.mode_list.index(start_mode)
            modes_to_run = self.mode_list[start_idx:]

        for mode_name in modes_to_run:
            if self.stop_generation:
                response_queue.put("\n[Benchmark stopped]\n")
                break
            response_queue.put(f"### {mode_name} ###\n")
            temp_q = Queue()
            threading.Thread(target=lambda m=mode_name: self.chat(prompt, token_queue=temp_q, mode_override=m), daemon=True).start()
            while True:
                c = temp_q.get()
                if c is None: break
                response_queue.put(c)
            response_queue.put("\n" + "-"*60 + "\n\n")
            time.sleep(0.3)
        response_queue.put(None)

    def run_pygame_ui(self):
        pygame.init()
        screen_width, screen_height = 1200, 800
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"CondaGPT • {self.model_name}")
        clock = pygame.time.Clock()

        font = pygame.font.SysFont("calibri,arial,segoeui", 20)
        small_font = pygame.font.SysFont("calibri,arial,segoeui", 16)

        chat_messages = []
        input_text = ""
        generating = False
        response_queue = Queue()
        scroll_offset = 0
        auto_scroll = True

        mode_button_width = 140
        mode_button_height = 40
        mode_area_left = 200
        mode_area_width = screen_width - 600
        mode_scroll_offset = 0
        total_modes_width = len(self.mode_list) * (mode_button_width + 10) - 10

        new_chat_button_rect = pygame.Rect(screen_width - 300, 12, 130, 36)
        send_button_rect = pygame.Rect(screen_width - 140, screen_height - 75, 110, 50)

        CHAT_TOP = 80
        CHAT_BOTTOM = screen_height - 100
        CHAT_HEIGHT = CHAT_BOTTOM - CHAT_TOP
        CHAT_LEFT = 30
        CHAT_RIGHT = screen_width - 50
        CHAT_WIDTH = CHAT_RIGHT - CHAT_LEFT

        BG_COLOR = (0, 0, 0)
        TOPBAR_BG = (20, 20, 20)
        CHAT_BG = (15, 15, 15)
        USER_BG = (40, 40, 40)
        ASSISTANT_BG = (30, 30, 30)
        DUAL_BG_1 = (42, 42, 42)
        DUAL_BG_2 = (26, 26, 26)
        INPUT_BG = (25, 25, 25)
        BUTTON_BG = (50, 50, 50)
        BUTTON_HOVER = (80, 80, 80)
        ACTIVE_BUTTON = (100, 100, 100)
        BORDER_COLOR = (60, 60, 60)
        TEXT_COLOR = (255, 255, 255)
        SECONDARY_TEXT = (180, 180, 180)

        message_heights = {}

        def wrap_text(text, max_width):
            words = text.split(' ')
            lines = []
            current_line = []
            avg_char_width = 12
            for word in words:
                test_line = ' '.join(current_line + [word])
                if len(test_line) * avg_char_width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            return lines if lines else ['']

        def get_message_height(content: str, has_speaker: bool = False):
            key = (content, has_speaker)
            if key in message_heights:
                return message_heights[key]

            max_width = CHAT_WIDTH - 180
            raw_lines = content.splitlines()
            lines = []
            for raw_line in raw_lines:
                if raw_line.strip() == "":
                    lines.append("")
                else:
                    lines.extend(wrap_text(raw_line, max_width))

            line_height = 26
            text_height = len(lines) * line_height
            bubble_height = text_height + (80 if has_speaker else 60)
            total_height = bubble_height + 20

            message_heights[key] = total_height
            return total_height

        def draw_message_bubble(y, role, content, speaker="", is_streaming=False):
            if role == "user":
                bg_color = USER_BG
            elif speaker:
                bg_color = DUAL_BG_1 if hash(speaker) % 2 == 0 else DUAL_BG_2
            else:
                bg_color = ASSISTANT_BG

            border_color = (100, 100, 100) if is_streaming else BORDER_COLOR
            border_width = 2 if is_streaming else 1

            max_width = CHAT_WIDTH - 180
            raw_lines = content.splitlines()
            lines = []
            for raw_line in raw_lines:
                if raw_line.strip() == "":
                    lines.append("")
                else:
                    lines.extend(wrap_text(raw_line, max_width))

            line_height = 26
            text_height = len(lines) * line_height
            bubble_height = text_height + (80 if speaker else 60)
            bubble_rect = pygame.Rect(CHAT_LEFT + 20, y, CHAT_WIDTH - 80, bubble_height)

            if is_streaming:
                glow = bubble_rect.inflate(16, 16)
                pygame.draw.rect(screen, (80, 80, 80, 40), glow, border_radius=18)

            pygame.draw.rect(screen, bg_color, bubble_rect, border_radius=18)
            pygame.draw.rect(screen, border_color, bubble_rect, border_width, border_radius=18)

            if speaker:
                name_surf = small_font.render(speaker, True, (180, 200, 255))
                screen.blit(name_surf, (bubble_rect.x + 20, y + 18))

            current_y = y + (70 if speaker else 50)
            for line in lines:
                if line:
                    text_surf = font.render(line, True, TEXT_COLOR)
                    screen.blit(text_surf, (CHAT_LEFT + 40, current_y))
                current_y += line_height

            button_y = y + 12
            buttons = {}
            if role == "assistant" and not generating and not speaker:
                regen_rect = pygame.Rect(bubble_rect.right - 150, button_y, 70, 30)
                copy_rect = pygame.Rect(bubble_rect.right - 75, button_y, 60, 30)
                for rect, label in [(regen_rect, "Regen"), (copy_rect, "Copy")]:
                    hovered = rect.collidepoint(pygame.mouse.get_pos())
                    pygame.draw.rect(screen, BUTTON_HOVER if hovered else BUTTON_BG, rect, border_radius=8)
                    label_surf = small_font.render(label, True, TEXT_COLOR)
                    screen.blit(label_surf, label_surf.get_rect(center=rect.center))
                    buttons[label.lower()] = rect

            total_height = bubble_height + 20
            return total_height, buttons

        running = True
        message_completed = False

        while running:
            mouse_pos = pygame.mouse.get_pos()
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    self.stop_generation = True

                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    for i, mode in enumerate(self.mode_list):
                        btn_x = mode_area_left + 10 + i * (mode_button_width + 10) - mode_scroll_offset
                        btn_rect = pygame.Rect(btn_x, 12, mode_button_width, mode_button_height)
                        if btn_rect.collidepoint(event.pos):
                            self.current_mode = mode
                            break

                    if new_chat_button_rect.collidepoint(event.pos):
                        chat_messages.clear()
                        message_heights.clear()
                        generating = False
                        auto_scroll = True
                        scroll_offset = 0
                        message_completed = False
                        while not response_queue.empty():
                            try: response_queue.get_nowait()
                            except: pass

                    y_pos = CHAT_TOP + 20 - scroll_offset
                    for idx, msg in enumerate(chat_messages):
                        role = msg["role"]
                        content = msg["content"]
                        speaker = msg.get("speaker", "")
                        total_height, buttons = draw_message_bubble(y_pos, role, content, speaker)
                        if buttons.get("copy") and buttons["copy"].collidepoint(event.pos):
                            pyperclip.copy(content)
                        y_pos += total_height

                    if send_button_rect.collidepoint(event.pos):
                        if generating:
                            self.stop_generation = True
                            generating = False
                        elif input_text.strip():
                            msg = input_text.strip()
                            input_text = ""
                            if msg.lower() == "/quit":
                                running = False
                            elif msg.lower().startswith("/chat "):
                                parts = msg[len("/chat "):].strip().split(maxsplit=2)
                                if len(parts) < 2:
                                    chat_messages.append({"role": "assistant", "content": "Usage: /chat Name1 Name2 [topic]"})
                                else:
                                    n1, n2 = parts[0], parts[1]
                                    topic = parts[2] if len(parts) > 2 else ""
                                    if n1 in self.modes and n2 in self.modes:
                                        chat_messages.append({"role": "user", "content": msg})
                                        generating = True
                                        self.stop_generation = False
                                        threading.Thread(target=self.run_dual_chat, args=(n1, n2, response_queue, topic), daemon=True).start()
                                    else:
                                        chat_messages.append({"role": "assistant", "content": "Invalid name(s)."})
                            elif msg.lower().startswith("/benchmark"):
                                rest = msg[len("/benchmark"):].strip()
                                parts = rest.split(maxsplit=1)
                                start_mode = parts[0] if parts else None
                                benchmark_prompt = parts[1] if len(parts) > 1 else ""
                                if not benchmark_prompt:
                                    chat_messages.append({"role": "assistant", "content": "Usage: /benchmark [Mode] <prompt>"})
                                else:
                                    chat_messages.append({"role": "user", "content": f"Benchmark: {benchmark_prompt}"})
                                    chat_messages.append({"role": "assistant", "content": ""})
                                    generating = True
                                    self.stop_generation = False
                                    threading.Thread(target=self.run_benchmark, args=(benchmark_prompt, response_queue, start_mode), daemon=True).start()
                            else:
                                chat_messages.append({"role": "user", "content": msg})
                                chat_messages.append({"role": "assistant", "content": ""})
                                generating = True
                                self.stop_generation = False
                                threading.Thread(target=self.chat, args=(msg, response_queue), daemon=True).start()

                elif event.type == KEYDOWN:
                    if event.key == K_RETURN and not generating:
                        if input_text.strip():
                            msg = input_text.strip()
                            input_text = ""
                            if msg.lower() == "/quit":
                                running = False
                            elif msg.lower().startswith("/chat "):
                                parts = msg[len("/chat "):].strip().split(maxsplit=2)
                                if len(parts) < 2:
                                    chat_messages.append({"role": "assistant", "content": "Usage: /chat Name1 Name2 [topic]"})
                                else:
                                    n1, n2 = parts[0], parts[1]
                                    topic = parts[2] if len(parts) > 2 else ""
                                    if n1 in self.modes and n2 in self.modes:
                                        chat_messages.append({"role": "user", "content": msg})
                                        generating = True
                                        self.stop_generation = False
                                        threading.Thread(target=self.run_dual_chat, args=(n1, n2, response_queue, topic), daemon=True).start()
                            elif msg.lower().startswith("/benchmark"):
                                pass
                            else:
                                chat_messages.append({"role": "user", "content": msg})
                                chat_messages.append({"role": "assistant", "content": ""})
                                generating = True
                                self.stop_generation = False
                                threading.Thread(target=self.chat, args=(msg, response_queue), daemon=True).start()
                    elif event.key == K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == K_v and pygame.key.get_mods() & KMOD_CTRL:
                        try: input_text += pyperclip.paste()
                        except: pass
                    else:
                        input_text += event.unicode

                elif event.type == MOUSEWHEEL:
                    if mode_area_left < mouse_pos[0] < mode_area_left + mode_area_width and mouse_pos[1] < 70:
                        mode_scroll_offset = max(0, min(mode_scroll_offset - event.y * 80, total_modes_width - mode_area_width))
                    else:
                        scroll_offset = max(0, scroll_offset - event.y * 60)
                        auto_scroll = False

            message_completed = False
            if generating:
                while not response_queue.empty():
                    try:
                        item = response_queue.get_nowait()
                        if item is None:
                            generating = False
                            message_completed = True
                            break
                        if isinstance(item, tuple):
                            if item[0] == "header":
                                chat_messages.append({"role": "assistant", "content": item[1], "speaker": ""})
                            elif item[0] == "speaker":
                                chat_messages.append({"role": "assistant", "content": "", "speaker": item[1]})
                            elif item[0] == "char":
                                if chat_messages and "speaker" in chat_messages[-1]:
                                    chat_messages[-1]["content"] += item[1]
                            elif item[0] == "end":
                                if chat_messages and "speaker" in chat_messages[-1]:
                                    chat_messages[-1]["content"] = item[1]
                                    message_completed = True
                        else:
                            if not chat_messages or ("speaker" not in chat_messages[-1] and chat_messages[-1]["role"] != "assistant"):
                                chat_messages.append({"role": "assistant", "content": ""})
                            chat_messages[-1]["content"] += item
                    except:
                        break

            total_content_height = 0
            for msg in chat_messages:
                has_speaker = bool(msg.get("speaker"))
                total_content_height += get_message_height(msg["content"], has_speaker)

            if message_completed:
                scroll_offset = max(0, total_content_height - CHAT_HEIGHT + 50)
                auto_scroll = True

            scroll_offset = max(0, min(scroll_offset, max(0, total_content_height - CHAT_HEIGHT)))

            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, TOPBAR_BG, (0, 0, screen_width, 60))
            title = font.render(f"CondaGPT • {self.model_name} • 100 Modes", True, TEXT_COLOR)
            screen.blit(title, (30, 16))

            mode_area = pygame.Rect(mode_area_left, 10, mode_area_width, 50)
            pygame.draw.rect(screen, (10, 10, 10), mode_area, border_radius=12)
            pygame.draw.rect(screen, BORDER_COLOR, mode_area, 2, border_radius=12)
            screen.set_clip(mode_area)
            for i, mode in enumerate(self.mode_list):
                btn_x = mode_area_left + 10 + i * (mode_button_width + 10) - mode_scroll_offset
                btn_rect = pygame.Rect(btn_x, 12, mode_button_width, mode_button_height)
                is_active = mode == self.current_mode
                is_hovered = btn_rect.collidepoint(mouse_pos)
                color = ACTIVE_BUTTON if is_active else (BUTTON_HOVER if is_hovered else BUTTON_BG)
                pygame.draw.rect(screen, color, btn_rect, border_radius=12)
                pygame.draw.rect(screen, BORDER_COLOR, btn_rect, 2, border_radius=12)
                text_surf = small_font.render(mode, True, TEXT_COLOR)
                text_rect = text_surf.get_rect(center=btn_rect.center)
                screen.blit(text_surf, text_rect)
            screen.set_clip(None)

            if total_modes_width > mode_area_width:
                sw = mode_area_width * (mode_area_width / total_modes_width)
                sx = mode_area_left + (mode_scroll_offset / (total_modes_width - mode_area_width)) * (mode_area_width - sw)
                pygame.draw.rect(screen, (70, 70, 70), (sx, 55, sw, 4), border_radius=2)

            hovered = new_chat_button_rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen, BUTTON_HOVER if hovered else BUTTON_BG, new_chat_button_rect, border_radius=12)
            pygame.draw.rect(screen, BORDER_COLOR, new_chat_button_rect, 2, border_radius=12)
            text_surf = small_font.render("New Chat", True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=new_chat_button_rect.center)
            screen.blit(text_surf, text_rect)

            pygame.draw.rect(screen, CHAT_BG, (CHAT_LEFT, CHAT_TOP, CHAT_WIDTH, CHAT_HEIGHT), border_radius=20)
            screen.set_clip(pygame.Rect(CHAT_LEFT, CHAT_TOP, CHAT_WIDTH, CHAT_HEIGHT))
            y = CHAT_TOP + 20 - scroll_offset
            for idx, msg in enumerate(chat_messages):
                speaker = msg.get("speaker", "")
                is_streaming = generating and idx == len(chat_messages) - 1
                height, _ = draw_message_bubble(y, msg["role"], msg["content"], speaker, is_streaming)
                y += height
            screen.set_clip(None)

            if generating:
                dots = "." * ((current_time // 400) % 4)
                screen.blit(small_font.render(f"Thinking{dots}", True, TEXT_COLOR), (CHAT_LEFT + 40, CHAT_BOTTOM - 40))

            input_rect = pygame.Rect(30, screen_height - 85, screen_width - 190, 60)
            pygame.draw.rect(screen, INPUT_BG, input_rect, border_radius=20)
            pygame.draw.rect(screen, BORDER_COLOR, input_rect, 2, border_radius=20)
            cursor = "|" if (current_time // 500) % 2 == 0 and not generating else ""
            screen.blit(font.render(input_text + cursor, True, TEXT_COLOR), (55, screen_height - 67))
            if not input_text:
                screen.blit(small_font.render("Ask me anything... (/chat Name1 Name2 • /benchmark)", True, SECONDARY_TEXT), (55, screen_height - 65))

            hov = send_button_rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen, BUTTON_HOVER if hov else BUTTON_BG, send_button_rect, border_radius=16)
            pygame.draw.rect(screen, BORDER_COLOR, send_button_rect, 2, border_radius=16)
            label = "Stop" if generating else "Send"
            label_surf = font.render(label, True, TEXT_COLOR)
            label_rect = label_surf.get_rect(center=send_button_rect.center)
            screen.blit(label_surf, label_rect)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    bot = LocalChatbot()
    bot.run_pygame_ui()
