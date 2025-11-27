import speech_recognition as sr
import pyttsx3
from threading import Thread
import time
import os
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: Gemini API not available. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False
    genai = None


class VoiceController:
    def __init__(self, action_handler, gemini_api_key=None):
        self.action_handler = action_handler
        self.recognizer = sr.Recognizer()
        try:
            self.engine = pyttsx3.init()
        except Exception:
            self.engine = None
        self.listening = False
        self.gemini_model = None
        self.accuracy = 0.0
        self.detection_count = 0
        self.success_count = 0

        # Get API key from environment or parameter
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if GEMINI_AVAILABLE and api_key and genai is not None:
            try:
                genai.configure(api_key=api_key)  # type: ignore
                self.gemini_model = genai.GenerativeModel('gemini-pro')  # type: ignore
                print("Gemini API initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini API: {e}")
                self.gemini_model = None
        else:
            print("Warning: Gemini API key not provided. Voice commands will use basic recognition.")

        self.commands = ["up", "down", "left", "right", "home", "back", "start", "stop", "select", "open"]

    def speak(self, text):
        """Speak text using TTS engine."""
        if self.engine is None:
            print(f"TTS: {text}")
            return
        
        def _speak():
            try:
                self.engine.say(text)  # type: ignore
                self.engine.runAndWait()  # type: ignore
            except Exception as e:
                print(f"TTS Error: {e}")

        Thread(target=_speak, daemon=True).start()

    def get_accuracy(self):
        """Return accuracy percentage for this mode."""
        if self.detection_count > 0:
            self.accuracy = (self.success_count / self.detection_count) * 100
        return self.accuracy

    def start_listening(self):
        if self.listening:
            print("Voice recognition already running")
            return
        self.listening = True
        print("Starting voice recognition...")
        Thread(target=self._loop, daemon=True).start()
        self.speak("Voice recognition started")
        print("Voice recognition thread started")

    def stop_listening(self):
        if self.listening:
            print("Stopping voice recognition...")
            self.listening = False
            # Wait a moment for the loop to exit
            import time
            time.sleep(0.1)
            print("Voice recognition stopped")

    def _process_with_gemini(self, text):
        """Process voice command using Gemini API for better understanding."""
        if not self.gemini_model:
            return None
        
        try:
            prompt = f"""Interpret this voice command and return only the action in JSON format:
Command: "{text}"
Available actions: up, down, left, right, home, back, select, open
Return format: {{"action": "action_name", "confidence": 0.0-1.0}}

Examples:
- "go up" -> {{"action": "up", "confidence": 0.9}}
- "move left" -> {{"action": "left", "confidence": 0.9}}
- "go home" -> {{"action": "home", "confidence": 0.95}}
- "select this" -> {{"action": "select", "confidence": 0.9}}
- "open app" -> {{"action": "open", "confidence": 0.9}}"""
            
            response = self.gemini_model.generate_content(prompt)  # type: ignore
            result_text = response.text.strip()
            
            # Try to extract JSON from response
            if "{" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)
                if result.get("confidence", 0) > 0.7:  # Only accept high confidence
                    return result.get("action")
        except Exception as e:
            print(f"Gemini API error: {e}")
        
        return None

    def _loop(self):
        print("Voice loop starting...")
        try:
            mic = sr.Microphone()
            print(f"Microphone found: {mic}")
        except Exception as e:
            print(f"ERROR: Could not initialize microphone: {e}")
            return
        
        try:
            with mic as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Ambient noise adjustment complete. Listening for commands...")
        except Exception as e:
            print(f"ERROR: Microphone setup error: {e}")
            return
        
        while self.listening:
            try:
                with mic as source:
                    # Reduced phrase time limit for faster response
                    audio = self.recognizer.listen(source, phrase_time_limit=2, timeout=1)
                try:
                    # First try Google speech recognition
                    print("Processing speech...")
                    text = self.recognizer.recognize_google(audio).lower().strip()  # type: ignore
                    print(f"✓ Heard: '{text}'")
                    self.detection_count += 1
                    
                    # Try Gemini API for better interpretation (with timeout)
                    gemini_action = self._process_with_gemini(text)
                    if gemini_action:
                        print(f"Gemini interpreted as: {gemini_action}")
                        self._handle_action(gemini_action)
                        self.success_count += 1
                    else:
                        # Fallback to basic command matching (faster)
                        print(f"Using basic command matching for: {text}")
                        self._handle_text(text)
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.WaitTimeoutError:
                    # Timeout is normal, continue listening
                    print("Listening timeout (no speech detected)")
                except sr.RequestError as e:
                    print(f"ERROR: Speech recognition service error: {e}")
                except Exception as e:
                    print(f"ERROR: Error processing audio: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"ERROR: Error in listening loop: {e}")
                import traceback
                traceback.print_exc()
            
            # Minimal sleep for lower latency
            time.sleep(0.02)
        
        print("Voice loop stopped")

    def _handle_text(self, transcript: str):
        """Handle text transcript with basic command matching."""
        for cmd in self.commands:
            if cmd in transcript:
                if cmd in ["up", "down", "left", "right", "home", "back", "select", "open"]:
                    self.success_count += 1
                    self._handle_action(cmd)
                elif cmd == "start":
                    pass
                elif cmd == "stop":
                    self.stop_listening()
                break

    def _handle_action(self, action: str):
        """Handle a recognized action."""
        # Only process if voice mode is actually active
        if not self.listening:
            print("Ignoring voice action - voice mode not active")
            return
        self.action_handler.handle_action(action, "voice")
        # Don't speak to reduce latency
        # self.speak(f"Moving {action}")
