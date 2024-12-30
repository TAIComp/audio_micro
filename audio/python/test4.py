import os
import warnings
import time
from openai import OpenAI
import threading
from datetime import datetime
from google.cloud import texttospeech
import pygame
from pathlib import Path
from enum import Enum
import random
import sounddevice as sd
from pynput import keyboard
import difflib
import numpy as np
import re

# Suppress ALSA warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygame.mixer")

# Redirect ALSA errors to /dev/null
try:
    from ctypes import *
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass


import os
import pyaudio
import queue
import threading
from google.cloud import speech

class ListeningState(Enum):
    FULL_LISTENING = "full_listening"
    INTERRUPT_ONLY = "interrupt_only"
    NOT_LISTENING = "not_listening"

class AudioTranscriber:
    def __init__(self):
        # Initialize audio parameters
        self.RATE = 16000
        self.CHUNK = int(self.RATE / 20)  # 50ms chunks instead of 100ms
        self.current_sentence = ""
        self.last_transcript = ""  # Add this to track interim results
        
        # Create a thread-safe queue for audio data
        self.audio_queue = queue.Queue()
        
        # Initialize the Speech client
        self.client = speech.SpeechClient()
        
        # Configure audio recording parameters
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True
        )

        # New additions
        self.openai_client = OpenAI()
        self.last_speech_time = datetime.now()
        self.last_final_transcript = ""
        self.is_processing = False
        self.last_sentence_complete = False

        # Add Text-to-Speech client
        self.tts_client = texttospeech.TextToSpeechClient()
        
        # Configure Text-to-Speech with a male voice
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Casual-K",  # Casual male voice
            ssml_gender=texttospeech.SsmlVoiceGender.MALE  # Changed from NEUTRAL to MALE
        )
        
        self.audio_config = {
            # Thresholds (in characters)
            'short_threshold': 25,     # e.g., "Hello" or "How are you?"
            'medium_threshold': 75,    # e.g., "Can you explain quantum physics?"
            'long_threshold': 150,     # e.g., "I need help understanding the relationship between..."
            
            # Context audio chances
            'context_chance': {
                'short': 0.05,  # 5%  - Very rare for greetings/simple queries
                'medium': 0.60, # 60% - Common for standard questions
                'long': 0.90,   # 90% - Almost always for complex queries
            },
            
            # Random filler audio chances
            'random_chance': {
                'short': 0.0,   # 0%  - Never for short queries
                'medium': 0.30, # 30% - Occasional for standard questions
                'long': 0.70,   # 70% - Common for complex queries
            },
            
            # Complexity multipliers
            'complexity_boost': 1.2,   # 20% boost when complexity indicators are found
            'max_chance': 0.95,        # Cap maximum probability at 95%
        }
        
        # Enhanced complexity indicators with weighted importance
        self.complexity_indicators = {
            # Question complexity (0.2-0.3 boost)
            'question_words': {
                'explain': 0.3,    # Higher weight for explanation requests
                'how': 0.25,       # Process-oriented questions
                'why': 0.25,       # Reasoning questions
                'what': 0.2,       # Basic information questions
                'when': 0.2,       # Temporal questions
                'where': 0.2,      # Spatial questions
                'which': 0.2,      # Selection questions
                'could you': 0.25, # Polite requests
                'would you': 0.25, # Polite requests
                'can you': 0.25    # Direct requests
            },
            
            # Technical complexity (0.3-0.4 boost)
            'technical_words': {
                'difference': 0.35,     # Comparison requests
                'between': 0.35,        # Relationship analysis
                'relationship': 0.4,    # Complex connections
                'compare': 0.35,        # Analytical thinking
                'analysis': 0.4,        # Detailed examination
                'concept': 0.35,        # Abstract thinking
                'theory': 0.4,          # Theoretical understanding
                'principle': 0.35,      # Fundamental knowledge
                'mechanism': 0.35,      # Process understanding
                'framework': 0.35       # Structural understanding
            },
            
            # Multi-part complexity (0.4-0.5 boost)
            'multi_part': {
                'and': 0.4,            # Additional content
                'also': 0.4,           # Supplementary information
                'additionally': 0.45,   # Complex additions
                'moreover': 0.45,       # Advanced connections
                'furthermore': 0.45,    # Extended explanation
                'first': 0.5,          # Sequential start
                'second': 0.5,         # Sequential continuation
                'finally': 0.5,        # Sequential conclusion
                'lastly': 0.5,         # Sequential end
                'however': 0.45        # Contrasting information
            }
        }
        
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Normal speed
            pitch=0.0  # Normal pitch
        )
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()

        # Add new attributes
        self.listening_state = ListeningState.FULL_LISTENING
        self.interrupt_commands = {
            "stop", "end", "shut up",
            "please stop", "stop please",
            "please end", "end please",
            "shut up please", "please shut up",
            "okay stop", "ok stop",
            "can you stop", "could you stop",
            "would you stop", "can you be quiet",
            "silence", "pause"
        }
        self.is_speaking = False
        
        # Update OpenAI model
        self.model = "gpt-4o-mini"  
        
        # Add transcript tracking
        self.last_interim_timestamp = time.time()
        self.interim_cooldown = 0.5  # seconds between interim updates
        
        # Add new flag for interrupt handling
        self.last_interrupt_time = time.time()
        self.interrupt_cooldown = 1.0  # 1 second cooldown between interrupts
        
        # Path to prerecorded interrupt acknowledgment
        self.interrupt_audio_path = Path("interruption.mp3")
        
        if not self.interrupt_audio_path.exists():
            print(f"Warning: Interrupt audio file '{self.interrupt_audio_path}' not found!")

        # Add this before keyboard listener initialization
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char == '`':
                    print("\nBacktick key interrupt detected!")
                    self.handle_keyboard_interrupt()
            except Exception as e:
                print(f"Keyboard handling error: {e}")

        # Initialize keyboard listener with the local on_press function
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

        # Initialize monitor stream as None
        self.monitor_stream = None

        # Initialize mic monitoring
        self.setup_mic_monitoring()

        # Add these new initializations
        self.current_audio_playing = False
        self.is_processing = False
        self.is_speaking = False
        self.pending_response = None

        # Define the mapping of indices to actual filenames (including apostrophes)
        self.context_file_mapping = {
            0: "Hmm_let_me_think_about_that",
            1: "So_basically",
            2: "Umm_okay_so",
            3: "Let_me_figure_this_out",
            4: "Hmm_that's_a_good_one",    # with apostrophe
            5: "Alright_let's_see",        # with apostrophe
            6: "Let's_think_about_this_for_a_moment",  # with apostrophe
            7: "Ooooh_that's_tricky",      # with apostrophe
            8: "Hmm_give_me_a_sec",
            9: "So_one_moment_um",
            10: "Oh_okay_okay",
            11: "Aha_just_a_sec",
            12: "Alright_let's_dive_into_that",  # with apostrophe
            13: "Okay_okay_let's_see",           # with apostrophe
            14: "Hmm_this_is_interesting",
            15: "Okay_okay_let's_get_to_work",   # with apostrophe
            16: "So_yeah",
            17: "Uhh_well",
            18: "You_know",
            19: "So_anyway",
            20: "Alright_umm",
            21: "Oh_well_hmm",
            22: "Well_you_see",
            23: "So_basically_yeah",
            24: "Umm_anyway",
            25: "It's_uh_kinda_like"       # with apostrophe
        }

        # Display mapping for reference
        print("\nRequired audio files in contextBased folder:")
        for idx, filename in self.context_file_mapping.items():
            print(f"{idx}: {filename}.mp3")

        # Update paths for audio folders with absolute paths
        self.base_dir = Path(__file__).parent.absolute()
        self.voice_caching_path = self.base_dir / "voiceCashingSys"
        self.context_folder = self.voice_caching_path / "contextBased"
        self.random_folder = self.voice_caching_path / "random"
        
        # Initialize the random audio files list
        self.random_file_names = [
            "Hmm_let_me_think.mp3",
            "So.mp3",
            "Umm_well_well_well.mp3",
            "You_know.mp3"
        ]
        
        # Verify and load random audio files
        self.random_audio_files = []
        for filename in self.random_file_names:
            file_path = self.random_folder / filename
            if file_path.exists():
                self.random_audio_files.append(file_path)
        
        self.used_random_files = set()

        # Add mic monitoring settings
        self.monitor_stream = None
        self.monitoring_active = True
        self.MONITOR_CHANNELS = 1
        self.MONITOR_DTYPE = 'float32'
        
        # Initialize mic monitoring
        self.setup_mic_monitoring()

        # Add new audio monitoring settings
        self.mic_volume = 0.5  # Default mic monitoring volume (0.0 to 1.0)
        self.noise_gate_threshold = 0.5  # Adjust this value to control noise reduction

        # Add new flag for tracking AI response generation
        self.stop_generation = False

    def setup_mic_monitoring(self):
        """Setup real-time mic monitoring with volume control and noise gate."""
        try:
            def audio_callback(indata, outdata, frames, time, status):
                if status:
                    print(f'Monitoring status: {status}')
                if self.monitoring_active and not self.is_speaking and not self.current_audio_playing:
                    # Convert to float32 if not already
                    audio_data = indata.astype(np.float32)
                    
                    # Apply noise gate
                    mask = np.abs(audio_data) < self.noise_gate_threshold
                    audio_data[mask] = 0
                    
                    # Apply volume control
                    audio_data = audio_data * self.mic_volume
                    
                    # Ensure we don't exceed [-1, 1] range
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    
                    outdata[:] = audio_data
                else:
                    outdata.fill(0)

            self.monitor_stream = sd.Stream(
                channels=self.MONITOR_CHANNELS,
                dtype=self.MONITOR_DTYPE,
                samplerate=self.RATE,
                callback=audio_callback,
                blocksize=self.CHUNK
            )
            self.monitor_stream.start()
            
        except Exception as e:
            print(f"Error setting up mic monitoring: {e}")
            self.monitor_stream = None

    def set_mic_volume(self, volume):
        """Set the microphone monitoring volume (0.0 to 1.0)."""
        self.mic_volume = max(0.0, min(1.0, volume))
        print(f"Mic monitoring volume set to: {self.mic_volume:.2f}")

    def set_noise_gate(self, threshold):
        """Set the noise gate threshold (0.0 to 1.0)."""
        self.noise_gate_threshold = max(0.0, min(1.0, threshold))
        print(f"Noise gate threshold set to: {self.noise_gate_threshold:.3f}")

    def get_context_and_completion(self, text):
        """Get both context index and completion status in a single API call."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Analyze the given text and provide TWO pieces of information:
                    1. The most appropriate filler phrase index (0-25) based on this context:
                    
                    Neutral Fillers:
                    0: "Hmm, let me think about that..."
                    1: "So, basically"
                    2: "Umm, okay so..."
                    3: "Let me figure this out..."
                    4: "Hmm, that's a good one..."
                    5: "Alright, let's see..."
                    6: "Let's think about this for a moment..."
                    
                    Casual and Friendly:
                    7: "Ooooh, that's tricky..."
                    8: "Hmm, give me a sec..."
                    9: "So, one moment... um"
                    10: "Oh, okay, okay..."
                    11: "Aha, just a sec..."
                    12: "Alright, let's dive into that!"
                    
                    Slightly Playful:
                    13: "Okay okay, let's see..."
                    14: "Hmm, this is interesting..."
                    15: "Okay okay, let's get to work..."
                    
                    Natural Fillers:
                    16: "So, yeah..."
                    17: "Uhh, well..."
                    18: "You know..."
                    19: "So, anyway..."
                    20: "Alright, umm..."
                    21: "Oh, well, hmm..."
                    
                    Casual Transitions:
                    22: "Well, you see..."
                    23: "So, basically, yeah..."
                    24: "Umm, anyway..."
                    25: "It's, uh, kinda like..."

                    2. Whether the sentence is complete, considering:
                    - Grammatical completeness (subject + predicate)
                    - Semantic completeness (complete thought/meaning)
                    - Natural ending point (proper punctuation or logical conclusion)
                    - Trailing indicators suggesting more is coming

                    Return ONLY in this format:
                    {"index": X, "complete": true/false}"""},
                    {"role": "user", "content": f"Analyze this text: '{text}'"}
                ]
            )
            
            # Safer parsing of the response
            response_text = response.choices[0].message.content
            # Replace 'true' and 'false' with 'True' and 'False' for Python
            response_text = response_text.replace('true', 'True').replace('false', 'False')
            result = eval(response_text)
            return result["index"], result["complete"]
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return 0, False  # Default values

    def get_ai_response(self, text):
        """Get response from OpenAI for the transcribed text."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,  # Updated model
                messages=[
                    {"role": "system", "content": """You are Quad, an AI-powered online teacher dedicated to making learning fun and engaging.

                    Guidelines:
                    1. Accuracy & Relevance: Provide correct, focused information
                    2. Clarity: Use simple language and short sentences
                    3. Examples: Include real-world examples when helpful
                    4. Engagement: Make learning interactive and interesting
                    5. Conciseness: Keep responses brief but informative
                    6. Adaptability: Match explanation complexity to the question
                    7. Encouragement: Use positive reinforcement
                    8. Verification: End with a quick comprehension check when appropriate

                    Response Format:
                    - Start with a direct answer
                    - Follow with a brief explanation if needed
                    - Include an example or analogy when relevant
                    - Keep total response length moderate
                    - Use natural, conversational language

                    Remember: You're speaking responses will be converted to speech, so keep sentences clear and well-paced."""},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    def reset_state(self):
        """Reset the transcriber state."""
        self.current_sentence = ""
        self.last_transcript = ""
        self.last_final_transcript = ""
        self.is_processing = False
        self.last_sentence_complete = False
        self.last_interim_timestamp = time.time()

    def text_to_speech(self, text):
        """Convert text to speech and save as MP3."""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            # Create output directory if it doesn't exist
            output_dir = Path("ai_responses")
            output_dir.mkdir(exist_ok=True)
            
            # Use a fixed filename instead of timestamp
            output_path = output_dir / "latest_response.mp3"
            
            # Save the audio file (overwrites existing file)
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
                
            return output_path
            
        except Exception as e:
            print(f"Text-to-Speech error: {e}")
            return None

    def play_audio_response(self, audio_path):
        """Modified to handle mic monitoring during playback."""
        try:
            self.is_speaking = True
            self.listening_state = ListeningState.INTERRUPT_ONLY
            
            # Temporarily disable mic monitoring during playback
            self.monitoring_active = False
            
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                if not self.is_speaking:  # Check if interrupted
                    pygame.mixer.music.stop()
                    break
                    
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            # Set the last speaking timestamp
            self._last_speaking_time = time.time()
            
            # Re-enable mic monitoring after playback
            self.monitoring_active = True
            
            # Reset all necessary states after audio finishes
            self.is_speaking = False
            self.listening_state = ListeningState.FULL_LISTENING
            self.current_audio_playing = False
            self.is_processing = False
            self.pending_response = None
            
            # Clear any remaining audio in the queue
            try:
                while True:
                    self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Reset transcription-related variables
            self.current_sentence = ""
            self.last_transcript = ""
            self.last_final_transcript = ""
            self.last_sentence_complete = False
            
            print("\nListening... (Press ` to interrupt)")

    def check_silence(self):
        """Check for silence and process accordingly."""
        while True:
            time.sleep(0.1)
            if self.is_processing:
                continue

            current_time = datetime.now()
            silence_duration = (current_time - self.last_speech_time).total_seconds()
            
            if self.last_final_transcript and not self.is_processing:
                # Get completion status for the current sentence
                _, is_complete = self.get_context_and_completion(self.last_final_transcript)
                
                # Condition 1: Complete sentence with 1 second silence
                # Condition 2: Any sentence with 5 seconds silence
                if (is_complete and silence_duration >= 1.0):
                    self.is_processing = True
                    print("\nProcessing: Sentence complete and 1 second silence passed")
                    audio_index, _ = self.get_context_and_completion(self.last_final_transcript)
                    processing_thread = threading.Thread(
                        target=self.process_complete_sentence,
                        args=(self.last_final_transcript, audio_index),
                        daemon=True
                    )
                    processing_thread.start()
                    self.reset_state()
                elif (not is_complete and silence_duration >= 4.0):
                    self.is_processing = True
                    print("\nProcessing: Incomplete sentence but 5 seconds silence passed")
                    audio_index, _ = self.get_context_and_completion(self.last_final_transcript)
                    processing_thread = threading.Thread(
                        target=self.process_complete_sentence,
                        args=(self.last_final_transcript, audio_index),
                        daemon=True
                    )
                    processing_thread.start()
                    self.reset_state()

    def audio_input_stream(self):
        while True:
            data = self.audio_queue.get()
            if data is None:
                break
            yield data

    def get_audio_input(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._fill_queue
        )
        return stream, audio

    def _fill_queue(self, in_data, frame_count, time_info, status_flags):
        self.audio_queue.put(in_data)
        return None, pyaudio.paContinue

    def handle_keyboard_interrupt(self):
        """Handle keyboard interruption."""
        current_time = time.time()
        if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
            print("\nBacktick interrupt detected!")
            self.handle_interrupt("keyboard")
            # Clear audio queue and force a small delay
            self.audio_queue.queue.clear()
            time.sleep(0.2)  # Small delay to ensure clean state

    def handle_interrupt(self, interrupt_type):
        """Common interrupt handling logic."""
        try:
            # Stop any ongoing audio playback
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                self.play_acknowledgment()
            
            # Signal to stop AI response generation
            self.stop_generation = True
            
            # Reset all audio and processing states
            self.is_speaking = False
            self.current_audio_playing = False
            self.is_processing = False
            self.pending_response = None
            self.listening_state = ListeningState.FULL_LISTENING
            
            if hasattr(self, 'ready_audio_path'):
                delattr(self, 'ready_audio_path')
            
            # Clear all transcript variables
            self.reset_state()
            self.last_interrupt_time = time.time()
            
            # Clear the audio queue more thoroughly
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Force a small delay to ensure clean state
            time.sleep(0.2)
            
            print("\nListening... (Press ` to interrupt)")
            
        except Exception as e:
            print(f"Error during interrupt handling: {e}")

    def process_audio_stream(self):
        stream, audio = self.get_audio_input()
        
        try:
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in self.audio_input_stream()
            )

            responses = self.client.streaming_recognize(
                self.streaming_config,
                requests
            )

            # Start the silence checking thread
            silence_thread = threading.Thread(target=self.check_silence, daemon=True)
            silence_thread.start()

            print("Listening... (Press ` to interrupt)")

            # Process responses
            try:
                for response in responses:
                    if response.results:
                        self.handle_responses([response])
            except StopIteration:
                pass
        
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            pygame.mixer.quit()
            self.keyboard_listener.stop()

    def handle_responses(self, responses):
        """Handle streaming responses with combined context and completion checks."""
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript.lower().strip()
            
            # Only check for voice commands if we're not already handling a keyboard interrupt
            current_time = time.time()
            
            # More precise interrupt command detection using regex
            if (current_time - self.last_interrupt_time >= self.interrupt_cooldown):
                # Create a regex pattern to match whole phrases
                pattern = r'\b(?:' + '|'.join(re.escape(cmd) for cmd in self.interrupt_commands) + r')\b'
                is_interrupt = re.search(pattern, transcript) is not None
                
                if is_interrupt:
                    print("\nInterrupt command detected!")
                    self.handle_interrupt("voice")
                    # Clear the current transcript and return immediately
                    self.audio_queue.queue.clear()  # Clear pending audio data
                    return  # Exit the method immediately
            
            # When system is speaking or during cooldown, skip all processing
            if (self.current_audio_playing or 
                self.is_speaking or 
                hasattr(self, '_last_speaking_time') and 
                time.time() - self._last_speaking_time < self.interrupt_cooldown):
                continue
            
            # When system is speaking, skip all other processing
            if self.current_audio_playing or self.is_speaking:
                continue
            
            # Add a cooldown period after speaking finishes
            if hasattr(self, '_last_speaking_time'):
                cooldown_period = 1.0  # 1 second cooldown
                if time.time() - self._last_speaking_time < cooldown_period:
                    continue
            
            # Only process speech when we're in FULL_LISTENING state
            if self.listening_state == ListeningState.FULL_LISTENING:
                # Ignore empty transcripts
                if not transcript.strip():
                    continue
                
                self.last_speech_time = datetime.now()
                
                # Handle interim results
                if not result.is_final:
                    current_time = time.time()
                    if (transcript != self.last_transcript and 
                        current_time - self.last_interim_timestamp >= self.interim_cooldown):
                        print(f'Interim: "{transcript}"')
                        self.last_transcript = transcript
                        self.last_interim_timestamp = current_time
                
                # Handle final results - Skip if this was an interrupt command
                else:
                    if any(cmd in transcript for cmd in self.interrupt_commands):
                        continue  # Skip processing if it's an interrupt command
                    
                    if not self.current_sentence:
                        self.current_sentence = transcript
                    else:
                        new_words = [word for word in transcript.split() 
                                   if word not in self.current_sentence.split()]
                        if new_words:
                            self.current_sentence += " " + " ".join(new_words)
                    
                    self.last_final_transcript = self.current_sentence
                    
                    # Get both context index and completion status in one call
                    audio_index, is_complete = self.get_context_and_completion(self.current_sentence)
                    self.last_sentence_complete = is_complete
                    
                    print(f'Final: "{self.current_sentence}" (Complete: {is_complete}, Audio Index: {audio_index})')
                    
                    # Only process immediately if the sentence is complete and 1 second has passed
                    current_time = datetime.now()
                    silence_duration = (current_time - self.last_speech_time).total_seconds()
                    
                    if is_complete and silence_duration >= 1.0 and not self.is_processing:
                        self.is_processing = True
                        print("\nProcessing: Sentence complete and 1 second silence passed")
                        processing_thread = threading.Thread(
                            target=self.process_complete_sentence,
                            args=(self.current_sentence, audio_index),
                            daemon=True
                        )
                        processing_thread.start()

    def process_complete_sentence(self, sentence, initial_audio_index):
        try:
            self.stop_generation = False  # Reset stop flag
            
            # 1. Start AI response generation and conversion in a thread
            def generate_and_convert():
                if self.stop_generation:
                    return
                    
                # Generate response
                response = self.get_ai_response(sentence)
                if response and not self.stop_generation:
                    self.pending_response = response
                    # Convert to audio
                    audio_path = self.text_to_speech(response)
                    if audio_path and not self.stop_generation:
                        self.ready_audio_path = audio_path
            
            response_thread = threading.Thread(
                target=generate_and_convert,
                daemon=True
            )
            response_thread.start()
            
            # 2. Play initial context audio and check for interruptions
            if not self.stop_generation:
                self.play_context_audio(initial_audio_index)
                while self.current_audio_playing and not self.stop_generation:
                    time.sleep(0.1)
            
            # 3. Play random audios until response is ready
            while not hasattr(self, 'ready_audio_path') and not self.stop_generation:
                if not self.current_audio_playing:
                    self.play_next_random_audio()
                while self.current_audio_playing and not self.stop_generation:
                    time.sleep(0.1)
            
            # 4. Play AI response if not interrupted
            if hasattr(self, 'ready_audio_path') and not self.stop_generation:
                self.play_audio_response(self.ready_audio_path)
                delattr(self, 'ready_audio_path')
            
        except Exception as e:
            print(f"Error processing sentence: {e}")
        finally:
            self.is_processing = False
            self.pending_response = None
            self.stop_generation = False

    def generate_response(self, sentence):
        """Generate AI response in a separate thread."""
        try:
            print("\nGenerating response...")
            response = self.get_ai_response(sentence)
            if response:
                print(f"\nAI Response: {response}")
                self.pending_response = response
        except Exception as e:
            print(f"Error generating response: {e}")
            self.pending_response = None

    def play_context_audio(self, index):
        """Play context-specific audio with improved feedback."""
        try:
            if self.is_speaking or self.current_audio_playing:
                return False
                
            if index in self.context_file_mapping:
                filename = f"{self.context_file_mapping[index]}.mp3"
                context_file = self.context_folder / filename
                
                if context_file.exists():
                    print(f"Playing context audio: {context_file.name}")
                    pygame.mixer.music.load(str(context_file))
                    pygame.mixer.music.play()
                    self.current_audio_playing = True
                    self.listening_state = ListeningState.INTERRUPT_ONLY
                    
                    def monitor_audio():
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        self.current_audio_playing = False
                        self.listening_state = ListeningState.FULL_LISTENING
                    
                    threading.Thread(target=monitor_audio, daemon=True).start()
                    return True
                
            return False
                
        except Exception as e:
            print(f"Error playing context audio: {e}")
            return False

    def play_acknowledgment(self):
        """Play the prerecorded acknowledgment audio."""
        try:
            # Stop any currently playing audio
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            
            # Verify file exists
            if not self.interrupt_audio_path.exists():
                print(f"Error: Interruption audio file not found at {self.interrupt_audio_path}")
                return
            
            # Load and play the interruption audio
            pygame.mixer.music.load(str(self.interrupt_audio_path))
            pygame.mixer.music.play()
            
            # Wait for the interruption audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Cleanup
            pygame.mixer.music.unload()
            
        except Exception as e:
            print(f"Error playing acknowledgment: {e}")

    def play_next_random_audio(self):
        """Play the next random audio file."""
        try:
            if not self.random_audio_files:
                return False
            
            # Reset used files if we've played them all
            if len(self.used_random_files) == len(self.random_audio_files):
                self.used_random_files.clear()
            
            # Select an unused random file
            available_files = [f for f in self.random_audio_files if f not in self.used_random_files]
            if not available_files:
                return False
            
            audio_file = random.choice(available_files)
            self.used_random_files.add(audio_file)
            
            pygame.mixer.music.load(str(audio_file))
            pygame.mixer.music.play()
            self.current_audio_playing = True
            self.listening_state = ListeningState.INTERRUPT_ONLY
            
            def monitor_audio():
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                self.current_audio_playing = False
                self.listening_state = ListeningState.FULL_LISTENING
            
            threading.Thread(target=monitor_audio, daemon=True).start()
            print(f"Playing random audio: {audio_file.name}")
            return True
            
        except Exception as e:
            print(f"Error playing random audio: {e}")
            return False

    def chain_random_audio(self):
        """Chain multiple random audio files while waiting for response."""
        while self.is_processing and not self.is_speaking:
            if not self.current_audio_playing:
                if not self.play_next_random_audio():
                    time.sleep(0.5)  # Wait before trying again if no audio available

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'monitor_stream') and self.monitor_stream is not None:
                self.monitor_stream.stop()
                self.monitor_stream.close()
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def analyze_request(self, sentence):
        """
        Enhanced request analysis with weighted complexity calculations.
        """
        length = len(sentence)
        word_count = len(sentence.split())
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        # Determine category based on length
        if length <= self.audio_config['short_threshold']:
            category = 'short'
        elif length <= self.audio_config['medium_threshold']:
            category = 'medium'
        else:
            category = 'long'
        
        # Initialize complexity analysis
        complexity_scores = {
            'question': 0.0,
            'technical': 0.0,
            'multi_part': 0.0
        }
        
        # Enhanced complexity analysis with phrase detection
        for i in range(len(words)):
            word = words[i]
            
            # Check single words
            for indicator_type, indicators in self.complexity_indicators.items():
                if word in indicators:
                    score_type = 'question' if indicator_type == 'question_words' else \
                               'technical' if indicator_type == 'technical_words' else 'multi_part'
                    complexity_scores[score_type] = max(
                        complexity_scores[score_type],
                        indicators[word]
                    )
            
            # Check phrases (up to 3 words)
            for phrase_length in range(2, 4):
                if i + phrase_length <= len(words):
                    phrase = ' '.join(words[i:i+phrase_length])
                    for indicator_type, indicators in self.complexity_indicators.items():
                        if phrase in indicators:
                            score_type = 'question' if indicator_type == 'question_words' else \
                                       'technical' if indicator_type == 'technical_words' else 'multi_part'
                            complexity_scores[score_type] = max(
                                complexity_scores[score_type],
                                indicators[phrase]
                            )
        
        # Calculate weighted total complexity
        total_complexity = sum(complexity_scores.values())
        
        # Get base chances from config
        context_chance = self.audio_config['context_chance'][category]
        random_chance = self.audio_config['random_chance'][category]
        
        # Apply complexity adjustments
        if total_complexity > 0:
            # Apply weighted boost based on complexity scores
            boost = 1 + (total_complexity * self.audio_config['complexity_boost'])
            
            # Calculate final probabilities with caps
            context_chance = min(
                self.audio_config['max_chance'],
                context_chance * boost
            )
            random_chance = min(
                self.audio_config['max_chance'],
                random_chance * boost
            )
        
        # Make final decisions
        use_context = random.random() < context_chance
        use_random = random.random() < random_chance
        
        # Detailed analysis output
        print(f"\nDetailed Request Analysis:")
        print(f"Length: {length} chars, Words: {word_count}")
        print(f"Category: {category}")
        print(f"Complexity Breakdown:")
        print(f"  - Question Complexity: {complexity_scores['question']:.2f}")
        print(f"  - Technical Complexity: {complexity_scores['technical']:.2f}")
        print(f"  - Multi-part Complexity: {complexity_scores['multi_part']:.2f}")
        print(f"Total Complexity Score: {total_complexity:.2f}")
        print(f"Final Probabilities:")
        print(f"  - Context Audio: {context_chance:.1%} (Using: {use_context})")
        print(f"  - Random Audio: {random_chance:.1%} (Using: {use_random})")
        
        return use_context, use_random

def main():
    try:
        transcriber = AudioTranscriber()
        print("Starting audio transcription... Speak into your microphone.")
        print("Press ` (backtick) to interrupt at any time.")
        transcriber.process_audio_stream()
    except KeyboardInterrupt:
        print("\nTranscription stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()



















