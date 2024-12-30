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
            "stop talking", "stop speaking",
            "be quiet", "quiet please",
            "that's enough", "thats enough",
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

        # Initialize pygame for event handling
        pygame.init()
        # Create a small visible window that stays on top
        self.screen = pygame.display.set_mode((200, 100))
        pygame.display.set_caption("Press SPACE to interrupt")
        # Keep window on top
        os.environ['SDL_WINDOW_ALWAYS_ON_TOP'] = '1'

    def check_sentence_completion(self, text):
        """Check if the sentence is complete using OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a linguistic expert focused solely on analyzing sentence completion. 
                    Evaluate whether the given sentence is complete by checking for:

                    1. Grammatical completeness (subject + predicate)
                    2. Semantic completeness (complete thought/meaning)
                    3. Natural ending point (proper punctuation or logical conclusion)
                    4. Trailing indicators that suggest more is coming:
                       - Hanging conjunctions (and, but, or)
                       - Incomplete phrases
                       - Missing essential parts of speech
                       - Unfinished comparative statements
                       - Interrupted thoughts

                    Return ONLY 'True' if the sentence is complete, or 'False' if it likely has continuation.

                    Examples:
                    - "My name is John." -> True
                    - "My name is" -> False
                    - "I went to the store and" -> False
                    - "Although it was raining" -> False
                    - "The weather is nice today" -> True
                    - "If you want to come with us" -> False
                    - "Let me think about" -> False"""},
                    {"role": "user", "content": f"Analyze this sentence for completion: '{text}'"}
                ]
            )

            return response.choices[0].message.content.strip().lower() == 'true'
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return False

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
        """Play the audio response with improved interrupt capability."""
        try:
            self.is_speaking = True
            self.listening_state = ListeningState.INTERRUPT_ONLY
            
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            
            # Add event handling for better interrupt control
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                if not self.is_speaking:  # Check if interrupted
                    pygame.mixer.music.stop()
                    break
                    
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            self.is_speaking = False
            self.listening_state = ListeningState.FULL_LISTENING

    def check_silence(self):
        """Check for silence and process accordingly."""
        while True:
            time.sleep(0.1)
            if self.is_processing:
                continue

            current_time = datetime.now()
            silence_duration = (current_time - self.last_speech_time).total_seconds()

            if self.last_final_transcript and not self.is_processing:
                if self.last_sentence_complete and silence_duration >= 1:
                    self.is_processing = True
                    print("\nSentence is complete. Generating response...")
                    response = self.get_ai_response(self.last_final_transcript)
                    if response:
                        print(f"\nAI Response: {response}")
                        # Convert response to speech and play it
                        audio_path = self.text_to_speech(response)
                        if audio_path:
                            print(f"Playing audio response from: {audio_path}")
                            self.play_audio_response(audio_path)
                        self.reset_state()
                    self.is_processing = False
                elif not self.last_sentence_complete and silence_duration >= 2:
                    # Similar handling for incomplete sentences
                    self.is_processing = True
                    print("\nIncomplete sentence, but generating response...")
                    response = self.get_ai_response(self.last_final_transcript)
                    if response:
                        print(f"\nAI Response: {response}")
                        audio_path = self.text_to_speech(response)
                        if audio_path:
                            print(f"Playing audio response from: {audio_path}")
                            self.play_audio_response(audio_path)
                        self.reset_state()
                    self.is_processing = False

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

    def check_keyboard_events(self):
        """Check for keyboard events."""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("\nSpace key interrupt detected!")
                    self.handle_keyboard_interrupt()
                    return True
            elif event.type == pygame.QUIT:
                return False
        return True

    def handle_keyboard_interrupt(self):
        """Handle keyboard interruption."""
        current_time = time.time()
        if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
            print("\nKeyboard interrupt detected!")
            if self.is_speaking:
                pygame.mixer.music.stop()
                self.play_acknowledgment()
            self.is_speaking = False
            self.listening_state = ListeningState.FULL_LISTENING
            
            self.reset_state()
            self.last_interrupt_time = current_time
            
            # Clear the audio queue
            try:
                while True:
                    self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            
            print("Ready for new input...")

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

            print("\nListening... (Press SPACE to interrupt)")
            
            # Create a separate thread for keyboard event checking
            def check_events():
                while True:
                    if not self.check_keyboard_events():
                        break
                    time.sleep(0.1)
            
            keyboard_thread = threading.Thread(target=check_events, daemon=True)
            keyboard_thread.start()

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
            pygame.quit()

    def handle_responses(self, responses):
        silence_thread = threading.Thread(target=self.check_silence, daemon=True)
        silence_thread.start()

        for response in responses:
            if not response.results or not response.results[0].alternatives:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript.lower().strip()
            
            # Check for interrupt commands first
            if any(cmd in transcript for cmd in self.interrupt_commands):
                current_time = time.time()
                if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
                    print("\nInterrupt command detected!")
                    if self.is_speaking:
                        pygame.mixer.music.stop()
                        # Play acknowledgment before resetting state
                        self.play_acknowledgment()
                    self.is_speaking = False
                    self.listening_state = ListeningState.FULL_LISTENING
                    
                    self.reset_state()
                    self.last_interrupt_time = current_time
                    
                    # Clear the audio queue
                    try:
                        while True:
                            self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    print("Ready for new input...")
                    continue
                continue  # Skip processing this transcript entirely
            
            if self.listening_state == ListeningState.FULL_LISTENING:
                self.last_speech_time = datetime.now()
                
                if not result.is_final:
                    current_time = time.time()
                    if (transcript != self.last_transcript and 
                        current_time - self.last_interim_timestamp >= self.interim_cooldown):
                        print(f'Interim: "{transcript}"')
                        self.last_transcript = transcript
                        self.last_interim_timestamp = current_time
                else:
                    if not self.current_sentence:
                        self.current_sentence = transcript
                    else:
                        new_words = [word for word in transcript.split() 
                                   if word not in self.current_sentence.split() 
                                   and word not in self.interrupt_commands]
                        if new_words:
                            self.current_sentence += " " + " ".join(new_words)
                    
                    self.last_final_transcript = self.current_sentence
                    print(f'Final: "{self.current_sentence}"')
                    self.last_sentence_complete = self.check_sentence_completion(self.current_sentence)

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

def main():
    try:
        transcriber = AudioTranscriber()
        print("Starting audio transcription... Speak into your microphone.")
        print("Press SPACE to interrupt at any time.")
        transcriber.process_audio_stream()
    except KeyboardInterrupt:
        print("\nTranscription stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
