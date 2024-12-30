import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

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
from dotenv import load_dotenv
from aiHandler import AIHandler

# Load environment variables from .env file
load_dotenv()

# Verify credentials path
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not credentials_path or not os.path.exists(credentials_path):
    raise Exception(f"Google credentials file not found at: {credentials_path}")

# Suppress ALSA warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygame.mixer")

# Redirect ALSA errors to /dev/null
try:
    from ctypes import *
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        return  # Just return instead of pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except OSError:
        print("Warning: Could not load ALSA library. Audio might not work properly.")
            
except Exception as e:
    print(f"Warning: Could not set ALSA error handler: {e}")


import os
import pyaudio
import queue
import threading
from google.cloud import speech

class ListeningState(Enum):
    FULL_LISTENING = "full_listening"
    INTERRUPT_ONLY = "interrupt_only"

def initialize_audio():
    """Initialize audio-related settings and suppress warnings."""
    # Load environment variables
    load_dotenv()

    # Verify credentials path
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path or not os.path.exists(credentials_path):
        raise Exception(f"Google credentials file not found at: {credentials_path}")

    # Suppress ALSA warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygame.mixer")

    # Redirect ALSA errors to /dev/null
    try:
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            return  # Just return instead of pass

        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        
        try:
            asound = cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(c_error_handler)
        except OSError:
            print("Warning: Could not load ALSA library. Audio might not work properly.")
            
    except Exception as e:
        print(f"Warning: Could not set ALSA error handler: {e}")

class AudioTranscriber:
    def __init__(self):
        try:
            # Initialize audio settings first (now using local function)
            initialize_audio()
            
            # Initialize pygame mixer with error handling
            try:
                pygame.mixer.init()
            except pygame.error as e:
                print(f"Warning: Could not initialize pygame mixer: {e}")
                # Continue anyway as some features might still work
                
            # Initialize AI handler
            self.aiHandler = AIHandler()
            
            # Initialize audio parameters
            self.RATE = 16000
            self.CHUNK = int(self.RATE / 10)  # 100ms chunks
            self.current_sentence = ""
            self.last_transcript = ""
            
            # Add mic monitoring settings
            self.MONITOR_CHANNELS = 1
            self.MONITOR_DTYPE = 'float32'
            self.monitoring_active = True
            self.mic_volume = 0.5
            self.noise_gate_threshold = 0.5
            
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

            self.error_count = 0
            self.max_errors = 3
            self.last_error_time = time.time()
            self.error_cooldown = 5  # seconds
            self.recovery_lock = threading.Lock()
            self.watchdog_active = True
            self.last_activity_time = time.time()
            self.activity_timeout = 30  # seconds
            
            # Start watchdog thread
            self.watchdog_thread = threading.Thread(target=self.watchdog_monitor, daemon=True)
            self.watchdog_thread.start()

            # Add new variables for interrupt detection
            self.current_interrupt_buffer = ""
            self.interrupt_buffer_timeout = 2.0  # seconds
            self.last_interrupt_buffer_update = time.time()

        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def setup_mic_monitoring(self):
        """Setup real-time mic monitoring with volume control and noise gate."""
        try:
            def audio_callback(indata, outdata, frames, time, status):
                try:
                    # Only print status messages if it's not an underflow
                    if status and not status.input_underflow and not status.output_underflow:
                        print(f'Monitoring status: {status}')
                        
                    if self.monitoring_active and not self.is_speaking and not self.current_audio_playing:
                        # Convert to float32 if not already
                        audio_data = indata.copy()  # Create a copy to avoid modifying input
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                        
                        # Apply noise gate
                        mask = np.abs(audio_data) < self.noise_gate_threshold
                        audio_data[mask] = 0
                        
                        # Apply volume control
                        audio_data = audio_data * self.mic_volume
                        
                        # Ensure we don't exceed [-1, 1] range
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                        
                        # Fill output buffer
                        outdata[:] = audio_data
                    else:
                        outdata.fill(0)
                        
                except Exception as e:
                    print(f"Error in audio callback: {e}")
                    outdata.fill(0)  # Ensure output is silent on error

            # Try to create the stream with more conservative settings
            self.monitor_stream = sd.Stream(
                channels=self.MONITOR_CHANNELS,
                dtype=self.MONITOR_DTYPE,
                samplerate=self.RATE,
                callback=audio_callback,
                blocksize=2048,  # Increased block size for stability
                latency='high',
                device=None,  # Use default device
                prime_output_buffers_using_stream_callback=True  # Help prevent underflows
            )
            
            # Start the stream in a try-except block
            try:
                self.monitor_stream.start()
            except sd.PortAudioError as e:
                print(f"Error starting audio stream: {e}")
                self.monitor_stream = None
                
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

    def get_ai_response(self, text):
        """Wrapper for AI handler method"""
        return self.aiHandler.get_ai_response(text)

    def reset_state(self, force=False):
        """Reset the transcriber state."""
        try:
            self.is_speaking = False
            self.is_processing = False
            self.stop_generation = False
            self.current_sentence = ""
            self.last_transcript = ""
            self.last_final_transcript = ""
            self.last_sentence_complete = False
            self.last_interim_timestamp = time.time()
            
            # Ensure pygame mixer is in a clean state
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.unload()
                    pygame.mixer.quit()
            except:
                pass
            
            print("\nListening... (Press ` to interrupt)")
        except Exception as e:
            print(f"Error resetting state: {e}")

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
            try:
                time.sleep(0.1)
                # Skip if already processing or speaking
                if self.is_processing or self.is_speaking:
                    continue

                current_time = datetime.now()
                silence_duration = (current_time - self.last_speech_time).total_seconds()
                
                if self.last_final_transcript and not self.is_processing:
                    # Get completion status for the current sentence
                    is_complete = self.aiHandler.is_sentence_complete(self.last_final_transcript)
                    
                    # Only process if not already processing and enough silence has passed
                    if (is_complete and silence_duration >= 1.0) and not self.is_processing:
                        self.is_processing = True
                        print("\nProcessing: Sentence complete and 1 second silence passed")
                        try:
                            # Process in the same thread for better control
                            self.process_complete_sentence(self.last_final_transcript)
                        except Exception as e:
                            print(f"Error in silence check processing: {e}")
                            self.reset_state()
                            
            except Exception as e:
                print(f"Error in check_silence: {e}")
                time.sleep(1)  # Add delay to prevent rapid error loops

    def audio_input_stream(self):
        while True:
            data = self.audio_queue.get()
            if data is None:
                break
            yield data

    def get_audio_input(self):
        """Initialize audio input with error handling and device selection."""
        try:
            audio = pyaudio.PyAudio()
            
            # List available input devices
            info = audio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            # Find default input device
            default_input = None
            for i in range(numdevices):
                device_info = audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    if device_info.get('defaultSampleRate') == self.RATE:
                        default_input = i
            
            if default_input is None:
                default_input = audio.get_default_input_device_info()['index']
            
            # Open the audio stream with explicit device selection
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.RATE,
                input=True,
                input_device_index=default_input,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._fill_queue
            )
            
            if not stream.is_active():
                stream.start_stream()
                
            return stream, audio
            
        except Exception as e:
            print(f"\nError initializing audio input: {e}")
            print("Please check if your microphone is properly connected and permissions are set correctly.")
            print("You may need to grant microphone access to the application.")
            raise

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
            current_time = time.time()
            if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
                # Stop any ongoing audio playback
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    self.play_acknowledgment()
                
                # Signal to stop AI response generation
                self.stop_generation = True
                
                # Reset all states
                self.is_speaking = False
                self.current_audio_playing = False
                self.is_processing = False
                self.pending_response = None
                self.listening_state = ListeningState.FULL_LISTENING
                
                # Clear audio queue
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                self.reset_state()
                self.last_interrupt_time = current_time
                
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
        """Handle streaming responses with state-based processing and improved interrupt detection."""
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript.lower().strip()
            current_time = time.time()
            
            # Check for interrupts first, regardless of state
            if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
                # Create a regex pattern to match whole phrases
                pattern = r'\b(?:' + '|'.join(re.escape(cmd) for cmd in self.interrupt_commands) + r')\b'
                is_interrupt = re.search(pattern, transcript) is not None
                
                if is_interrupt:
                    print("\nInterrupt command detected!")
                    self.handle_interrupt("voice")
                    self.audio_queue.queue.clear()  # Clear pending audio data
                    return  # Exit immediately after interrupt
            
            # Skip processing if system is speaking or in cooldown
            if (self.current_audio_playing or 
                self.is_speaking or 
                hasattr(self, '_last_speaking_time') and 
                time.time() - self._last_speaking_time < self.interrupt_cooldown):
                continue
            
            if self.listening_state == ListeningState.FULL_LISTENING:
                # Skip empty transcripts
                if not transcript.strip():
                    continue
                
                self.last_speech_time = datetime.now()
                
                if result.is_final:
                    if not self.is_processing:
                        print(f'\nFinal: "{transcript}"')
                        # Check sentence completion
                        is_complete = self.aiHandler.is_sentence_complete(transcript)
                        print(f"\nProcessing triggered - Sentence complete: {is_complete}")
                        
                        if is_complete:
                            print("\nChanging state to INTERRUPT_ONLY for processing")
                            self.listening_state = ListeningState.INTERRUPT_ONLY
                            
                            # Clear any pending audio data
                            while not self.audio_queue.empty():
                                try:
                                    self.audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                            
                            # Start processing in a separate thread
                            processing_thread = threading.Thread(
                                target=self.process_transcript,
                                args=(transcript,),
                                daemon=True
                            )
                            processing_thread.start()
                            continue
                else:
                    # Handle interim results for transcription updates only
                    if transcript != self.last_transcript and current_time - self.last_interim_timestamp >= 0.5:
                        print(f'\nInterim: "{transcript}"')
                        self.last_transcript = transcript
                        self.last_interim_timestamp = current_time
            


    def process_transcript(self, transcript):
        """Process transcript and generate response in parallel."""
        try:
            self.is_processing = True
            
            # Start AI response generation immediately
            response_future = threading.Thread(
                target=self.get_ai_response,
                args=(transcript,),
                daemon=True
            )
            response_future.start()
            
            # Process the response as it comes in
            self.process_complete_sentence(transcript)
            
        except Exception as e:
            print(f"Error processing transcript: {e}")
        finally:
            self.is_processing = False

    def process_complete_sentence(self, sentence):
        try:
            # Use the existing temp directory
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Create a unique subdirectory for this processing session
            session_dir = temp_dir / f"audio_{int(time.time() * 1000)}"
            session_dir.mkdir(exist_ok=True)
            
            # Set states at the beginning
            self.stop_generation = False
            self.is_speaking = True
            self.is_processing = True
            
            # Store the current sentence being processed
            current_processing_sentence = sentence
            
            # Use thread-safe queue for audio buffer
            audio_buffer = queue.Queue()
            is_playing = False
            playback_active = True
            
            def cleanup_old_sessions():
                """Clean up old session directories."""
                try:
                    for old_dir in temp_dir.glob("audio_*"):
                        if old_dir != session_dir:
                            try:
                                for file in old_dir.glob("*"):
                                    file.unlink()
                                old_dir.rmdir()
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Clean up old sessions before starting
            cleanup_old_sessions()
            
            def play_audio_chunks():
                nonlocal is_playing, playback_active
                try:
                    while playback_active and not self.stop_generation:
                        if not is_playing and not audio_buffer.empty():
                            try:
                                chunk = audio_buffer.get_nowait()
                                if chunk and chunk.exists():
                                    try:
                                        is_playing = True
                                        
                                        # Ensure pygame mixer is initialized
                                        if not pygame.mixer.get_init():
                                            pygame.mixer.init()
                                        
                                        pygame.mixer.music.load(str(chunk))
                                        pygame.mixer.music.play()
                                        
                                        # Wait for current chunk to finish
                                        while pygame.mixer.music.get_busy() and not self.stop_generation:
                                            pygame.time.Clock().tick(10)
                                        
                                        # Cleanup after playing
                                        pygame.mixer.music.unload()
                                        chunk.unlink()
                                        is_playing = False
                                        
                                    except Exception as e:
                                        print(f"Error playing chunk: {e}")
                                        is_playing = False
                            except queue.Empty:
                                time.sleep(0.1)
                        else:
                            time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in playback thread: {e}")
                finally:
                    is_playing = False
                    try:
                        pygame.mixer.music.unload()
                    except:
                        pass
            
            # Start audio playback thread
            playback_thread = threading.Thread(target=play_audio_chunks, daemon=True)
            playback_thread.start()
            
            try:
                current_sentence = ""
                chunk_counter = 0
                
                # Generate and process response chunks
                for text_chunk in self.get_ai_response(sentence):
                    if self.stop_generation:
                        break
                    
                    if text_chunk:
                        current_sentence += text_chunk
                        
                        # Check if we have a complete sentence
                        sentences = re.split(r'([.!?])\s*', current_sentence)
                        
                        # Process complete sentences
                        while len(sentences) >= 2:  # We need at least [text, punctuation]
                            sentence_text = sentences[0] + sentences[1]  # Combine text with punctuation
                            
                            try:
                                # Convert sentence to speech
                                synthesis_input = texttospeech.SynthesisInput(text=sentence_text)
                                response = self.tts_client.synthesize_speech(
                                    input=synthesis_input,
                                    voice=self.voice,
                                    audio_config=self.audio_config
                                )
                                
                                # Save sentence to temporary file
                                temp_path = session_dir / f"chunk_{chunk_counter}.mp3"
                                with open(temp_path, "wb") as out:
                                    out.write(response.audio_content)
                                
                                audio_buffer.put(temp_path)
                                chunk_counter += 1
                                
                                # Remove processed sentence from the buffer
                                sentences = sentences[2:]
                                current_sentence = ''.join(sentences)
                                
                            except Exception as e:
                                print(f"Error processing sentence: {e}")
                                sentences = sentences[2:]
                                current_sentence = ''.join(sentences)
                                continue
                
                # Process any remaining text as the final sentence
                if current_sentence.strip():
                    try:
                        synthesis_input = texttospeech.SynthesisInput(text=current_sentence)
                        response = self.tts_client.synthesize_speech(
                            input=synthesis_input,
                            voice=self.voice,
                            audio_config=self.audio_config
                        )
                        
                        temp_path = session_dir / f"chunk_{chunk_counter}.mp3"
                        with open(temp_path, "wb") as out:
                            out.write(response.audio_content)
                        
                        audio_buffer.put(temp_path)
                        
                    except Exception as e:
                        print(f"Error processing final sentence: {e}")
                
                # Wait for all audio to finish playing
                while not audio_buffer.empty() or is_playing:
                    if self.stop_generation:
                        break
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"Error generating response: {e}")
            finally:
                # Signal playback thread to stop
                playback_active = False
            
            # Wait for playback thread to finish
            playback_thread.join(timeout=2.0)
            
            # Cleanup session directory
            try:
                for file in session_dir.glob("*"):
                    try:
                        if file.exists():
                            file.unlink()
                    except Exception:
                        pass
                try:
                    session_dir.rmdir()
                except Exception:
                    pass
            except Exception as e:
                print(f"Error cleaning up session directory: {e}")
            
        except Exception as e:
            print(f"Error processing sentence: {e}")
        finally:
            # Reset all states after everything is complete
            self.is_speaking = False
            self.is_processing = False
            self.stop_generation = False
            
            # Only reset transcripts if we're still processing the same sentence
            if self.last_final_transcript == current_processing_sentence:
                self.current_sentence = ""
                self.last_transcript = ""
                self.last_final_transcript = ""
                self.last_sentence_complete = False
                self.last_interim_timestamp = time.time()
            
            # Ensure pygame mixer is in a clean state
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.unload()
                    pygame.mixer.quit()
            except:
                pass
            
            print("\nListening... (Press ` to interrupt)")

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

    def watchdog_monitor(self):
        """Monitor system health and recover from deadlocks."""
        while self.watchdog_active:
            try:
                time.sleep(1)
                current_time = time.time()
                
                # Check for system freeze
                if current_time - self.last_activity_time > self.activity_timeout:
                    print("\nWatchdog: System appears frozen, initiating recovery...")
                    self.emergency_recovery()
                    
            except Exception as e:
                print(f"Error in watchdog: {e}")
    
    def emergency_recovery(self):
        """Emergency recovery procedure."""
        try:
            with self.recovery_lock:
                print("\nInitiating emergency recovery...")
                
                # Force stop all processing
                self.stop_generation = True
                self.is_processing = False
                self.is_speaking = False
                
                # Clean up audio resources
                try:
                    if pygame.mixer.get_init():
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        pygame.mixer.quit()
                except:
                    pass
                
                # Clean up streams
                try:
                    if hasattr(self, 'monitor_stream') and self.monitor_stream is not None:
                        self.monitor_stream.stop()
                        self.monitor_stream.close()
                except:
                    pass
                
                # Reinitialize system
                self.reinitialize_audio_system()
                
                # Reset states
                self.reset_state(force=True)
                
                # Update activity timestamp
                self.last_activity_time = time.time()
                
                print("\nEmergency recovery completed")
                
        except Exception as e:
            print(f"Error in emergency recovery: {e}")
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_time = time.time()
    
    def safe_process_audio_stream(self):
        """Wrapper for process_audio_stream with error recovery."""
        while True:
            try:
                self.process_audio_stream()
            except Exception as e:
                current_time = time.time()
                if current_time - self.last_error_time > self.error_cooldown:
                    self.error_count = 0
                
                self.error_count += 1
                self.last_error_time = current_time
                
                print(f"\nError in audio stream: {e}")
                
                if self.error_count >= self.max_errors:
                    print("\nToo many errors, initiating emergency recovery...")
                    self.emergency_recovery()
                    self.error_count = 0
                else:
                    self.reset_state(force=True)
                
                time.sleep(1)
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.watchdog_active = False
            if hasattr(self, 'watchdog_thread'):
                self.watchdog_thread.join(timeout=1.0)
            
            # Stop all audio playback
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                pygame.mixer.quit()
            
            # Clean up monitor stream
            if hasattr(self, 'monitor_stream') and self.monitor_stream is not None:
                self.monitor_stream.stop()
                self.monitor_stream.close()
            
            # Clean up temporary files
            temp_dir = Path("temp")
            if temp_dir.exists():
                for session_dir in temp_dir.glob("audio_*"):
                    try:
                        for file in session_dir.glob("*"):
                            try:
                                if file.exists():
                                    file.unlink()
                            except Exception:
                                pass
                        try:
                            session_dir.rmdir()
                        except Exception:
                            pass
                    except Exception:
                        pass
                
        except Exception as e:
            print(f"Cleanup error: {e}")

    def reinitialize_audio_system(self):
        """Reinitialize audio system after recovery."""
        try:
            # Reinitialize pygame mixer
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            pygame.mixer.init()
            
            # Reinitialize mic monitoring
            self.setup_mic_monitoring()
            
            # Reset all flags
            self.is_speaking = False
            self.is_processing = False
            self.stop_generation = False
            self.current_audio_playing = False
            
            print("\nAudio system reinitialized successfully")
        except Exception as e:
            print(f"Error reinitializing audio system: {e}")

def main():
    try:
        transcriber = AudioTranscriber()
        print("\nStarting audio transcription... Speak into your microphone.")
        print("Press ` (backtick) to interrupt at any time.")
        transcriber.process_audio_stream()
    except KeyboardInterrupt:
        print("\nTranscription stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()



















