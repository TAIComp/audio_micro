import os
import warnings
import time

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

class AudioTranscriber:
    def __init__(self):
        # Initialize audio parameters
        self.RATE = 16000
        self.CHUNK = int(self.RATE / 10)  # 100ms chunks
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

            self.handle_responses(responses)
        
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def handle_responses(self, responses):
        print("Listening...")
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            # Handle both interim and final results
            if not result.is_final:
                # Only print if the transcript has changed
                if transcript != self.last_transcript:
                    print(f'Interim: "{transcript}"')
                    self.last_transcript = transcript
            else:
                if not self.current_sentence:
                    self.current_sentence = transcript
                else:
                    # Get the new words by comparing with existing sentence
                    current_words = set(self.current_sentence.split())
                    new_transcript_words = transcript.split()
                    
                    # Find new words that aren't in the current sentence
                    new_words = []
                    for word in new_transcript_words:
                        if word not in current_words:
                            new_words.append(word)
                    
                    # Append new words to the current sentence
                    if new_words:
                        self.current_sentence += " " + " ".join(new_words)
                
                print(f'Final: "{self.current_sentence}"')

def main():
    try:
        transcriber = AudioTranscriber()
        print("Starting audio transcription... Speak into your microphone.")
        transcriber.process_audio_stream()
    except KeyboardInterrupt:
        print("\nTranscription stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
