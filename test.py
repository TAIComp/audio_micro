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

import openai
import speech_recognition as sr  # To capture speech
from google.cloud import texttospeech  # To convert OpenAI's response to speech
import pygame  # For audio playback
from time import sleep
from openai import OpenAI  # Updated import
from google.oauth2 import service_account
import json
from concurrent.futures import ThreadPoolExecutor
import threading

openai.api_key = os.getenv('OPENAI_API_KEY')
def generate_text(prompt, max_tokens=50, temperature=0):
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "If the sentence most likely has continuation, return 'True', otherwise return 'False'. The idea is to find out whether the person finished his sentence or just takeing a pause. For instance, hello my name is Amirbek and I - would most likely have continuation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def listen_and_process_realtime(executor):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    current_sentence = ""
    speech_to_text_start = time.time()
    
    print("Listening for speech...")
    with microphone as source:
        # Adjusted recognition settings for better real-time performance
        recognizer.energy_threshold = 250  # Lowered for better sensitivity
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.6  # Reduced for faster response
        recognizer.phrase_threshold = 0.2  # Reduced for better word capture
        recognizer.adjust_for_ambient_noise(source, duration=0.3)  # Reduced duration
        
        last_speech_time = time.time()
        SILENCE_THRESHOLD = 2.0  # 2 seconds of silence to consider speech complete
        
        while True:
            try:
                # Reduced timeout for more frequent checks
                audio_chunk = recognizer.listen(source, timeout=0.5, phrase_time_limit=10)
                
                try:
                    # Added language parameter for better accuracy
                    chunk_text = recognizer.recognize_google(audio_chunk, 
                                                          show_all=False,
                                                          language='en-US')
                    if chunk_text:
                        current_time = time.time()
                        last_speech_time = current_time
                        
                        # Added real-time feedback
                        print(f"\r🎤 Current input: {chunk_text}", end="", flush=True)
                        current_sentence += chunk_text + " "
                        print(f"\n📝 Building sentence: {current_sentence}", flush=True)
                        
                        # Optimized sentence completion check
                        future_continuing = executor.submit(generate_text, current_sentence)
                        is_continuing = future_continuing.result(timeout=2.0)  # Added timeout
                        
                        if is_continuing and is_continuing.lower() == "false":
                            print("\n✅ Sentence complete!")
                            speech_to_text_time = time.time() - speech_to_text_start
                            process_complete_sentence(current_sentence, executor, speech_to_text_time)
                            current_sentence = ""
                            speech_to_text_start = time.time()
                            
                except sr.UnknownValueError:
                    # No speech detected in this chunk
                    current_time = time.time()
                    if current_sentence and (current_time - last_speech_time) > SILENCE_THRESHOLD:
                        # If we have a sentence and detected enough silence, process it
                        print("\nSilence detected, processing sentence...")
                        speech_to_text_time = time.time() - speech_to_text_start
                        process_complete_sentence(current_sentence, executor, speech_to_text_time)
                        current_sentence = ""
                        speech_to_text_start = time.time()
                    continue
                
            except sr.WaitTimeoutError:
                # Timeout on listening, check if we should process the current sentence
                current_time = time.time()
                if current_sentence and (current_time - last_speech_time) > SILENCE_THRESHOLD:
                    print("\nTimeout detected, processing sentence...")
                    speech_to_text_time = time.time() - speech_to_text_start
                    process_complete_sentence(current_sentence, executor, speech_to_text_time)
                    current_sentence = ""
                    speech_to_text_start = time.time()
                continue

def convert_text_to_speech(response):
    if not hasattr(convert_text_to_speech, 'credentials'):
        convert_text_to_speech.credentials = service_account.Credentials.from_service_account_file(
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
    
    client = texttospeech.TextToSpeechClient(credentials=convert_text_to_speech.credentials)
    
    synthesis_input = texttospeech.SynthesisInput(text=response)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Casual-K",  # Changed to Casual-K voice
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,  # You can adjust this between 0.25 and 4.0
        pitch=0.0  # You can adjust this between -20.0 and 20.0
    )

    # Generate speech
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Save to a file
    with open("response.mp3", "wb") as out:
        out.write(response.audio_content)
        print("Audio response saved as 'response.mp3'")

def generate_response(prompt, max_tokens=75, temperature=0.7):
    """Generate a tutor response using OpenAI."""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def play_audio(file_path):
    """Play the generated audio response."""
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        sleep(0.1)
    pygame.mixer.quit()

def main():
    pygame.init()
    executor = ThreadPoolExecutor(max_workers=3)
    
    try:
        while True:
            listen_and_process_realtime(executor)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        executor.shutdown()
        pygame.quit()

def process_complete_sentence(sentence, executor, speech_to_text_time):
    print("\n🤖 Processing: ", sentence)
    
    # Added caching for repeated queries
    cache_key = sentence.strip().lower()
    if hasattr(process_complete_sentence, 'response_cache') and cache_key in process_complete_sentence.response_cache:
        tutor_response = process_complete_sentence.response_cache[cache_key]
        print("💡 Using cached response")
    else:
        text_to_response_start = time.time()
        print("💭 Generating AI response...")
        future_response = executor.submit(generate_response, sentence)
        tutor_response = future_response.result(timeout=10.0)  # Added timeout
        text_to_response_time = time.time() - text_to_response_start
        
        # Initialize and update cache
        if not hasattr(process_complete_sentence, 'response_cache'):
            process_complete_sentence.response_cache = {}
        process_complete_sentence.response_cache[cache_key] = tutor_response
    
    # Phase 2: Text to Response
    text_to_response_start = time.time()
    print("Generating AI response...")
    future_response = executor.submit(generate_response, sentence)
    tutor_response = future_response.result()
    text_to_response_time = time.time() - text_to_response_start
    
    if tutor_response:
        print(f"Tutor: {tutor_response}")
        print(f"Text to response took: {text_to_response_time:.2f} seconds")
        
        # Phase 3: Response to Speech
        response_to_speech_start = time.time()
        
        # Convert to speech
        print("Converting to speech...")
        future_speech = executor.submit(convert_text_to_speech, tutor_response)
        future_speech.result()
        
        # Play audio
        print("Playing audio...")
        play_audio("response.mp3")
        
        response_to_speech_time = time.time() - response_to_speech_start
        print(f"Response to speech took: {response_to_speech_time:.2f} seconds")
        
        # Total time for all phases
        total_time = speech_to_text_time + text_to_response_time + response_to_speech_time
        print("\n=== Timing Summary ===")
        print(f"Phase 1 - Speech to Text: {speech_to_text_time:.2f}s")
        print(f"Phase 2 - Text to Response: {text_to_response_time:.2f}s")
        print(f"Phase 3 - Response to Speech: {response_to_speech_time:.2f}s")
        print(f"Total Processing Time: {total_time:.2f}s")
        print("====================\n")
    
    print("\nListening for next question...")

if __name__ == "__main__":
    main()
