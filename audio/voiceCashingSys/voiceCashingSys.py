# from google.cloud import texttospeech
# import os
# from dotenv import load_dotenv

# num = 25

# def text_to_speech(text, output_file='output.mp3', language_code='en-US'):
#     try:
#         # Load environment variables
#         load_dotenv()
        
#         # Set Google Cloud credentials
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
#         # Print debug information
#         print(f"Converting text: {text}")
#         print(f"Output file: {output_file}")
#         print(f"Language code: {language_code}")

#         # Initialize the client
#         client = texttospeech.TextToSpeechClient()
#         print("Successfully initialized Text-to-Speech client")

#         # Set the text input to be synthesized
#         synthesis_input = texttospeech.SynthesisInput(text=text)

#         # Build the voice request
#         voice = texttospeech.VoiceSelectionParams(
#             language_code=language_code,
#             name='en-US-Casual-K',
#             ssml_gender=texttospeech.SsmlVoiceGender.MALE
#         )

#         # Build the audio config
#         audio_config = texttospeech.AudioConfig(
#             audio_encoding=texttospeech.AudioEncoding.MP3,
#             speaking_rate=.4,
#             pitch=2,  # -20.0 to 20.0
#             volume_gain_db=0
#         )

#         # Perform the text-to-speech request
#         print("Sending request to Google Cloud...")
#         response = client.synthesize_speech(
#             input=synthesis_input,
#             voice=voice,
#             audio_config=audio_config
#         )

#         # Get the full path for the output file
#         current_dir = os.getcwd()
#         output_path = os.path.join(current_dir, 'random', output_file)
        
#         # Create the directory structure if it doesn't exist
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         # Write the response to the output file
#         with open(output_path, 'wb') as out:
#             out.write(response.audio_content)
#             print(f'Audio content successfully written to "{output_path}"')

#     except Exception as e:
#         print(f"Detailed error: {str(e)}")
#         print(f"Error type: {type(e).__name__}")
#         raise

# def get_unique_filename(base_filename):
#     """Generate a unique filename by adding a number if the file exists."""
#     # Modify to check in the random folder
#     full_path = os.path.join('random', base_filename)
#     if not os.path.exists(full_path):
#         return base_filename
    
#     # Split the filename into name and extension
#     name, ext = os.path.splitext(base_filename)
#     counter = 1
    
#     # Keep trying new numbers until we find an unused filename
#     while os.path.exists(os.path.join('random', f"{name}_{counter}{ext}")):
#         counter += 1
    
#     return f"{name}_{counter}{ext}"

# if __name__ == "__main__":
    
#         # List of all filler phrases
#         filler_phrases = [
#             # Neutral Fillers
#             "Hmm, let me think about that...", # 0
        
#             "So, basically", # 1
#             "Umm, okay so...", # 2
#             "Let me figure this out...", # 3
#             "Hmm, that's a good one...", # 4
#             "Alright, let's see...", # 5
#             "Let's think about this for a moment...", # 6
            
            
#             # Casual and Friendly
#             "Ooooh, that's tricky...", # 7
#             "Hmm, give me a sec...", # 8
           
#             "So, one moment... um", # 9
#             "Oh, okay, okay...", # 10
#             "Aha, just a sec...", # 11
#             "Alright, let's dive into that!", # 12
            
#             # Slightly Playful
#             "Okay okay, let's see...", # 13
            
#             "Hmm, this is interesting...", # 14
#             "Okay okay, let's get to work...", # 15
            
#             # Natural Fillers
#             "So, yeah...", # 16
#             "Uhh, well...", # 17
#             "You know...", # 18
            
#             "So, anyway...", # 19
#             "Alright, umm...", # 20
            
#             "Oh, well, hmm...", # 21
            
            
#             # Casual Transitions
#             "Well, you see...", #22
            
#             "So, basically, yeah...", # 23
#             "Umm, anyway...", # 24
#             "It's, uh, kinda like..." # 25
            
#         ]
        
#         # Process each phrase
       
#         phrase = "Hmm, let me think ..."
#                 # Create base filename from text (replace spaces and sanitize)
#         base_filename = (
#         phrase
#         .replace(' ', '_')
#         .replace('/', '_')
#         .replace('\\', '_')
#         .replace(':', '_')
#         .replace('"', '_')
#         .replace('<', '_')
#         .replace('>', '_')
#         .replace('|', '_')
#         .replace('?', '_')
#         .replace('*', '_')
#         .replace(',', '')
#         .replace('...', '')
#         .replace('!', '')
#         ) + '.mp3'
                
#         # Get a unique filename
#         output_filename = get_unique_filename(base_filename)
        
#         print(f"\nProcessing: {phrase}")
#         # Convert text to speech with unique filename
#         text_to_speech(phrase, output_file=output_filename)
        
    
       
        
    
