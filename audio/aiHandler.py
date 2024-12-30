from openai import OpenAI
import os
from dotenv import load_dotenv

class AIHandler:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Set the model
        self.model = "gpt-4o-mini"  # Using GPT-4o mini model

    def get_ai_response(self, user_input):
        """Get AI response for user input."""
        try:
            # Get response from OpenAI with system and user messages
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a helpful AI assistant engaged in verbal conversation. 
                    Keep your responses natural, concise, and conversational. 
                    Aim to be informative while maintaining a friendly tone.
                    Avoid overly long or technical responses unless specifically asked.
                    If you're not sure about something, it's okay to say so."""},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                stream=True
            )
            
            # Stream the response
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield content
            
        except Exception as e:
            print(f"Error getting AI response: {e}")
            yield "I apologize, but I encountered an error. Could you please repeat that?"

    def is_sentence_complete(self, text):
        """Check if the sentence is complete."""
        try:
            # Add timeout to prevent hanging
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a sentence completion analyzer.
                    Analyze if the given text forms a complete thought or sentence.
                    Consider:
                    - Grammatical completeness
                    - Semantic completeness
                    - Natural ending (proper punctuation)
                    - No trailing indicators or incomplete phrases
                    
                    Respond with ONLY 'true' or 'false' (lowercase)."""},
                    {"role": "user", "content": f"Is this a complete sentence: '{text}'"}
                ],
                timeout=5  # 5 second timeout
            )
            
            # Get the response and ensure it's either 'true' or 'false'
            result = response.choices[0].message.content.lower().strip()
            
            # Validate the response
            if result not in ['true', 'false']:
                print(f"Invalid response from OpenAI: {result}")
                return False
            
            return result == 'true'
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return False  # Default to false on error
