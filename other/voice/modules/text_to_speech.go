package modules

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	tts "cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
	"github.com/faiface/beep"
	"github.com/faiface/beep/mp3"
	"github.com/faiface/beep/speaker"
)

// getContextAndCompletion retrieves context index and completion status from OpenAI
func (at *AudioTranscriber) getContextAndCompletion(text string) (int, bool) {
	ctx := context.Background()
	req := openai.ChatCompletionRequest{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessage{
			{
				Role: "system",
				Content: `Analyze the given text and provide TWO pieces of information:
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

2. Determine if the input is complete (true/false):
ALWAYS mark as complete (true) if ANY of these are present:
- Contains "?" (e.g., "How are you?", "What's up?")
- Ends with "." (e.g., "That's great.", "I understand.")
- Ends with "!" (e.g., "Hello!", "Great!")
- Is a greeting (e.g., "Hi", "Hello", "Hey")

Mark as incomplete (false) ONLY if ALL of these are true:
- Does NOT end with "?", ".", or "!"
- Ends with comma or no punctuation
- Is an unfinished thought

Return ONLY in this format:
{"index": X, "complete": true/false}`,
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("Analyze this text: '%s'", text),
			},
		},
	}

	resp, err := at.openaiClient.CreateChatCompletion(ctx, req)
	if err != nil {
		log.Printf("Error analyzing text: %v", err)
		return 0, false
	}

	if len(resp.Choices) == 0 {
		log.Println("No choices returned from OpenAI")
		return 0, false
	}

	content := resp.Choices[0].Message.Content
	content = strings.ReplaceAll(content, "'", "\"")
	custom := make(map[string]interface{})
	err = json.Unmarshal([]byte(content), &custom)
	if err != nil {
		log.Printf("Error parsing response: %v", err)
		return 0, false
	}

	index, ok1 := custom["index"].(float64)
	complete, ok2 := custom["complete"].(bool)
	if !ok1 || !ok2 {
		log.Println("Invalid response format")
		return 0, false
	}

	return int(index), complete
}

// getAIResponse retrieves a response from OpenAI based on the input text using streaming
func (at *AudioTranscriber) getAIResponse(text string) string {
	ctx := context.Background()
	req := openai.ChatCompletionRequest{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessage{
			{
				Role: "system",
				Content: `You are Quad, an AI-powered online teacher dedicated to making learning fun and engaging.

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

Remember: Your speaking responses will be converted to speech, so keep sentences clear and well-paced.`,
			},
			{
				Role:    "user",
				Content: text,
			},
		},
		Stream: true, // Enable streaming
	}

	stream, err := at.openaiClient.CreateChatCompletionStream(ctx, req)
	if err != nil {
		log.Printf("OpenAI API error: %v", err)
		return ""
	}
	defer stream.Close()

	var responseBuilder strings.Builder

	// Process the streamed response
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("Stream error: %v", err)
			break
		}

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			if delta.Content != "" {
				responseBuilder.WriteString(delta.Content)
				fmt.Print(delta.Content) // Print each token as it arrives
			}
		}
	}

	return responseBuilder.String()
}

// textToSpeech converts text to speech using Google's TTS service
func (at *AudioTranscriber) textToSpeech(text string) ([]byte, error) {
	ctx := context.Background()
	req := &tts.SynthesizeSpeechRequest{
		Input: &tts.SynthesisInput{
			InputSource: &tts.SynthesisInput_Text{
				Text: text,
			},
		},
		Voice:       at.voice,
		AudioConfig: at.audioConfig,
	}

	resp, err := at.ttsClient.SynthesizeSpeech(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("text-to-speech error: %v", err)
	}

	return resp.AudioContent, nil
}

// streamAudioResponse streams the audio response and manages state
func (at *AudioTranscriber) streamAudioResponse(audioData []byte) error {
	at.setAudioPlaying(true)
	defer func() {
		at.aggressiveClearTranscripts()
		at.setAudioPlaying(false)
		at.setListeningState(FULL_LISTENING)
		at.aggressiveClearTranscripts()
		log.Println("Audio playback completed or interrupted")
		fmt.Printf("\nListening .... (press ` or say \"shut up\" to interrupt)\n")
	}()
	
	log.Printf("Starting audio response playback (length: %d bytes)", len(audioData))
	
	reader := bytes.NewReader(audioData)
	streamer, format, err := mp3.Decode(io.NopCloser(reader))
	if err != nil {
		return fmt.Errorf("failed to decode audio: %v", err)
	}
	defer streamer.Close()
	
	err = speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))
	if err != nil {
		return fmt.Errorf("failed to initialize speaker: %v", err)
	}
	
	done := make(chan bool)
	interrupted := make(chan bool)
	
	// Play audio with interrupt handling
	speaker.Play(beep.Seq(streamer, beep.Callback(func() {
		select {
		case <-interrupted:
			return
		default:
			done <- true
		}
	})))
	
	// Wait for completion or interruption
	select {
	case <-done:
		log.Println("Audio playback completed normally")
		return nil
	case <-interrupted:
		log.Println("Audio playback was interrupted")
		speaker.Clear()
		return nil
	case <-time.After(60 * time.Second):
		return fmt.Errorf("audio playback timed out")
	}
}

// aggressiveClearTranscripts clears all transcript variables
func (at *AudioTranscriber) aggressiveClearTranscripts() {
	log.Println("Starting aggressive transcript clearing")
	
	// Multiple clearing passes with delays
	for i := 0; i < 5; i++ {
		at.mu.Lock()
		at.InterruptTranscript = ""
		at.InterimResult = ""
		at.FinalResult = ""
		at.LastTranscript = ""
		at.CurrentSentence = ""
		at.mu.Unlock()
		
		log.Printf("Clearing pass %d/3 completed", i+1)
		// Add delay between passes
		time.Sleep(100 * time.Millisecond)
	}
	
	// Reset processing flags
	at.audioStateMutex.Lock()
	at.isProcessingAudio = false
	at.audioStateMutex.Unlock()
	
	// Clear audio queue
	at.clearAudioQueue()
	
	log.Println("Transcript clearing completed")
	// Final delay to ensure everything is cleared
	time.Sleep(100 * time.Millisecond)
}
