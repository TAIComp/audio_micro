package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	speech "cloud.google.com/go/speech/apiv1"
	speechpb "cloud.google.com/go/speech/apiv1/speechpb"
	texttospeech "cloud.google.com/go/texttospeech/apiv1"
	tts "cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
	"github.com/eiannone/keyboard"
	"github.com/faiface/beep"
	"github.com/faiface/beep/mp3"
	"github.com/faiface/beep/speaker"
	"github.com/gordonklaus/portaudio"
	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/api/option"
)

// ListeningState represents the state of audio listening
type ListeningState int32

const (
	FULL_LISTENING ListeningState = iota
	INTERRUPT_ONLY
	interruptKeyChar = '`'
)

// AudioTranscriber handles audio transcription and response
type AudioTranscriber struct {
	// Audio parameters
	Rate            int
	Chunk           int
	CurrentSentence string
	LastTranscript  string
	InterruptTranscript string // New field for interrupt detection
	InterimResult   string   // New field for interim result
	FinalResult     string   // New field for final result
	// Thread-safe queue for audio data
	audioQueue chan []byte

	// Google Speech client
	speechClient *speech.Client
	streamConfig *speechpb.StreamingRecognitionConfig

	// OpenAI client
	openaiClient *openai.Client
	Model        string

	// Google Text-to-Speech client
	ttsClient   *texttospeech.Client
	voice       *tts.VoiceSelectionParams
	audioConfig *tts.AudioConfig

	// PortAudio stream
	audioStream *portaudio.Stream

	// Listening state
	listeningState ListeningState

	// Interrupt commands
	interruptCommands map[string]struct{}

	// Flags
	audioPlaying bool
	audioPlayingMutex sync.Mutex

	// Transcript tracking
	lastInterimTimestamp time.Time
	interimCooldown      time.Duration

	// Interrupt handling
	lastInterruptTime time.Time
	interruptCooldown time.Duration

	// Paths
	interruptAudioPath string
	contextFolder      string
	randomFolder       string

	// Audio mappings
	contextFileMapping map[int]string
	randomFileNames    []string
	randomAudioFiles   []string
	usedRandomFiles    map[string]struct{}

	// Mic monitoring settings (Placeholder for future implementation)
	micVolume          float64
	noiseGateThreshold float64

	// Response generation
	stopGeneration  bool
	pendingResponse string

	// Synchronization
	mu sync.Mutex

	// Add new fields for better control
	muteThreshold    float64
	audioInputBuffer []float32
	processingLock   sync.Mutex
	interruptChan    chan struct{}

	// Audio state management
	isProcessingAudio bool
	audioStateMutex   sync.Mutex

	// Channel for coordinating audio playback
	audioPlaybackDone chan struct{}

	// Audio state management
	audioState struct {
		sync.Mutex
		lastSpeechTime    time.Time
		lastProcessedText string
		isProcessing      bool
		feedbackBuffer    []string // Store recent responses to prevent feedback
	}

	// Audio processing control
	processingControl struct {
		sync.Mutex
		noiseGate       float64
		volumeThreshold float64
		cooldownPeriod  time.Duration
	}

	// Add a reference to the cancellation function
	cancelFunc context.CancelFunc
}

// NewAudioTranscriber initializes the AudioTranscriber
func NewAudioTranscriber() *AudioTranscriber {
	// Load environment variables from .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file")
	}

	// Initialize OpenAI client
	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
	if openaiAPIKey == "" {
		log.Fatal("OPENAI_API_KEY not set")
	}
	openaiClient := openai.NewClient(openaiAPIKey)

	// Get Google credentials path from environment
	googleCredPath := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if googleCredPath == "" {
		log.Fatal("GOOGLE_APPLICATION_CREDENTIALS not set")
	}

	// Initialize Google Speech client
	ctx := context.Background()
	speechClient, err := speech.NewClient(ctx, option.WithCredentialsFile(googleCredPath))
	if err != nil {
		log.Fatalf("Failed to create speech client: %v", err)
	}

	// Initialize Google Text-to-Speech client
	ttsClient, err := texttospeech.NewClient(ctx, option.WithCredentialsFile(googleCredPath))
	if err != nil {
		log.Fatalf("Failed to create TTS client: %v", err)
	}
	// Create a context with cancellation for the transcriber
	_, transcriberCancel := context.WithCancel(context.Background())

	transcriber := &AudioTranscriber{
		Rate:         16000,
		Chunk:        800, // 16000 / 20
		audioQueue:   make(chan []byte, 100),
		speechClient: speechClient,
		streamConfig: &speechpb.StreamingRecognitionConfig{
			Config: &speechpb.RecognitionConfig{
				Encoding:                   speechpb.RecognitionConfig_LINEAR16,
				SampleRateHertz:            int32(16000),
				LanguageCode:               "en-US",
				EnableAutomaticPunctuation: true,
			},
			InterimResults: true,
		},
		openaiClient: openaiClient,
		Model:        "gpt-4o-mini", // Ensure this model exists or adjust accordingly
		ttsClient:    ttsClient,
		voice: &tts.VoiceSelectionParams{
			LanguageCode: "en-US",
			Name:         "en-US-Casual-K",
			SsmlGender:   tts.SsmlVoiceGender_MALE,
		},
		audioConfig: &tts.AudioConfig{
			AudioEncoding: tts.AudioEncoding_MP3,
			SpeakingRate:  1.0,
			Pitch:         0.0,
		},
		listeningState: FULL_LISTENING,
		interruptCommands: map[string]struct{}{
			"stop":             {},
			"end":              {},
			"shut up":          {},
			"please stop":      {},
			"stop please":      {},
			"please end":       {},
			"end please":       {},
			"shut up please":   {},
			"please shut up":   {},
			"okay stop":        {},
			"ok stop":          {},
			"can you stop":     {},
			"could you stop":   {},
			"would you stop":   {},
			"can you be quiet": {},
			"silence":          {},
			"pause":            {},
		},
		lastInterimTimestamp: time.Now(),
		interimCooldown:      500 * time.Millisecond,
		lastInterruptTime:    time.Now(),
		interruptCooldown:    2 * time.Second,
		interruptAudioPath:   "interruption.mp3",
		contextFileMapping: map[int]string{
			0:  "Hmm_let_me_think_about_that",
			1:  "So_basically",
			2:  "Umm_okay_so",
			3:  "Let_me_figure_this_out",
			4:  "Hmm_that's_a_good_one",
			5:  "Alright_let's_see",
			6:  "Let's_think_about_this_for_a_moment",
			7:  "Ooooh_that's_tricky",
			8:  "Hmm_give_me_a_sec",
			9:  "So_one_moment_um",
			10: "Oh_okay_okay",
			11: "Aha_just_a_sec",
			12: "Alright_let's_dive_into_that",
			13: "Okay_okay_let's_see",
			14: "Hmm_this_is_interesting",
			15: "Okay_okay_let's_get_to_work",
			16: "So_yeah",
			17: "Uhh_well",
			18: "You_know",
			19: "So_anyway",
			20: "Alright_umm",
			21: "Oh_well_hmm",
			22: "Well_you_see",
			23: "So_basically_yeah",
			24: "Umm_anyway",
			25: "It's_uh_kinda_like",
		},
		randomFileNames: []string{
			"Hmm_let_me_think.mp3",
			"So.mp3",
			"Umm_well_well_well.mp3",
			"You_know.mp3",
		},
		randomAudioFiles:   []string{},
		usedRandomFiles:    make(map[string]struct{}),
		micVolume:          0.5,
		noiseGateThreshold: 0.5,
		stopGeneration:     false,
		pendingResponse:    "",
		muteThreshold:      0.01,
		interruptChan:      make(chan struct{}, 1),
		audioPlaybackDone:  make(chan struct{}),
		processingControl: struct {
			sync.Mutex
			noiseGate       float64
			volumeThreshold float64
			cooldownPeriod  time.Duration
		}{
			noiseGate:       0.02,
			volumeThreshold: 0.1,
			cooldownPeriod:  time.Second,
		},
		cancelFunc: transcriberCancel,
	}

	// Define audio folders
	transcriber.contextFolder = filepath.Join("voiceCashingSys", "contextBased")
	transcriber.randomFolder = filepath.Join("voiceCashingSys", "random")

	// Load random audio files
	for _, filename := range transcriber.randomFileNames {
		filePath := filepath.Join(transcriber.randomFolder, filename)
		if _, err := os.Stat(filePath); err == nil {
			transcriber.randomAudioFiles = append(transcriber.randomAudioFiles, filePath)
		}
	}

	// Check if interrupt audio file exists
	if _, err := os.Stat(transcriber.interruptAudioPath); os.IsNotExist(err) {
		fmt.Printf("Warning: Interrupt audio file '%s' not found!\n", transcriber.interruptAudioPath)
	}

	// Add this to NewAudioTranscriber
	for idx, filename := range transcriber.contextFileMapping {
		path := filepath.Join(transcriber.contextFolder, filename+".mp3")
		if _, err := os.Stat(path); os.IsNotExist(err) {
			fmt.Printf("Warning: Context audio file missing: %d - %s\n", idx, path)
		}
	}

	// Initialize the speaker if it hasn't been initialized yet
	if err := transcriber.initSpeaker(); err != nil {
		log.Printf("Warning: Failed to initialize speaker: %v", err)
	}

	// Initialize audio state
	transcriber.audioState.lastSpeechTime = time.Now()
	transcriber.audioState.feedbackBuffer = make([]string, 0, 10)

	return transcriber
}

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

2. Whether the sentence requires a response, considering:
- Is it a greeting or question that expects a response?
- Does it contain a question mark?
- Is it seeking information or engagement?
- Is it a statement that doesn't require elaboration?

Return ONLY in this format:
{"index": X, "complete": false if response needed, true if no response needed}`,
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
	// Parse the response; assuming JSON-like format
	content = strings.ReplaceAll(content, "'", "\"")
	custom := make(map[string]interface{})
	err = json.Unmarshal([]byte(content), &custom)
	if err != nil {
		log.Printf("Error parsing response: %v", err)
		return 0, false
	}

	index, ok1 := custom["index"].(float64) // JSON numbers are float64
	complete, ok2 := custom["complete"].(bool)
	if !ok1 || !ok2 {
		log.Println("Invalid response format")
		return 0, false
	}

	// For greetings and questions, we want to generate a response
	if strings.Contains(strings.ToLower(text), "hello") || 
	   strings.Contains(strings.ToLower(text), "hi") || 
	   strings.Contains(text, "?") {
		complete = false
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

// Helper function for audio data conversion
func convertAudioData(data []int) []byte {
	dataBytes := make([]byte, len(data)*2)
	for i, sample := range data {
		sample16 := int16(sample)
		dataBytes[i*2] = byte(sample16)
		dataBytes[i*2+1] = byte(sample16 >> 8)
	}
	return dataBytes
}

// Text to speech conversion
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

// processAudioStream manages the streaming recognition process
func (at *AudioTranscriber) processAudioStream(ctx context.Context) error {
	var currentStream speechpb.Speech_StreamingRecognizeClient
	var streamMutex sync.Mutex
	var err error

	// Function to create a new stream
	createNewStream := func() error {
		streamMutex.Lock()
		defer streamMutex.Unlock()

		// Only close the current stream if it exists
		if currentStream != nil {
			if err := currentStream.CloseSend(); err != nil {
				log.Printf("Error closing stream: %v", err)
			}
			// Wait a bit for cleanup
			time.Sleep(100 * time.Millisecond)
		}

		currentStream, err = at.speechClient.StreamingRecognize(ctx)
		if err != nil {
			return fmt.Errorf("could not create stream: %v", err)
		}

		// Send the initial configuration
		if err := currentStream.Send(&speechpb.StreamingRecognizeRequest{
			StreamingRequest: &speechpb.StreamingRecognizeRequest_StreamingConfig{
				StreamingConfig: at.streamConfig,
			},
		}); err != nil {
			return fmt.Errorf("could not send config: %v", err)
		}

		return nil
	}

	// Create initial stream
	if err := createNewStream(); err != nil {
		return err
	}

	// Create error channel for error handling
	errChan := make(chan error, 1)

	// Start goroutine for handling audio data
	go func() {
		for {
			select {
			case <-ctx.Done():
				errChan <- nil
				return
			case data, ok := <-at.audioQueue:
				if !ok {
					errChan <- nil
					return
				}

				streamMutex.Lock()
				stream := currentStream
				streamMutex.Unlock()

				if stream == nil {
					continue
				}

				// Always send audio data regardless of state
				if err := stream.Send(&speechpb.StreamingRecognizeRequest{
					StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{
						AudioContent: data,
					},
				}); err != nil {
					log.Printf("Error sending audio data: %v", err)
					// Try to recreate stream on error
					if err := createNewStream(); err != nil {
						errChan <- err
						return
					}
				}
			}
		}
	}()

	// Start goroutine for receiving responses
	go func() {
		for {
			streamMutex.Lock()
			stream := currentStream
			streamMutex.Unlock()

			if stream == nil {
				time.Sleep(100 * time.Millisecond)
				continue
			}

			resp, err := stream.Recv()
			if err == io.EOF {
				// Try to recreate stream on EOF
				if err := createNewStream(); err != nil {
					errChan <- err
					return
				}
				continue
			}
			if ctx.Err() != nil {
				errChan <- nil
				return
			}
			if err != nil {
				log.Printf("Error receiving from stream: %v", err)
				// Try to recreate stream on error
				if err := createNewStream(); err != nil {
					errChan <- err
					return
				}
				continue
			}

			// Get current state before processing
			currentState := at.getListeningState()
			audioPlaying := at.isAudioPlaying()  
			fmt.Printf("\rCurrent State: %v, Audio Playing: %v\n", currentState, audioPlaying)

			// Process responses based on the current state
			at.processResponses([]*speechpb.StreamingRecognizeResponse{resp})
		}
	}()

	// Start goroutine to monitor state changes
	go func() {
		var lastState ListeningState
		var lastAudioPlaying bool
		var lastStateChange time.Time = time.Now()

		for {
			select {
			case <-ctx.Done():
				return
			default:
				currentState := at.getListeningState()
				audioPlaying := at.isAudioPlaying()

				// Only create new stream if state has changed and enough time has passed
				if (currentState != lastState || audioPlaying != lastAudioPlaying) &&
					time.Since(lastStateChange) >= 500*time.Millisecond {
					// State has changed, create new stream
					if err := createNewStream(); err != nil {
						errChan <- err
						return
					}

					lastState = currentState
					lastAudioPlaying = audioPlaying
					lastStateChange = time.Now()
				}

				time.Sleep(200 * time.Millisecond)
			}
		}
	}()

	return <-errChan
}

// processResponses processes incoming audio responses based on current state
func (at *AudioTranscriber) processResponses(responses []*speechpb.StreamingRecognizeResponse) {
    if len(responses) == 0 {
        return
    }

    // Get current state and audio status before processing any responses
    at.mu.Lock()
    currentState := at.getListeningState()
    isPlaying := at.isAudioPlaying()
    at.mu.Unlock()

    // Debug output
    fmt.Printf("Processing responses - State: %d, Audio Playing: %v\n", currentState, isPlaying)

    for _, response := range responses {
        if response.Results == nil || len(response.Results) == 0 {
            continue
        }

        result := response.Results[0]
        transcript := strings.TrimSpace(result.Alternatives[0].Transcript)
        if transcript == "" {
            continue
        }

        // Re-check state for each result as it might have changed
        at.mu.Lock()
        currentState = at.getListeningState()
        isPlaying = at.isAudioPlaying()
        at.mu.Unlock()

        switch currentState {
        case INTERRUPT_ONLY:
            if isPlaying {
                // Update interrupt transcript and check for commands regardless of final/interim
                at.InterruptTranscript = transcript
                fmt.Printf("Interrupt Detection: %s\n", transcript)
                if at.containsInterruptCommand(transcript) {
                    at.handleVoiceInterrupt()
                }
            }

        case FULL_LISTENING:
            if !isPlaying {
                if result.IsFinal {
                    fmt.Printf("Full Listening - Final: %s\n", transcript)
                    at.handleFinalResult(transcript)
                } else {
                    fmt.Printf("Full Listening - Interim: %s\n", transcript)
                    at.handleInterimResult(transcript)
                }
            }
        }
    }
}

// Play audio response

func (at *AudioTranscriber) playAudioResponse(audioPath string) {
	// Set audio playing state
	at.setAudioPlaying(true)

	// Open the audio file
	f, err := os.Open(audioPath)
	if err != nil {
		log.Printf("Error opening audio file: %v", err)
		return
	}
	defer f.Close()

	// Decode the MP3 file
	streamer, format, err := mp3.Decode(f)
	if err != nil {
		log.Printf("Error decoding MP3: %v", err)
		return
	}
	defer streamer.Close()

	// Initialize the speaker if it hasn't been initialized yet
	if err := speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10)); err != nil {
		log.Printf("Error initializing speaker: %v", err)
	}

	// Create a channel to signal when playback is done
	done := make(chan bool)

	// Play the audio
	speaker.Play(beep.Seq(streamer, beep.Callback(func() {
		done <- true
	})))

	// Wait for playback to finish
	<-done
}

// handleKeyboardInterrupt handles the backtick key interrupt
func (at *AudioTranscriber) handleKeyboardInterrupt() {
	fmt.Println("\nInterrupt: Backtick keyboard detected!")
	at.handleInterrupt()
}

// handleInterrupt handles interrupt logic
func (at *AudioTranscriber) handleInterrupt() {
    // Only handle interrupts if we're in INTERRUPT_ONLY mode and playing audio
    if at.getListeningState() != INTERRUPT_ONLY {
        return
    }

    // Stop any ongoing audio
    at.stopAudio()
    
    // Play acknowledgment sound
    at.playAcknowledgment()
    
    // Reset state back to FULL_LISTENING
    at.setListeningState(FULL_LISTENING)
    at.setAudioPlaying(false)
    at.LastTranscript = ""
    at.CurrentSentence = ""
    
    // Clear any pending audio data
    at.clearAudioQueue()
    
    fmt.Println("\nListening .... (press ` or say \"shut up\" to interrupt)")
}

// playAcknowledgment plays the interruption acknowledgment audio
func (at *AudioTranscriber) playAcknowledgment() {
    // Set state before playing
    at.setListeningState(INTERRUPT_ONLY)
    at.setAudioPlaying(true)
    
    // Play the acknowledgment audio
    at.playAudioResponse(at.interruptAudioPath)
}

// playNextRandomAudio plays the next random audio file
func (at *AudioTranscriber) playNextRandomAudio() bool {
    // Set state before playing
    at.setListeningState(INTERRUPT_ONLY)
    at.setAudioPlaying(true)

    // Get a random audio file that hasn't been used
    for _, name := range at.randomFileNames {
        if _, used := at.usedRandomFiles[name]; !used {
            audioFile := fmt.Sprintf("%s/%s", at.randomFolder, name)
            fmt.Printf("\nPlaying context audio: %s\n", name)

            // Play the audio file
            at.playAudioResponse(audioFile)

            // Mark this file as used
            at.usedRandomFiles[name] = struct{}{}
            return true
        }
    }

    // Reset used files if all have been played
    at.usedRandomFiles = make(map[string]struct{})
    return false
}

// resetState resets the transcription state
func (at *AudioTranscriber) resetState() {
	at.mu.Lock()
	defer at.mu.Unlock()

	at.CurrentSentence = ""
	at.LastTranscript = ""
	at.lastInterimTimestamp = time.Now()
	at.lastInterruptTime = time.Now() // Reset interrupt timer too
	at.InterruptTranscript = ""
	at.InterimResult = ""
	at.FinalResult = ""

	// Reset audio state
	at.audioState.Lock()
	at.audioState.lastProcessedText = ""
	at.audioState.isProcessing = false
	at.audioState.feedbackBuffer = make([]string, 0, 10)
	at.audioState.Unlock()

	// Clear the audio queue
	at.clearAudioQueue()

}

// processCompleteSentence handles the processing of a complete sentence
func (at *AudioTranscriber) processCompleteSentence(sentence string, initialAudioIndex int) {
    // Get context and completion status
    contextIndex, isComplete := at.getContextAndCompletion(sentence)
    
    // Set state to INTERRUPT_ONLY and mark audio as playing
    at.setListeningState(INTERRUPT_ONLY)
    at.setAudioPlaying(true)
    
    // Clear any existing audio queue
    at.clearAudioQueue()

    // Play context-based audio if available
    if contextIndex >= 0 {
        if played := at.playContextAudio(contextIndex); !played {
            log.Printf("Failed to play context audio for index: %d\n", contextIndex)
        }
    }

    // If not complete, get AI response and play it
    if !isComplete {
        // Get AI response
        response := at.getAIResponse(sentence)
        if response != "" {
            // Convert to speech
            audioData, err := at.textToSpeech(response)
            if err == nil {
                // Stream the audio response
                if err := at.streamAudioResponse(audioData); err != nil {
                    log.Printf("Error streaming audio response: %v\n", err)
                }
            } else {
                log.Printf("Error converting text to speech: %v\n", err)
            }
        }
    }

    // After all audio is done playing, reset state to FULL_LISTENING
    // But only if we're still in INTERRUPT_ONLY state (no interrupts occurred)
	at.setListeningState(FULL_LISTENING)
	at.setAudioPlaying(false)
	at.CurrentSentence = ""
	at.LastTranscript = ""
	at.InterimResult = ""
	at.FinalResult = ""
	fmt.Println("\nListening .... (press ` or say \"shut up\" to interrupt)")
}

// streamAudioResponse streams the audio response and resets state
func (at *AudioTranscriber) streamAudioResponse(audioData []byte) error {
    // Ensure proper state transition with mutex lock
    at.mu.Lock()
    at.setListeningState(INTERRUPT_ONLY)
    at.setAudioPlaying(true)
    at.CurrentSentence = ""
    at.LastTranscript = ""
    at.InterruptTranscript = ""  // Clear interrupt transcript
    at.lastInterimTimestamp = time.Now()
    at.pendingResponse = ""
    at.mu.Unlock()

    // Clear audio state
    at.audioState.Lock()
    at.audioState.lastProcessedText = ""
    at.audioState.isProcessing = false
    at.audioState.feedbackBuffer = make([]string, 0, 10)
    at.audioState.Unlock()

    // Create a new reader for the MP3 data
    reader := bytes.NewReader(audioData)

    // Decode the MP3
    streamer, format, err := mp3.Decode(io.NopCloser(reader))
    if err != nil {
        at.mu.Lock()
        at.setListeningState(FULL_LISTENING)
        at.setAudioPlaying(false)
        at.mu.Unlock()
        return fmt.Errorf("error decoding MP3: %v", err)
    }
    defer streamer.Close()

    // Initialize speaker if needed
    speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))

    // Create done channel
    done := make(chan bool)

    // Play the audio
    speaker.Play(beep.Seq(streamer, beep.Callback(func() {
        done <- true
    })))

    // Wait for playback to finish
    <-done

    // Add a small delay after playback
    time.Sleep(300 * time.Millisecond)

    // Reset states after playback with proper locking
    at.mu.Lock()
    at.setListeningState(FULL_LISTENING)
    at.setAudioPlaying(false)
    at.CurrentSentence = ""
    at.LastTranscript = ""
    at.InterruptTranscript = ""
    at.InterimResult = ""  // Clear interim result
    at.FinalResult = ""    // Clear final result
    at.lastInterimTimestamp = time.Now()
    
    // Reset audio state
    at.audioState.Lock()
    at.audioState.lastProcessedText = ""
    at.audioState.isProcessing = false
    at.audioState.feedbackBuffer = make([]string, 0, 10)
    at.audioState.Unlock()
    
    // Clear the audio queue
    at.clearAudioQueue()
    at.mu.Unlock()

    // Call resetState to ensure all states are cleared
    at.resetState()

    return nil
}

// StartProcessing starts the audio processing with the given context
func (at *AudioTranscriber) StartProcessing(ctx context.Context) {
	go func() {
		for {
			if err := at.processAudioStream(ctx); err != nil {
				if err != context.Canceled {
					log.Printf("Error in audio stream: %v", err)
					// Wait before retrying to prevent tight loop
					time.Sleep(500 * time.Millisecond)
				} else {
					return
				}
			}

			// Check if context is still active
			select {
			case <-ctx.Done():
				return
			default:
			}
		}
	}()
}

// Add these methods to AudioTranscriber struct
func (at *AudioTranscriber) setListeningState(state ListeningState) {
	atomic.StoreInt32((*int32)(&at.listeningState), int32(state))
	// Log state change for debugging
	log.Printf("Listening state changed to: %v", state)
}

func (at *AudioTranscriber) getListeningState() ListeningState {
	return ListeningState(atomic.LoadInt32((*int32)(&at.listeningState)))
}

func (at *AudioTranscriber) clearAudioQueue() {
	for len(at.audioQueue) > 0 {
		<-at.audioQueue
	}
}

// Play context audio
func (at *AudioTranscriber) playContextAudio(index int) bool {
    // Set state before playing
    at.setListeningState(INTERRUPT_ONLY)
    at.setAudioPlaying(true)

    // Get the mapped filename for this index
    filename, exists := at.contextFileMapping[index]
    if !exists {
        log.Printf("No mapping found for context index: %d", index)
        return false
    }

    // Construct the full path with the mapped filename
    audioFile := filepath.Join(at.contextFolder, filename+".mp3")
    fmt.Printf("Playing context audio: %s\n", audioFile)
    
    if _, err := os.Stat(audioFile); os.IsNotExist(err) {
        log.Printf("Context audio file not found: %s", audioFile)
        return false
    }

    // Play the audio - don't change states here since they're managed by processCompleteSentence
    at.playAudioResponse(audioFile)
    return true
}

// stopAudio interrupts any ongoing audio playback
func (at *AudioTranscriber) stopAudio() {
	speaker.Clear()
}

func init() {
	// Open /dev/null for writing
	null, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		log.Fatal(err)
	}

	// Duplicate the file descriptor
	if err = syscall.Dup2(int(null.Fd()), int(os.Stderr.Fd())); err != nil {
		log.Fatal(err)
	}
}

func main() {
	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		log.Fatal(err)
	}
	defer portaudio.Terminate()

	// Create a main context that is not canceled by the transcriber
	mainCtx, mainCancel := context.WithCancel(context.Background())
	defer mainCancel()

	// Create transcriber without passing the main cancel function
	transcriber := NewAudioTranscriber()

	// Initialize audio input stream
	inputChannels := 1
	outputChannels := 0
	sampleRate := 16000
	framesPerBuffer := make([]int16, 800)

	// Create a channel for stream recreation
	streamRestart := make(chan struct{}, 1)

	// Function to create and start a new audio stream
	createStream := func() (*portaudio.Stream, error) {
		stream, err := portaudio.OpenDefaultStream(
			inputChannels,
			outputChannels,
			float64(sampleRate),
			len(framesPerBuffer),
			framesPerBuffer,
		)
		if err != nil {
			return nil, fmt.Errorf("error opening stream: %v", err)
		}

		if err := stream.Start(); err != nil {
			stream.Close()
			return nil, fmt.Errorf("error starting stream: %v", err)
		}

		return stream, nil
	}

	// Create initial stream
	stream, err := createStream()
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Starting audio transcription... Speak into your microphone.")
	fmt.Println("\nListening .... (press ` or say \"shut up\" to interrupt)")

	// Handle OS interrupts
	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		log.Println("\nTranscription stopped by user")
		mainCancel()
	}()

	// Start handling keyboard interrupts
	go transcriber.handleInterrupts()

	// Start audio capture goroutine
	go func() {
		for {
			select {
			case <-mainCtx.Done():
				return
			case <-streamRestart:
				// Clean up old stream
				if stream != nil {
					stream.Stop()
					stream.Close()
				}

				// Create new stream
				var err error
				stream, err = createStream()
				if err != nil {
					log.Printf("Error recreating stream: %v", err)
					continue
				}
			default:
				err := stream.Read()
				if err != nil {
					log.Printf("Error reading from stream: %v", err)
					streamRestart <- struct{}{} // Trigger stream recreation
					continue
				}

				// Convert samples to bytes
				audioData := make([]byte, len(framesPerBuffer)*2)
				for i, sample := range framesPerBuffer {
					audioData[i*2] = byte(sample)
					audioData[i*2+1] = byte(sample >> 8)
				}

				// Send audio data to the queue
				select {
				case transcriber.audioQueue <- audioData:
				default:
					// Queue is full, skip this chunk
				}
			}
		}
	}()

	// Start the audio processing in a goroutine with the main context
	go transcriber.StartProcessing(mainCtx)

	// Keep the main function running
	select {}
}

// handleInterrupts listens for keyboard events using the keyboard library
func (at *AudioTranscriber) handleInterrupts() {
	// Open the keyboard
	if err := keyboard.Open(); err != nil {
		log.Fatalf("Failed to open keyboard: %v", err)
	}
	defer keyboard.Close()

	log.Println("Press ` (backtick) to interrupt at any time.")

	for {
		char, key, err := keyboard.GetKey()
		if err != nil {
			log.Printf("Error getting key: %v", err)
			continue
		}

		if key == keyboard.KeyEsc {
			log.Println("Esc key pressed. Exiting...")
			at.cancelFunc() // Cancel the transcriber context
			return
		}

		if char == interruptKeyChar {
			log.Println("Backtick interrupt detected!")
			at.handleKeyboardInterrupt()
		}
	}
}

// containsInterruptCommand checks if the transcript contains an interrupt command
func (at *AudioTranscriber) containsInterruptCommand(transcript string) bool {
	at.mu.Lock()
	defer at.mu.Unlock()

	// Remove punctuation from the transcript
	re := regexp.MustCompile(`[^\w\s]`)
	cleanedText := re.ReplaceAllString(transcript, "")

	normalizedText := strings.ToLower(strings.TrimSpace(cleanedText))

	// Log the normalized transcript for debugging
	log.Printf("Normalized transcript: \"%s\"\n", normalizedText)

	// Check for exact matches
	if _, exists := at.interruptCommands[normalizedText]; exists {
		fmt.Printf("Interrupt command used: \"%s\"\n", normalizedText)
		return true
	}

	// Check if the transcript contains any of the interrupt commands
	for cmd := range at.interruptCommands {
		if strings.Contains(normalizedText, cmd) {
			fmt.Printf("Interrupt command used: \"%s\"\n", cmd)
			return true
		}
	}

	return false
}

// setAudioPlaying safely sets the audio playing state
func (at *AudioTranscriber) setAudioPlaying(playing bool) {
	at.audioPlayingMutex.Lock()
	defer at.audioPlayingMutex.Unlock()
	at.audioPlaying = playing
	if playing {
		log.Println("State Change: Audio Playing set to TRUE")
	} else {
		log.Println("State Change: Audio Playing set to FALSE")
	}
}

// isAudioPlaying safely gets the current audio playing state
func (at *AudioTranscriber) isAudioPlaying() bool {
	at.audioPlayingMutex.Lock()
	defer at.audioPlayingMutex.Unlock()
	return at.audioPlaying
}

// handleInterimResult processes interim transcription results
func (at *AudioTranscriber) handleInterimResult(transcript string) {
	// Skip processing if in INTERRUPT_ONLY mode or if playing audio
	if at.getListeningState() == INTERRUPT_ONLY || at.isAudioPlaying() {
		return
	}

	if transcript != at.LastTranscript && time.Since(at.lastInterimTimestamp) >= at.interimCooldown {
		fmt.Printf(`Interim: "%s"`+"\n", transcript)
		at.LastTranscript = transcript
		at.lastInterimTimestamp = time.Now()
	}
}

// handleFinalResult processes final transcription results
func (at *AudioTranscriber) handleFinalResult(transcript string) {
	// Skip if we're already processing audio or in INTERRUPT_ONLY state
	if at.getListeningState() == INTERRUPT_ONLY || at.isAudioPlaying() {
        return
    }

	// Process the final result
    at.CurrentSentence = transcript
	fmt.Printf(`Final: "%s"`+"\n", transcript)

	// Get context and completion status
	index, isComplete := at.getContextAndCompletion(transcript)

	if !at.isAudioPlaying() {
        // Start a timer in a separate goroutine
        go func() {
            startTime := time.Now()
            ticker := time.NewTicker(100 * time.Millisecond)
            defer ticker.Stop()

            maxWait := 2 * time.Second
            if isComplete {
                maxWait = 500 * time.Millisecond
            }

            for {
                select {
                case <-ticker.C:
                    elapsed := time.Since(startTime)
                    fmt.Printf("\rWaiting for more input: %.1f/%.1f seconds", elapsed.Seconds(), maxWait.Seconds())
    
                    if elapsed >= maxWait {
                        fmt.Println()
                        if isComplete {
                            fmt.Println("Processing: Complete sentence detected with 0.5 second silence")
                        } else {
                            fmt.Println("Processing: Incomplete sentence detected with 2 seconds silence")
                        }
                        at.processCompleteSentence(at.CurrentSentence, index)
                        return
                    }

                    if time.Since(at.lastInterimTimestamp) < elapsed {
                        fmt.Println("\rTimer reset - new audio detected")
                        return
                    }
                }
            }
        }()
    }
}

// handleVoiceInterrupt handles voice-based interrupt commands
func (at *AudioTranscriber) handleVoiceInterrupt() {
	fmt.Println("\nInterrupt: Voice command detected!")
	
	// Stop any ongoing audio playback immediately
	speaker.Clear()
	
	// Reset states before calling handleInterrupt
	at.setAudioPlaying(false)
	at.setListeningState(FULL_LISTENING)
	
	at.handleInterrupt()
}

// initSpeaker initializes the speaker with a sample audio file
func (at *AudioTranscriber) initSpeaker() error {
	// Open a sample audio file to get the format
	sampleFile := filepath.Join(at.contextFolder, at.contextFileMapping[0]+".mp3")
	f, err := os.Open(sampleFile)
	if err != nil {
		return fmt.Errorf("error opening sample file: %v", err)
	}
	defer f.Close()

	// Decode the sample file to get the format
	_, format, err := mp3.Decode(f)
	if err != nil {
		return fmt.Errorf("error decoding sample file: %v", err)
	}

	// Initialize the speaker
	err = speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))
	if err != nil {
		return fmt.Errorf("error initializing speaker: %v", err)
	}

	return nil
}
