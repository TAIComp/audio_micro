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
	isSpeaking bool

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
	isSpeakingMutex   sync.Mutex
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
		isSpeaking:           false,
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

2. Whether the sentence is complete, considering:
- Grammatical completeness (subject + predicate)
- Semantic completeness (complete thought/meaning)
- Natural ending point (proper punctuation or logical conclusion)
- Trailing indicators suggesting more is coming

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
	stream, err := at.speechClient.StreamingRecognize(ctx)
	if err != nil {
		return fmt.Errorf("could not create stream: %v", err)
	}
	defer stream.CloseSend()

	// Send the initial configuration
	if err := stream.Send(&speechpb.StreamingRecognizeRequest{
		StreamingRequest: &speechpb.StreamingRecognizeRequest_StreamingConfig{
			StreamingConfig: at.streamConfig,
		},
	}); err != nil {
		return fmt.Errorf("could not send config: %v", err)
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

				// Always send audio data regardless of state
				if err := stream.Send(&speechpb.StreamingRecognizeRequest{
					StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{
						AudioContent: data,
					},
				}); err != nil {
					errChan <- err
					return
				}
			}
		}
	}()

	// Start goroutine for receiving responses
	go func() {
		for {
			resp, err := stream.Recv()
			if err == io.EOF || ctx.Err() != nil {
				errChan <- nil
				return
			}
			if err != nil {
				errChan <- err
				return
			}

			// Get current state before processing
			currentState := at.getListeningState()

			// Process responses based on the current state
			if currentState == FULL_LISTENING {
				at.processResponseFullListening([]*speechpb.StreamingRecognizeResponse{resp})
			} else if currentState == INTERRUPT_ONLY {
				at.processResponseInterruptOnly([]*speechpb.StreamingRecognizeResponse{resp})
			}
		}
	}()

	return <-errChan
}

// Play audio response
func (at *AudioTranscriber) playAudioResponse(audioPath string) {
	// Set speaking state
	at.setSpeakingState(true)
	at.setListeningState(INTERRUPT_ONLY)

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

	// Reset states immediately after playback
	at.setSpeakingState(false)
	at.setListeningState(FULL_LISTENING)
	at.pendingResponse = "" // Clear the pending response
	at.resetState()         // Reset all states
	fmt.Println("\nListening .... (press ` or say \"shut up\" to interrupt)")
}

// processResponseFullListening processes responses in full listening mode
func (at *AudioTranscriber) processResponseFullListening(responses []*speechpb.StreamingRecognizeResponse) {
	// Skip if not in FULL_LISTENING mode
	if at.getListeningState() != FULL_LISTENING {
		return
	}

	for _, resp := range responses {
		for _, result := range resp.Results {
			if len(result.Alternatives) == 0 {
				continue
			}

			transcript := result.Alternatives[0].Transcript

			// Still check for interrupts even in full listening mode
			if at.containsInterruptCommand(transcript) {
				at.handleVoiceInterrupt()
				return
			}

			// Process based on result type
			if !result.IsFinal {
				at.handleInterimResult(transcript)
			} else {
				at.handleFinalResult(transcript)
			}
		}
	}
}

// processResponseInterruptOnly processes responses in interrupt only mode
func (at *AudioTranscriber) processResponseInterruptOnly(responses []*speechpb.StreamingRecognizeResponse) {
	// Skip if not in INTERRUPT_ONLY mode
	if at.getListeningState() != INTERRUPT_ONLY {
		return
	}

	for _, resp := range responses {
		for _, result := range resp.Results {
			if len(result.Alternatives) == 0 {
				continue
			}

			transcript := result.Alternatives[0].Transcript
			
			// Check for interrupt commands
			if at.containsInterruptCommand(transcript) {
				log.Printf("Interrupt command detected: %s", transcript)
				at.handleVoiceInterrupt()
				return
			}
		}
	}
	
	// Only show the listening message, don't process any results
	fmt.Printf("\rInterrupt Mode - Listening for commands...")
}

// handleKeyboardInterrupt handles the backtick key interrupt
func (at *AudioTranscriber) handleKeyboardInterrupt() {
	fmt.Println("\nInterrupt: Backtick keyboard detected!")
	at.handleInterrupt()
}

// handleInterrupt handles interrupt logic
func (at *AudioTranscriber) handleInterrupt() {
	at.mu.Lock()
	defer at.mu.Unlock()

	// Check if we're within the cooldown period
	if time.Since(at.lastInterruptTime) < at.interruptCooldown {
		fmt.Println("\nInterrupt ignored: Please wait before interrupting again")
		return
	}

	// Stop any ongoing audio playback and clear speaker
	speaker.Clear()
	at.stopAudio()

	// Reset all states and variables
	at.stopGeneration = true
	at.setSpeakingState(false)
	at.setListeningState(FULL_LISTENING)
	at.CurrentSentence = ""
	at.LastTranscript = ""
	at.lastInterimTimestamp = time.Now()
	at.lastInterruptTime = time.Now() // Reset interrupt timer too
	at.InterruptTranscript = ""

	// Reset audio state
	at.audioState.Lock()
	at.audioState.lastProcessedText = ""
	at.audioState.isProcessing = false
	at.audioState.feedbackBuffer = make([]string, 0, 10)
	at.audioState.Unlock()

	// Clear the audio queue
	at.clearAudioQueue()

	// Cancel existing stream if any
	if at.cancelFunc != nil {
		at.cancelFunc()
		at.cancelFunc = nil
	}

	// Play acknowledgment sound if available
	if _, err := os.Stat(at.interruptAudioPath); err == nil {
		at.playAcknowledgment()
	}

	// Add a small delay to ensure clean state
	time.Sleep(300 * time.Millisecond)

	// Create and start new processing context
	newCtx, newCancel := context.WithCancel(context.Background())
	at.cancelFunc = newCancel
	
	// Print listening message before restarting
	fmt.Println("\nListening .... (press ` or say \"shut up\" to interrupt)")

	// Start new processing goroutine
	go at.StartProcessing(newCtx)
}

// playAcknowledgment plays the interruption acknowledgment audio
func (at *AudioTranscriber) playAcknowledgment() {
	if _, err := os.Stat(at.interruptAudioPath); os.IsNotExist(err) {
		log.Printf("Interruption audio file not found: %s", at.interruptAudioPath)
		return
	}

	// Open the audio file
	f, err := os.Open(at.interruptAudioPath)
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

// playNextRandomAudio plays the next random audio file
func (at *AudioTranscriber) playNextRandomAudio() bool {
	if len(at.randomAudioFiles) == 0 {
		return false
	}

	// Reset used files if we've played them all
	if len(at.usedRandomFiles) == len(at.randomAudioFiles) {
		at.usedRandomFiles = make(map[string]struct{})
	}

	// Select an unused random file
	var availableFiles []string
	for _, file := range at.randomAudioFiles {
		if _, used := at.usedRandomFiles[file]; !used {
			availableFiles = append(availableFiles, file)
		}
	}

	if len(availableFiles) == 0 {
		return false
	}

	// Select a random file
	randomIndex := time.Now().UnixNano() % int64(len(availableFiles))
	audioFile := availableFiles[randomIndex]
	at.usedRandomFiles[audioFile] = struct{}{}

	// Play the random audio file
	file, err := os.Open(audioFile)
	if err != nil {
		log.Printf("Error playing random audio: %v", err)
		return false
	}
	defer file.Close()

	// Create a new PortAudio stream for playback
	buffer := make([]float32, 1024)
	outputStream, err := portaudio.OpenDefaultStream(0, 2, 44100, len(buffer), buffer)
	if err != nil {
		log.Printf("Error opening playback stream: %v", err)
		return false
	}
	defer outputStream.Close()

	if err := outputStream.Start(); err != nil {
		log.Printf("Error starting playback stream: %v", err)
		return false
	}
	defer outputStream.Stop()

	// Read and play the audio data
	readBuffer := make([]byte, 4096)
	samples := make([]float32, 2048) // stereo buffer
	for {
		n, err := file.Read(readBuffer)
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("Error reading audio file: %v", err)
			break
		}

		// Convert bytes to float32 samples (16-bit PCM to float32)
		numSamples := n / 2 // number of samples we can convert
		if numSamples > len(samples)/2 {
			numSamples = len(samples) / 2
		}

		for i := 0; i < numSamples; i++ {
			if i*2+1 >= n {
				break
			}
			sample := float32(int16(readBuffer[i*2])|int16(readBuffer[i*2+1])<<8) / 32768.0
			// Duplicate sample for stereo
			samples[i*2] = sample
			samples[i*2+1] = sample
		}

		// Clear the rest of the buffer if we didn't fill it completely
		for i := numSamples * 2; i < len(samples); i++ {
			samples[i] = 0
		}

		if err := outputStream.Write(); err != nil {
			log.Printf("Error playing audio: %v", err)
			break
		}

		// Add a small delay to prevent buffer overrun
		time.Sleep(10 * time.Millisecond)
	}

	at.setListeningState(INTERRUPT_ONLY)
	log.Printf("Playing random audio: %s", filepath.Base(audioFile))
	return true
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
	at.mu.Lock()
	at.stopGeneration = false
	at.setListeningState(INTERRUPT_ONLY) // Set state to INTERRUPT_ONLY before streaming
	at.mu.Unlock()

	// Generate AI response
	response := at.getAIResponse(sentence)
	if response == "" {
		return
	}

	// Print what AI wants to say
	fmt.Printf("\nAI wants to say: %s\n\n", response)

	// Store the response for filtering
	at.pendingResponse = response

	// Play context audio if not interrupted
	if !at.stopGeneration {
		contextFile := at.contextFileMapping[initialAudioIndex]
		fmt.Printf("Playing context audio: %s.mp3\n", contextFile)
		at.playContextAudio(initialAudioIndex)
	}

	// Convert and stream AI response if not interrupted
	if response != "" && !at.stopGeneration {
		fmt.Println("Streaming AI response audio...")
		audioData, err := at.textToSpeech(response)
		if err != nil {
			log.Printf("Error converting text to speech: %v", err)
			return
		}

		// Clear audio queue before streaming
		at.clearAudioQueue()

		if err := at.streamAudioResponse(audioData); err != nil {
			log.Printf("Error streaming audio: %v", err)
			return
		}
	}

	// Transition back to FULL_LISTENING state after streaming is complete
	at.mu.Lock()
	at.setListeningState(FULL_LISTENING)
	at.mu.Unlock()
	fmt.Println("\nListening .... (press ` or say \"shut up\" to interrupt)")
}

// clearAudioQueue clears any pending audio data in the queue
func (at *AudioTranscriber) clearAudioQueue() {
	for len(at.audioQueue) > 0 {
		<-at.audioQueue
	}
}

// Play context audio
func (at *AudioTranscriber) playContextAudio(index int) bool {
	if at.isSpeaking {
		return false
	}

	if filename, exists := at.contextFileMapping[index]; exists {
		contextFile := filepath.Join(at.contextFolder, filename+".mp3")
		if _, err := os.Stat(contextFile); os.IsNotExist(err) {
			log.Printf("Context audio file not found: %s", contextFile)
			return false
		}

		// Open the audio file
		f, err := os.Open(contextFile)
		if err != nil {
		log.Printf("Error opening context audio: %v", err)
			return false
		}
		defer f.Close()

		// Decode the MP3 file
		streamer, format, err := mp3.Decode(f)
		if err != nil {
			log.Printf("Error decoding MP3: %v", err)
			return false
		}
		defer streamer.Close()

		// Initialize the speaker if it hasn't been initialized yet
		speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))

		// Create a channel to signal when playback is done
		done := make(chan bool)

		// Play the audio
		speaker.Play(beep.Seq(streamer, beep.Callback(func() {
			done <- true
		})))

		// Wait for playback to finish
		<-done

		at.setListeningState(INTERRUPT_ONLY)
		return true
	}

	return false
}

// handleVoiceInterrupt handles voice-based interrupt commands
func (at *AudioTranscriber) handleVoiceInterrupt() {
	fmt.Println("\nInterrupt: Voice command detected!")
	
	// Stop any ongoing audio playback immediately
	speaker.Clear()
	
	// Reset states before calling handleInterrupt
	at.setSpeakingState(false)
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

// setSpeakingState safely sets the speaking state
func (at *AudioTranscriber) setSpeakingState(speaking bool) {
	at.isSpeakingMutex.Lock()
	defer at.isSpeakingMutex.Unlock()
	at.isSpeaking = speaking
	if speaking {
		log.Println("State Change: isSpeaking set to TRUE")
	} else {
		log.Println("State Change: isSpeaking set to FALSE")
	}
}

// isAudioFeedback checks if the transcript is part of AI-generated feedback
func (at *AudioTranscriber) isAudioFeedback(text string) bool {
	at.audioState.Lock()
	defer at.audioState.Unlock()

	// If we're in INTERRUPT_ONLY mode, treat all non-interrupt text as feedback
	if at.getListeningState() == INTERRUPT_ONLY {
		return true
	}

	// Check against recent responses
	for _, response := range at.audioState.feedbackBuffer {
		if strings.Contains(strings.ToLower(text), strings.ToLower(response)) {
			return true
		}
	}

	// Check cooldown period
	if time.Since(at.audioState.lastSpeechTime) < at.processingControl.cooldownPeriod {
		return true
	}

	return false
}

// handleInterimResult processes interim transcription results
func (at *AudioTranscriber) handleInterimResult(transcript string) {
	// Avoid processing completely during INTERRUPT_ONLY mode
	if at.getListeningState() != FULL_LISTENING {
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
	// Skip if this is a recent response
	if at.isAudioFeedback(transcript) {
		return
	}

	// Skip processing completely if not in FULL_LISTENING mode
	if at.getListeningState() != FULL_LISTENING {
		return
	}

	// Update current sentence for full listening mode
	at.audioState.Lock()
	if at.CurrentSentence == "" {
		at.CurrentSentence = transcript
	} else {
		newWords := strings.Fields(transcript)
		for _, word := range newWords {
			if !strings.Contains(at.CurrentSentence, word) {
				at.CurrentSentence += " " + word
			}
		}
	}
	at.audioState.Unlock()

	// Get context and completion status
	index, isComplete := at.getContextAndCompletion(at.CurrentSentence)
	fmt.Printf(`Final: "%s" (Complete: %t, Audio Index: %d)`+"\n", at.CurrentSentence, isComplete, index)

	// Only start timer if not currently speaking
	if !at.isSpeaking {
		// Start a timer in a separate goroutine
		go func() {
			startTime := time.Now()
			ticker := time.NewTicker(100 * time.Millisecond) // Update every 100ms
			defer ticker.Stop()

			// Set maxWait based on completion status
			maxWait := 2 * time.Second // Changed from 3 to 2 seconds for incomplete
			if isComplete {
				maxWait = 500 * time.Millisecond // Keep at 0.5 seconds for complete
			}

			for {
				select {
				case <-ticker.C:
					elapsed := time.Since(startTime)
					elapsedSeconds := elapsed.Seconds()

					// Clear the previous line and print the timer
					fmt.Printf("\rWaiting: %.1f seconds / %.1f seconds", elapsedSeconds, maxWait.Seconds())

					if elapsed >= maxWait {
						fmt.Println() // New line after timer completes
						if isComplete {
							fmt.Println("Processing: Complete sentence detected with 0.5 second silence")
						} else {
							fmt.Println("Processing: Incomplete sentence detected with 2 seconds silence")
						}
						at.processCompleteSentence(at.CurrentSentence, index)
						return
					}

					// Check if new audio is received
					if time.Since(at.lastInterimTimestamp) < elapsed {
						fmt.Println("\rTimer reset - new audio detected")
						return
					}
				}
			}
		}()
	}
}

// streamAudioResponse streams the audio response and resets state
func (at *AudioTranscriber) streamAudioResponse(audioData []byte) error {
	// Set speaking state before starting playback
	at.setSpeakingState(true)
	at.setListeningState(INTERRUPT_ONLY)

	// Clear the audio queue before playback
	at.clearAudioQueue()

	// Debug logging
	log.Printf("Starting audio response streaming (data length: %d bytes)", len(audioData))

	// Create a new reader for the MP3 data
	reader := bytes.NewReader(audioData)

	// Decode the MP3
	streamer, format, err := mp3.Decode(io.NopCloser(reader))
	if err != nil {
		return fmt.Errorf("error decoding MP3: %v", err)
	}
	defer streamer.Close()

	// Debug logging
	log.Printf("Audio format: %v Hz, %v channels", format.SampleRate, format.NumChannels)

	// Initialize speaker if needed
	err = speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))
	if err != nil {
		log.Printf("Speaker initialization error: %v", err)
		// Continue anyway as speaker might already be initialized
	}

	// Create done channel with buffer to prevent goroutine leak
	done := make(chan bool, 1)

	// Debug logging
	log.Println("Starting audio playback...")

	// Play the audio with error handling
	speaker.Play(beep.Seq(streamer, beep.Callback(func() {
		log.Println("Audio playback completed")
		done <- true
	})))

	// Wait for playback to finish with timeout
	select {
	case <-done:
		log.Println("Playback finished successfully")
	case <-time.After(30 * time.Second): // Timeout after 30 seconds
		log.Println("Playback timeout - forcing reset")
	}

	// Clear the speaker
	speaker.Clear()

	// Reset states immediately after playback
	at.setSpeakingState(false)
	at.setListeningState(FULL_LISTENING)

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
