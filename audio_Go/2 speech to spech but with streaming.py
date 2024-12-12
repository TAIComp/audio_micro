# package main

# import (
# 	"bufio"
# 	"bytes"
# 	"context"
# 	"encoding/json"
# 	"fmt"
# 	"io"
# 	"log"
# 	"os"
# 	"os/signal"
# 	"path/filepath"
# 	"strings"
# 	"sync"
# 	"syscall"
# 	"time"

# 	speech "cloud.google.com/go/speech/apiv1"
# 	speechpb "cloud.google.com/go/speech/apiv1/speechpb"
# 	texttospeech "cloud.google.com/go/texttospeech/apiv1"
# 	tts "cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
# 	"github.com/faiface/beep"
# 	"github.com/faiface/beep/mp3"
# 	"github.com/faiface/beep/speaker"
# 	"github.com/gordonklaus/portaudio"
# 	"github.com/joho/godotenv"
# 	"github.com/sashabaranov/go-openai"
# 	"google.golang.org/api/option"
# )

# // ListeningState represents the state of audio listening
# type ListeningState string

# const (
# 	FULL_LISTENING   ListeningState = "full_listening"
# 	INTERRUPT_ONLY   ListeningState = "interrupt_only"
# 	NOT_LISTENING    ListeningState = "not_listening"
# 	interruptKeyChar                = '`'
# )

# // AudioTranscriber handles audio transcription and response
# type AudioTranscriber struct {
# 	// Audio parameters
# 	Rate            int
# 	Chunk           int
# 	CurrentSentence string
# 	LastTranscript  string

# 	// Thread-safe queue for audio data
# 	audioQueue chan []byte

# 	// Google Speech client
# 	speechClient *speech.Client
# 	streamConfig *speechpb.StreamingRecognitionConfig

# 	// OpenAI client
# 	openaiClient *openai.Client
# 	Model        string

# 	// Google Text-to-Speech client
# 	ttsClient   *texttospeech.Client
# 	voice       *tts.VoiceSelectionParams
# 	audioConfig *tts.AudioConfig

# 	// PortAudio stream
# 	audioStream *portaudio.Stream

# 	// Listening state
# 	listeningState ListeningState

# 	// Interrupt commands
# 	interruptCommands map[string]struct{}

# 	// Flags
# 	isSpeaking bool

# 	// Transcript tracking
# 	lastInterimTimestamp time.Time
# 	interimCooldown      time.Duration

# 	// Interrupt handling
# 	lastInterruptTime time.Time
# 	interruptCooldown time.Duration

# 	// Paths
# 	interruptAudioPath string
# 	contextFolder      string
# 	randomFolder       string

# 	// Audio mappings
# 	contextFileMapping map[int]string
# 	randomFileNames    []string
# 	randomAudioFiles   []string
# 	usedRandomFiles    map[string]struct{}

# 	// Mic monitoring settings (Placeholder for future implementation)
# 	micVolume          float64
# 	noiseGateThreshold float64

# 	// Response generation
# 	stopGeneration  bool
# 	pendingResponse string

# 	// Synchronization
# 	mu sync.Mutex

# 	// Add new fields for better control
# 	inputMuted       bool
# 	muteThreshold    float64
# 	audioInputBuffer []float32
# 	processingLock   sync.Mutex
# 	interruptChan    chan struct{}

# 	// Audio state management
# 	isProcessingAudio bool
# 	isSpeakingMutex   sync.Mutex
# 	audioStateMutex   sync.Mutex

# 	// Channel for coordinating audio playback
# 	audioPlaybackDone chan struct{}

# 	// Audio state management
# 	audioState struct {
# 		sync.Mutex
# 		lastSpeechTime    time.Time
# 		lastProcessedText string
# 		isProcessing      bool
# 		feedbackBuffer    []string // Store recent responses to prevent feedback
# 	}

# 	// Audio processing control
# 	processingControl struct {
# 		sync.Mutex
# 		noiseGate       float64
# 		volumeThreshold float64
# 		cooldownPeriod  time.Duration
# 	}
# }

# // NewAudioTranscriber initializes the AudioTranscriber
# func NewAudioTranscriber() *AudioTranscriber {
# 	// Load environment variables from .env file
# 	err := godotenv.Load()
# 	if err != nil {
# 		log.Fatalf("Error loading .env file")
# 	}

# 	// Initialize OpenAI client
# 	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
# 	if openaiAPIKey == "" {
# 		log.Fatal("OPENAI_API_KEY not set")
# 	}
# 	openaiClient := openai.NewClient(openaiAPIKey)

# 	// Get Google credentials path from environment
# 	googleCredPath := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
# 	if googleCredPath == "" {
# 		log.Fatal("GOOGLE_APPLICATION_CREDENTIALS not set")
# 	}

# 	// Initialize Google Speech client
# 	ctx := context.Background()
# 	speechClient, err := speech.NewClient(ctx, option.WithCredentialsFile(googleCredPath))
# 	if err != nil {
# 		log.Fatalf("Failed to create speech client: %v", err)
# 	}

# 	// Initialize Google Text-to-Speech client
# 	ttsClient, err := texttospeech.NewClient(ctx, option.WithCredentialsFile(googleCredPath))
# 	if err != nil {
# 		log.Fatalf("Failed to create TTS client: %v", err)
# 	}

# 	transcriber := &AudioTranscriber{
# 		Rate:         16000,
# 		Chunk:        800, // 16000 / 20
# 		audioQueue:   make(chan []byte, 100),
# 		speechClient: speechClient,
# 		streamConfig: &speechpb.StreamingRecognitionConfig{
# 			Config: &speechpb.RecognitionConfig{
# 				Encoding:                   speechpb.RecognitionConfig_LINEAR16,
# 				SampleRateHertz:            int32(16000),
# 				LanguageCode:               "en-US",
# 				EnableAutomaticPunctuation: true,
# 			},
# 			InterimResults: true,
# 		},
# 		openaiClient: openaiClient,
# 		Model:        "gpt-4o-mini", // Ensure this model exists or adjust accordingly
# 		ttsClient:    ttsClient,
# 		voice: &tts.VoiceSelectionParams{
# 			LanguageCode: "en-US",
# 			Name:         "en-US-Casual-K",
# 			SsmlGender:   tts.SsmlVoiceGender_MALE,
# 		},
# 		audioConfig: &tts.AudioConfig{
# 			AudioEncoding: tts.AudioEncoding_MP3,
# 			SpeakingRate:  1.0,
# 			Pitch:         0.0,
# 		},
# 		listeningState: FULL_LISTENING,
# 		interruptCommands: map[string]struct{}{
# 			"stop":             {},
# 			"end":              {},
# 			"shut up":          {},
# 			"please stop":      {},
# 			"stop please":      {},
# 			"please end":       {},
# 			"end please":       {},
# 			"shut up please":   {},
# 			"please shut up":   {},
# 			"okay stop":        {},
# 			"ok stop":          {},
# 			"can you stop":     {},
# 			"could you stop":   {},
# 			"would you stop":   {},
# 			"can you be quiet": {},
# 			"silence":          {},
# 			"pause":            {},
# 		},
# 		isSpeaking:           false,
# 		lastInterimTimestamp: time.Now(),
# 		interimCooldown:      500 * time.Millisecond,
# 		lastInterruptTime:    time.Now(),
# 		interruptCooldown:    1 * time.Second,
# 		interruptAudioPath:   "interruption.mp3",
# 		contextFileMapping: map[int]string{
# 			0:  "Hmm_let_me_think_about_that",
# 			1:  "So_basically",
# 			2:  "Umm_okay_so",
# 			3:  "Let_me_figure_this_out",
# 			4:  "Hmm_that's_a_good_one",
# 			5:  "Alright_let's_see",
# 			6:  "Let's_think_about_this_for_a_moment",
# 			7:  "Ooooh_that's_tricky",
# 			8:  "Hmm_give_me_a_sec",
# 			9:  "So_one_moment_um",
# 			10: "Oh_okay_okay",
# 			11: "Aha_just_a_sec",
# 			12: "Alright_let's_dive_into_that",
# 			13: "Okay_okay_let's_see",
# 			14: "Hmm_this_is_interesting",
# 			15: "Okay_okay_let's_get_to_work",
# 			16: "So_yeah",
# 			17: "Uhh_well",
# 			18: "You_know",
# 			19: "So_anyway",
# 			20: "Alright_umm",
# 			21: "Oh_well_hmm",
# 			22: "Well_you_see",
# 			23: "So_basically_yeah",
# 			24: "Umm_anyway",
# 			25: "It's_uh_kinda_like",
# 		},
# 		randomFileNames: []string{
# 			"Hmm_let_me_think.mp3",
# 			"So.mp3",
# 			"Umm_well_well_well.mp3",
# 			"You_know.mp3",
# 		},
# 		randomAudioFiles:   []string{},
# 		usedRandomFiles:    make(map[string]struct{}),
# 		micVolume:          0.5,
# 		noiseGateThreshold: 0.5,
# 		stopGeneration:     false,
# 		pendingResponse:    "",
# 		inputMuted:         false,
# 		muteThreshold:      0.01,
# 		interruptChan:      make(chan struct{}, 1),
# 		audioPlaybackDone:  make(chan struct{}),
# 		processingControl: struct {
# 			sync.Mutex
# 			noiseGate       float64
# 			volumeThreshold float64
# 			cooldownPeriod  time.Duration
# 		}{
# 			noiseGate:       0.02,
# 			volumeThreshold: 0.1,
# 			cooldownPeriod:  time.Second,
# 		},
# 	}

# 	// Define audio folders
# 	transcriber.contextFolder = filepath.Join("voiceCashingSys", "contextBased")
# 	transcriber.randomFolder = filepath.Join("voiceCashingSys", "random")

# 	// Load random audio files
# 	for _, filename := range transcriber.randomFileNames {
# 		filePath := filepath.Join(transcriber.randomFolder, filename)
# 		if _, err := os.Stat(filePath); err == nil {
# 			transcriber.randomAudioFiles = append(transcriber.randomAudioFiles, filePath)
# 		}
# 	}

# 	// Check if interrupt audio file exists
# 	if _, err := os.Stat(transcriber.interruptAudioPath); os.IsNotExist(err) {
# 		fmt.Printf("Warning: Interrupt audio file '%s' not found!\n", transcriber.interruptAudioPath)
# 	}

# 	// Add this to NewAudioTranscriber
# 	for idx, filename := range transcriber.contextFileMapping {
# 		path := filepath.Join(transcriber.contextFolder, filename+".mp3")
# 		if _, err := os.Stat(path); os.IsNotExist(err) {
# 			fmt.Printf("Warning: Context audio file missing: %d - %s\n", idx, path)
# 		}
# 	}

# 	// Initialize the speaker
# 	if err := transcriber.initSpeaker(); err != nil {
# 		log.Printf("Warning: Failed to initialize speaker: %v", err)
# 	}

# 	// Initialize audio state
# 	transcriber.audioState.lastSpeechTime = time.Now()
# 	transcriber.audioState.feedbackBuffer = make([]string, 0, 10)

# 	return transcriber
# }

# // getContextAndCompletion retrieves context index and completion status from OpenAI
# func (at *AudioTranscriber) getContextAndCompletion(text string) (int, bool) {
# 	ctx := context.Background()
# 	req := openai.ChatCompletionRequest{
# 		Model: "gpt-4o-mini",
# 		Messages: []openai.ChatCompletionMessage{
# 			{
# 				Role: "system",
# 				Content: `Analyze the given text and provide TWO pieces of information:
# 1. The most appropriate filler phrase index (0-25) based on this context:

# Neutral Fillers:
# 0: "Hmm, let me think about that..."
# 1: "So, basically"
# 2: "Umm, okay so..."
# 3: "Let me figure this out..."
# 4: "Hmm, that's a good one..."
# 5: "Alright, let's see..."
# 6: "Let's think about this for a moment..."

# Casual and Friendly:
# 7: "Ooooh, that's tricky..."
# 8: "Hmm, give me a sec..."
# 9: "So, one moment... um"
# 10: "Oh, okay, okay..."
# 11: "Aha, just a sec..."
# 12: "Alright, let's dive into that!"

# Slightly Playful:
# 13: "Okay okay, let's see..."
# 14: "Hmm, this is interesting..."
# 15: "Okay okay, let's get to work..."

# Natural Fillers:
# 16: "So, yeah..."
# 17: "Uhh, well..."
# 18: "You know..."
# 19: "So, anyway..."
# 20: "Alright, umm..."
# 21: "Oh, well, hmm..."

# Casual Transitions:
# 22: "Well, you see..."
# 23: "So, basically, yeah..."
# 24: "Umm, anyway..."
# 25: "It's, uh, kinda like..."

# 2. Whether the sentence is complete, considering:
# - Grammatical completeness (subject + predicate)
# - Semantic completeness (complete thought/meaning)
# - Natural ending point (proper punctuation or logical conclusion)
# - Trailing indicators suggesting more is coming

# Return ONLY in this format:
# {"index": X, "complete": true/false}`,
# 			},
# 			{
# 				Role:    "user",
# 				Content: fmt.Sprintf("Analyze this text: '%s'", text),
# 			},
# 		},
# 	}

# 	resp, err := at.openaiClient.CreateChatCompletion(ctx, req)
# 	if err != nil {
# 		log.Printf("Error analyzing text: %v", err)
# 		return 0, false
# 	}

# 	if len(resp.Choices) == 0 {
# 		log.Println("No choices returned from OpenAI")
# 		return 0, false
# 	}

# 	content := resp.Choices[0].Message.Content
# 	// Parse the response; assuming JSON-like format
# 	content = strings.ReplaceAll(content, "'", "\"")
# 	custom := make(map[string]interface{})
# 	err = json.Unmarshal([]byte(content), &custom)
# 	if err != nil {
# 		log.Printf("Error parsing response: %v", err)
# 		return 0, false
# 	}

# 	index, ok1 := custom["index"].(float64) // JSON numbers are float64
# 	complete, ok2 := custom["complete"].(bool)
# 	if !ok1 || !ok2 {
# 		log.Println("Invalid response format")
# 		return 0, false
# 	}

# 	return int(index), complete
# }

# // getAIResponse retrieves a response from OpenAI based on the input text
# func (at *AudioTranscriber) getAIResponse(text string) string {
# 	ctx := context.Background()
# 	req := openai.ChatCompletionRequest{
# 		Model: "gpt-4o-mini",
# 		Messages: []openai.ChatCompletionMessage{
# 			{
# 				Role: "system",
# 				Content: `You are Quad, an AI-powered online teacher dedicated to making learning fun and engaging.

# Guidelines:
# 1. Accuracy & Relevance: Provide correct, focused information
# 2. Clarity: Use simple language and short sentences
# 3. Examples: Include real-world examples when helpful
# 4. Engagement: Make learning interactive and interesting
# 5. Conciseness: Keep responses brief but informative
# 6. Adaptability: Match explanation complexity to the question
# 7. Encouragement: Use positive reinforcement
# 8. Verification: End with a quick comprehension check when appropriate

# Response Format:
# - Start with a direct answer
# - Follow with a brief explanation if needed
# - Include an example or analogy when relevant
# - Keep total response length moderate
# - Use natural, conversational language

# Remember: Your speaking responses will be converted to speech, so keep sentences clear and well-paced.`,
# 			},
# 			{
# 				Role:    "user",
# 				Content: text,
# 			},
# 		},
# 	}

# 	resp, err := at.openaiClient.CreateChatCompletion(ctx, req)
# 	if err != nil {
# 		log.Printf("OpenAI API error: %v", err)
# 		return ""
# 	}

# 	if len(resp.Choices) == 0 {
# 		log.Println("No choices returned from OpenAI")
# 		return ""
# 	}

# 	return resp.Choices[0].Message.Content
# }

# // Helper function for audio data conversion
# func convertAudioData(data []int) []byte {
# 	dataBytes := make([]byte, len(data)*2)
# 	for i, sample := range data {
# 		sample16 := int16(sample)
# 		dataBytes[i*2] = byte(sample16)
# 		dataBytes[i*2+1] = byte(sample16 >> 8)
# 	}
# 	return dataBytes
# }

# // Text to speech conversion
# func (at *AudioTranscriber) textToSpeech(text string) ([]byte, error) {
# 	ctx := context.Background()
# 	req := &tts.SynthesizeSpeechRequest{
# 		Input: &tts.SynthesisInput{
# 			InputSource: &tts.SynthesisInput_Text{
# 				Text: text,
# 			},
# 		},
# 		Voice:       at.voice,
# 		AudioConfig: at.audioConfig,
# 	}

# 	resp, err := at.ttsClient.SynthesizeSpeech(ctx, req)
# 	if err != nil {
# 		return nil, fmt.Errorf("text-to-speech error: %v", err)
# 	}

# 	return resp.AudioContent, nil
# }

# // Process audio stream
# func (at *AudioTranscriber) processAudioStream(ctx context.Context) error {
# 	at.processingLock.Lock()
# 	defer at.processingLock.Unlock()

# 	// Initialize audio buffer
# 	at.audioInputBuffer = make([]float32, at.Chunk)
# 	at.muteThreshold = 0.01 // Adjust this value based on testing

# 	stream, err := at.speechClient.StreamingRecognize(ctx)
# 	if err != nil {
# 		return fmt.Errorf("could not create stream: %v", err)
# 	}

# 	// Start goroutine for handling interrupts
# 	go at.handleInterrupts(ctx)

# 	// Audio processing goroutine
# 	go func() {
# 		for {
# 			select {
# 			case data := <-at.audioQueue:
# 				if !at.shouldProcessAudio() {
# 					continue
# 				}

# 				// Process audio data
# 				if at.isSpeaking {
# 					// Only check for interrupt commands when speaking
# 					if at.containsInterruptInAudio(data) {
# 						at.interruptChan <- struct{}{}
# 						continue
# 					}
# 					// Mute regular input while speaking
# 					continue
# 				}

# 				// Send audio data when not speaking
# 				req := &speechpb.StreamingRecognizeRequest{
# 					StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{
# 						AudioContent: data,
# 					},
# 				}
# 				if err := stream.Send(req); err != nil {
# 					log.Printf("Could not send audio: %v", err)
# 				}
# 			case <-ctx.Done():
# 				stream.CloseSend()
# 				return
# 			}
# 		}
# 	}()

# 	for {
# 		resp, err := stream.Recv()
# 		if err != nil {
# 			return fmt.Errorf("error receiving response: %v", err)
# 		}
# 		at.processResponses([]*speechpb.StreamingRecognizeResponse{resp})
# 	}
# }

# // Process responses
# func (at *AudioTranscriber) processResponses(responses []*speechpb.StreamingRecognizeResponse) {
# 	for _, response := range responses {
# 		if len(response.Results) == 0 {
# 			continue
# 		}

# 		result := response.Results[0]
# 		transcript := strings.ToLower(strings.TrimSpace(result.Alternatives[0].Transcript))

# 		// Skip empty transcripts
# 		if transcript == "" {
# 			continue
# 		}

# 		// Check for audio feedback or system speech
# 		if at.isAudioFeedback(transcript) {
# 			continue
# 		}

# 		// Check speaking state with proper locking
# 		at.isSpeakingMutex.Lock()
# 		speaking := at.isSpeaking
# 		at.isSpeakingMutex.Unlock()

# 		// If speaking, only process interrupts
# 		if speaking {
# 			if at.containsInterruptCommand(transcript) {
# 				at.handleVoiceInterrupt()
# 			}
# 			continue
# 		}

# 		// Process based on result type
# 		if !result.IsFinal {
# 			at.handleInterimResult(transcript)
# 		} else {
# 			// Skip if this was an interrupt command
# 			if at.containsInterruptCommand(transcript) {
# 				continue
# 			}
# 			at.handleFinalResult(transcript)
# 		}
# 	}
# }

# // Play audio response
# func (at *AudioTranscriber) playAudioResponse(audioPath string) {
# 	// Set speaking state
# 	at.setSpeakingState(true)
# 	at.listeningState = INTERRUPT_ONLY
# 	at.inputMuted = true

# 	defer func() {
# 		at.setSpeakingState(false)
# 		at.listeningState = FULL_LISTENING
# 		at.inputMuted = false
# 		at.pendingResponse = ""            // Clear the pending response
# 		at.resetState()                    // Reset all states
# 		time.Sleep(500 * time.Millisecond) // Add longer delay after speech
# 		fmt.Println("\nListening... (Press ` to interrupt)")
# 	}()

# 	// Open the audio file
# 	f, err := os.Open(audioPath)
# 	if err != nil {
# 		log.Printf("Error opening audio file: %v", err)
# 		return
# 	}
# 	defer f.Close()

# 	// Decode the MP3 file
# 	streamer, format, err := mp3.Decode(f)
# 	if err != nil {
# 		log.Printf("Error decoding MP3: %v", err)
# 		return
# 	}
# 	defer streamer.Close()

# 	// Initialize the speaker if it hasn't been initialized yet
# 	speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))

# 	// Create a channel to signal when playback is done
# 	done := make(chan bool)

# 	// Play the audio
# 	speaker.Play(beep.Seq(streamer, beep.Callback(func() {
# 		done <- true
# 	})))

# 	// Wait for playback to finish
# 	<-done

# 	// Add a small delay after speech to prevent picking up echo
# 	time.Sleep(200 * time.Millisecond)
# }

# // handleKeyboardInterrupt handles the backtick key interrupt
# func (at *AudioTranscriber) handleKeyboardInterrupt() {
# 	currentTime := time.Now()
# 	if currentTime.Sub(at.lastInterruptTime) >= at.interruptCooldown {
# 		fmt.Println("\nBacktick interrupt detected!")

# 		// Immediately stop any playing audio
# 		speaker.Clear()

# 		// Set flags to stop ongoing processes
# 		at.stopGeneration = true
# 		at.setSpeakingState(false)

# 		// Handle the interrupt
# 		at.handleInterrupt()

# 		// Update interrupt time
# 		at.lastInterruptTime = currentTime

# 		// Clear audio queue
# 		for len(at.audioQueue) > 0 {
# 			<-at.audioQueue
# 		}

# 		// Force a small delay
# 		time.Sleep(200 * time.Millisecond)
# 	}
# }

# // handleInterrupt handles interrupt logic
# func (at *AudioTranscriber) handleInterrupt() {
# 	at.mu.Lock()
# 	defer at.mu.Unlock()

# 	// Stop any playing audio
# 	speaker.Clear()

# 	// Reset all states
# 	at.stopGeneration = true
# 	at.setSpeakingState(false)
# 	at.inputMuted = false
# 	at.listeningState = FULL_LISTENING
# 	at.CurrentSentence = ""
# 	at.LastTranscript = ""
# 	at.pendingResponse = ""

# 	// Clear audio queue
# 	for len(at.audioQueue) > 0 {
# 		<-at.audioQueue
# 	}

# 	// Play interrupt acknowledgment
# 	at.playAcknowledgment()

# 	// Force a small delay to ensure clean state
# 	time.Sleep(200 * time.Millisecond)

# 	// Reset the audio state
# 	at.audioState.Lock()
# 	at.audioState.lastProcessedText = ""
# 	at.audioState.isProcessing = false
# 	at.audioState.Unlock()

# 	fmt.Println("\nListening .... (press ` or say \"shut up\" for interruption)")
# }

# // playAcknowledgment plays the interruption acknowledgment audio
# func (at *AudioTranscriber) playAcknowledgment() {
# 	if _, err := os.Stat(at.interruptAudioPath); os.IsNotExist(err) {
# 		log.Printf("Interruption audio file not found: %s", at.interruptAudioPath)
# 		return
# 	}

# 	// Load and decode the MP3 file
# 	f, err := os.Open(at.interruptAudioPath)
# 	if err != nil {
# 		log.Printf("Error opening interrupt audio: %v", err)
# 		return
# 	}
# 	defer f.Close()

# 	streamer, format, err := mp3.Decode(f)
# 	if err != nil {
# 		log.Printf("Error decoding MP3: %v", err)
# 		return
# 	}
# 	defer streamer.Close()

# 	// Initialize speaker if needed
# 	speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))

# 	// Create done channel
# 	done := make(chan bool)

# 	// Play the audio
# 	speaker.Play(beep.Seq(streamer, beep.Callback(func() {
# 		done <- true
# 	})))

# 	// Wait for playback to finish
# 	<-done
# }

# // playNextRandomAudio plays the next random audio file
# func (at *AudioTranscriber) playNextRandomAudio() bool {
# 	if len(at.randomAudioFiles) == 0 {
# 		return false
# 	}

# 	// Reset used files if we've played them all
# 	if len(at.usedRandomFiles) == len(at.randomAudioFiles) {
# 		at.usedRandomFiles = make(map[string]struct{})
# 	}

# 	// Select an unused random file
# 	var availableFiles []string
# 	for _, file := range at.randomAudioFiles {
# 		if _, used := at.usedRandomFiles[file]; !used {
# 			availableFiles = append(availableFiles, file)
# 		}
# 	}

# 	if len(availableFiles) == 0 {
# 		return false
# 	}

# 	// Select a random file
# 	randomIndex := time.Now().UnixNano() % int64(len(availableFiles))
# 	audioFile := availableFiles[randomIndex]
# 	at.usedRandomFiles[audioFile] = struct{}{}

# 	// Play the random audio file
# 	file, err := os.Open(audioFile)
# 	if err != nil {
# 		log.Printf("Error playing random audio: %v", err)
# 		return false
# 	}
# 	defer file.Close()

# 	// Create a new PortAudio stream for playback
# 	outputStream, err := portaudio.OpenDefaultStream(0, 2, 44100, 1024, make([]float32, 1024))
# 	if err != nil {
# 		log.Printf("Error opening playback stream: %v", err)
# 		return false
# 	}
# 	defer outputStream.Close()

# 	if err := outputStream.Start(); err != nil {
# 		log.Printf("Error starting playback stream: %v", err)
# 		return false
# 	}
# 	defer outputStream.Stop()

# 	// Read and play the audio data
# 	buffer := make([]byte, 4096)
# 	samples := make([]float32, 2048) // stereo buffer
# 	for {
# 		n, err := file.Read(buffer)
# 		if err == io.EOF {
# 			break
# 		}
# 		if err != nil {
# 			log.Printf("Error reading audio file: %v", err)
# 			break
# 		}

# 		// Convert bytes to float32 samples (16-bit PCM to float32)
# 		numSamples := n / 2 // number of samples we can convert
# 		if numSamples > len(samples)/2 {
# 			numSamples = len(samples) / 2
# 		}

# 		for i := 0; i < numSamples; i++ {
# 			if i*2+1 >= n {
# 				break
# 			}
# 			sample := float32(int16(buffer[i*2])|int16(buffer[i*2+1])<<8) / 32768.0
# 			// Duplicate sample for stereo
# 			samples[i*2] = sample
# 			samples[i*2+1] = sample
# 		}

# 		// Clear the rest of the buffer if we didn't fill it completely
# 		for i := numSamples * 2; i < len(samples); i++ {
# 			samples[i] = 0
# 		}

# 		if err := outputStream.Write(); err != nil {
# 			log.Printf("Error playing audio: %v", err)
# 			break
# 		}

# 		// Add a small delay to prevent buffer overrun
# 		time.Sleep(10 * time.Millisecond)
# 	}

# 	at.listeningState = INTERRUPT_ONLY
# 	log.Printf("Playing random audio: %s", filepath.Base(audioFile))
# 	return true
# }

# // resetState resets the transcription state
# func (at *AudioTranscriber) resetState() {
# 	at.mu.Lock()
# 	defer at.mu.Unlock()

# 	at.CurrentSentence = ""
# 	at.LastTranscript = ""
# 	at.lastInterimTimestamp = time.Now()
# 	at.lastInterruptTime = time.Now() // Reset interrupt timer too
# }

# // processCompleteSentence handles the processing of a complete sentence
# func (at *AudioTranscriber) processCompleteSentence(sentence string, initialAudioIndex int) {
# 	at.mu.Lock()
# 	at.stopGeneration = false
# 	at.mu.Unlock()

# 	// Generate AI response
# 	response := at.getAIResponse(sentence)
# 	if response == "" {
# 		return
# 	}

# 	// Print what AI wants to say
# 	fmt.Printf("\nAI wants to say: %s\n\n", response)

# 	// Store the response for filtering
# 	at.pendingResponse = response

# 	// Play context audio if not interrupted
# 	if !at.stopGeneration {
# 		contextFile := at.contextFileMapping[initialAudioIndex]
# 		fmt.Printf("Playing context audio: %s.mp3\n", contextFile)
# 		at.playContextAudio(initialAudioIndex)
# 		time.Sleep(500 * time.Millisecond)
# 	}

# 	// Convert and stream AI response if not interrupted
# 	if response != "" && !at.stopGeneration {
# 		fmt.Println("Streaming AI response audio...")
# 		audioData, err := at.textToSpeech(response)
# 		if err != nil {
# 			log.Printf("Error converting text to speech: %v", err)
# 			return
# 		}

# 		if err := at.streamAudioResponse(audioData); err != nil {
# 			log.Printf("Error streaming audio: %v", err)
# 			return
# 		}
# 	}

# 	// Update last speech time
# 	at.audioState.Lock()
# 	at.audioState.lastSpeechTime = time.Now()
# 	at.audioState.Unlock()

# 	// Reset states
# 	at.resetState()

# 	// Show listening message
# 	fmt.Println("\nListening .... (press ` or say \"shut up\" for interruption)")
# }

# // Play context audio
# func (at *AudioTranscriber) playContextAudio(index int) bool {
# 	if at.isSpeaking {
# 		return false
# 	}

# 	if filename, exists := at.contextFileMapping[index]; exists {
# 		contextFile := filepath.Join(at.contextFolder, filename+".mp3")
# 		if _, err := os.Stat(contextFile); os.IsNotExist(err) {
# 			log.Printf("Context audio file not found: %s", contextFile)
# 			return false
# 		}

# 		// Open the audio file
# 		f, err := os.Open(contextFile)
# 		if err != nil {
# 			log.Printf("Error opening context audio: %v", err)
# 			return false
# 		}
# 		defer f.Close()

# 		// Decode the MP3 file
# 		streamer, format, err := mp3.Decode(f)
# 		if err != nil {
# 			log.Printf("Error decoding MP3: %v", err)
# 			return false
# 		}
# 		defer streamer.Close()

# 		// Initialize the speaker if it hasn't been initialized yet
# 		speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))

# 		// Create a channel to signal when playback is done
# 		done := make(chan bool)

# 		// Play the audio
# 		speaker.Play(beep.Seq(streamer, beep.Callback(func() {
# 			done <- true
# 		})))

# 		// Wait for playback to finish
# 		<-done

# 		at.listeningState = INTERRUPT_ONLY
# 		return true
# 	}

# 	return false
# }

# // handleVoiceInterrupt handles voice-based interrupt commands
# func (at *AudioTranscriber) handleVoiceInterrupt() {
# 	currentTime := time.Now()
# 	if currentTime.Sub(at.lastInterruptTime) >= at.interruptCooldown {
# 		fmt.Println("\nInterrupt command detected!")
# 		at.handleInterrupt()
# 		at.lastInterruptTime = currentTime
# 	}
# }

# // Add a new method to handle speaker initialization
# func (at *AudioTranscriber) initSpeaker() error {
# 	// Open a sample audio file to get the format
# 	sampleFile := filepath.Join(at.contextFolder, at.contextFileMapping[0]+".mp3")
# 	f, err := os.Open(sampleFile)
# 	if err != nil {
# 		return fmt.Errorf("error opening sample file: %v", err)
# 	}
# 	defer f.Close()

# 	// Decode the sample file to get the format
# 	_, format, err := mp3.Decode(f)
# 	if err != nil {
# 		return fmt.Errorf("error decoding sample file: %v", err)
# 	}

# 	// Initialize the speaker
# 	err = speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))
# 	if err != nil {
# 		return fmt.Errorf("error initializing speaker: %v", err)
# 	}

# 	return nil
# }

# // Add a method to handle interruption of audio playback
# func (at *AudioTranscriber) stopAudio() {
# 	speaker.Clear()
# }

# func init() {
# 	// Open /dev/null for writing
# 	null, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
# 	if err != nil {
# 		log.Fatal(err)
# 	}

# 	// Duplicate the file descriptor
# 	if err = syscall.Dup2(int(null.Fd()), int(os.Stderr.Fd())); err != nil {
# 		log.Fatal(err)
# 	}
# }

# func main() {
# 	// Initialize PortAudio
# 	if err := portaudio.Initialize(); err != nil {
# 		log.Fatal(err)
# 	}
# 	defer portaudio.Terminate()

# 	// Create transcriber
# 	transcriber := NewAudioTranscriber()
# 	log.Println("Starting audio transcription... Speak into your microphone.")
# 	log.Println("Press ` (backtick) to interrupt at any time.")

# 	fmt.Println("\nListening .... (press ` or say \"shut up\" for interruption)")

# 	// Create context with cancellation
# 	ctx, cancel := context.WithCancel(context.Background())
# 	defer cancel()

# 	// Create channel for keyboard interrupts
# 	interruptChan := make(chan struct{}, 1)

# 	// Start keyboard listener in a separate goroutine
# 	go func() {
# 		reader := bufio.NewReader(os.Stdin)
# 		for {
# 			char, _, err := reader.ReadRune()
# 			if err != nil {
# 				log.Printf("Error reading keyboard input: %v", err)
# 				continue
# 			}
# 			if char == '`' {
# 				select {
# 				case interruptChan <- struct{}{}:
# 					// Signal sent successfully
# 				default:
# 					// Channel is full, skip this interrupt
# 				}
# 			}
# 		}
# 	}()

# 	// Handle interrupts in a separate goroutine
# 	go func() {
# 		for range interruptChan {
# 			transcriber.handleKeyboardInterrupt()
# 		}
# 	}()

# 	// Handle OS interrupts
# 	go func() {
# 		c := make(chan os.Signal, 1)
# 		signal.Notify(c, os.Interrupt)
# 		<-c
# 		log.Println("\nTranscription stopped by user")
# 		cancel()
# 	}()

# 	// Create streaming recognition client
# 	stream, err := transcriber.speechClient.StreamingRecognize(ctx)
# 	if err != nil {
# 		log.Fatal(err)
# 	}

# 	// Send the initial configuration message
# 	configRequest := &speechpb.StreamingRecognizeRequest{
# 		StreamingRequest: &speechpb.StreamingRecognizeRequest_StreamingConfig{
# 			StreamingConfig: &speechpb.StreamingRecognitionConfig{
# 				Config: &speechpb.RecognitionConfig{
# 					Encoding:                   speechpb.RecognitionConfig_LINEAR16,
# 					SampleRateHertz:            16000,
# 					LanguageCode:               "en-US",
# 					EnableAutomaticPunctuation: true,
# 				},
# 				InterimResults: true,
# 			},
# 		},
# 	}

# 	if err := stream.Send(configRequest); err != nil {
# 		log.Fatalf("Could not send streaming config: %v", err)
# 	}

# 	// Initialize audio input stream
# 	inputChannels := 1
# 	outputChannels := 0
# 	sampleRate := 16000
# 	framesPerBuffer := make([]int16, 800)

# 	audioStream, err := portaudio.OpenDefaultStream(
# 		inputChannels,
# 		outputChannels,
# 		float64(sampleRate),
# 		len(framesPerBuffer),
# 		framesPerBuffer,
# 	)
# 	if err != nil {
# 		log.Fatal(err)
# 	}
# 	defer audioStream.Close()

# 	if err := audioStream.Start(); err != nil {
# 		log.Fatal(err)
# 	}
# 	defer audioStream.Stop()

# 	// Start processing audio in a separate goroutine
# 	go func() {
# 		for {
# 			select {
# 			case <-ctx.Done():
# 				return
# 			default:
# 				if err := audioStream.Read(); err != nil {
# 					log.Printf("Error reading from audio stream: %v", err)
# 					continue
# 				}

# 				// Convert samples to bytes
# 				audioBytes := make([]byte, len(framesPerBuffer)*2)
# 				for i, sample := range framesPerBuffer {
# 					audioBytes[i*2] = byte(sample)
# 					audioBytes[i*2+1] = byte(sample >> 8)
# 				}

# 				// Send audio data
# 				if err := stream.Send(&speechpb.StreamingRecognizeRequest{
# 					StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{
# 						AudioContent: audioBytes,
# 					},
# 				}); err != nil {
# 					log.Printf("Could not send audio: %v", err)
# 					continue
# 				}
# 			}
# 		}
# 	}()

# 	// Process responses
# 	for {
# 		resp, err := stream.Recv()
# 		if err == io.EOF {
# 			break
# 		}
# 		if err != nil {
# 			log.Fatalf("Cannot stream results: %v", err)
# 		}
# 		if err := ctx.Err(); err != nil {
# 			return
# 		}

# 		// Check for keyboard interrupt
# 		select {
# 		case <-interruptChan:
# 			transcriber.handleKeyboardInterrupt()
# 		default:
# 			// Process the response if no interrupt
# 			transcriber.processResponses([]*speechpb.StreamingRecognizeResponse{resp})
# 		}
# 	}
# }

# // Add these new methods to the AudioTranscriber struct

# // shouldProcessAudio determines whether incoming audio should be processed
# func (at *AudioTranscriber) shouldProcessAudio() bool {
# 	at.mu.Lock()
# 	defer at.mu.Unlock()

# 	if at.isSpeaking {
# 		return at.listeningState == INTERRUPT_ONLY
# 	}
# 	return true
# }

# // handleInterrupts manages both keyboard and voice interrupts
# func (at *AudioTranscriber) handleInterrupts(ctx context.Context) {
# 	// Set up keyboard event listener
# 	keyboardChan := make(chan rune)
# 	go func() {
# 		buf := make([]byte, 1)
# 		for {
# 			_, err := os.Stdin.Read(buf)
# 			if err != nil {
# 				log.Printf("Error reading keyboard input: %v", err)
# 				continue
# 			}
# 			if buf[0] == byte(interruptKeyChar) {
# 				keyboardChan <- interruptKeyChar
# 			}
# 		}
# 	}()

# 	for {
# 		select {
# 		case <-keyboardChan:
# 			at.handleKeyboardInterrupt()
# 		case <-at.interruptChan:
# 			at.handleVoiceInterrupt()
# 		case <-ctx.Done():
# 			return
# 		}
# 	}
# }

# // containsInterruptCommand checks if the audio data contains an interrupt command
# func (at *AudioTranscriber) containsInterruptCommand(text string) bool {
# 	if text == "" {
# 		return false
# 	}

# 	at.mu.Lock()
# 	defer at.mu.Unlock()

# 	// Check if enough time has passed since last interrupt
# 	if time.Since(at.lastInterruptTime) < at.interruptCooldown {
# 		return false
# 	}

# 	// Check each command as a whole word
# 	words := strings.Fields(strings.ToLower(text))
# 	text = strings.Join(words, " ")

# 	for cmd := range at.interruptCommands {
# 		if strings.Contains(text, cmd) {
# 			return true
# 		}
# 	}

# 	return false
# }

# // Add method to safely control speaking state
# func (at *AudioTranscriber) setSpeakingState(speaking bool) {
# 	at.isSpeakingMutex.Lock()
# 	defer at.isSpeakingMutex.Unlock()
# 	at.isSpeaking = speaking
# }

# // Add a new method for checking raw audio data
# func (at *AudioTranscriber) containsInterruptInAudio(data []byte) bool {
# 	// Check audio energy/volume
# 	var energy float64
# 	for i := 0; i < len(data); i += 2 {
# 		if i+1 >= len(data) {
# 			break
# 		}
# 		sample := int16(data[i]) | int16(data[i+1])<<8
# 		energy += float64(sample) * float64(sample)
# 	}
# 	energy /= float64(len(data) / 2)

# 	// If energy is above threshold, consider it as potential interrupt
# 	return energy > 1000000 // Adjust threshold as needed
# }

# // Add new method to check for audio feedback
# func (at *AudioTranscriber) isAudioFeedback(text string) bool {
# 	at.audioState.Lock()
# 	defer at.audioState.Unlock()

# 	// Check against recent responses
# 	for _, response := range at.audioState.feedbackBuffer {
# 		if strings.Contains(strings.ToLower(text), strings.ToLower(response)) {
# 			return true
# 		}
# 	}

# 	// Check cooldown period
# 	if time.Since(at.audioState.lastSpeechTime) < at.processingControl.cooldownPeriod {
# 		return true
# 	}

# 	return false
# }

# // Add new method for handling interim results
# func (at *AudioTranscriber) handleInterimResult(transcript string) {
# 	if transcript != at.LastTranscript && time.Since(at.lastInterimTimestamp) >= at.interimCooldown {
# 		fmt.Printf(`Interim: "%s"`+"\n", transcript)
# 		at.LastTranscript = transcript
# 		at.lastInterimTimestamp = time.Now()
# 	}
# }

# // Add new method for handling final results
# func (at *AudioTranscriber) handleFinalResult(transcript string) {
# 	// Skip if this is a recent response
# 	if at.isAudioFeedback(transcript) {
# 		return
# 	}

# 	// Update current sentence
# 	at.audioState.Lock()
# 	if at.CurrentSentence == "" {
# 		at.CurrentSentence = transcript
# 	} else {
# 		newWords := strings.Fields(transcript)
# 		for _, word := range newWords {
# 			if !strings.Contains(at.CurrentSentence, word) {
# 				at.CurrentSentence += " " + word
# 			}
# 		}
# 	}
# 	at.audioState.Unlock()

# 	// Get context and completion status
# 	index, isComplete := at.getContextAndCompletion(at.CurrentSentence)
# 	fmt.Printf(`Final: "%s" (Complete: %t, Audio Index: %d)`+"\n", at.CurrentSentence, isComplete, index)

# 	// Process if conditions are met
# 	silenceDuration := time.Since(at.lastInterruptTime)
# 	if !at.isSpeaking {
# 		if (isComplete && silenceDuration >= time.Second) ||
# 			(!isComplete && silenceDuration >= 5*time.Second) {
# 			fmt.Println("Processing: Sentence " +
# 				(map[bool]string{true: "complete", false: "incomplete"})[isComplete] +
# 				" and " +
# 				(map[bool]string{true: "1 second", false: "5 seconds"})[isComplete] +
# 				" silence passed")
# 			at.processCompleteSentence(at.CurrentSentence, index)
# 		}
# 	}
# }

# // Add new method for streaming audio playback
# func (at *AudioTranscriber) streamAudioResponse(audioData []byte) error {
# 	// Create a new reader for the MP3 data
# 	reader := bytes.NewReader(audioData)

# 	// Decode the MP3
# 	streamer, format, err := mp3.Decode(io.NopCloser(reader))
# 	if err != nil {
# 		return fmt.Errorf("error decoding MP3: %v", err)
# 	}
# 	defer streamer.Close()

# 	// Initialize speaker if needed
# 	speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))

# 	// Create done channel
# 	done := make(chan bool)

# 	// Play the audio
# 	speaker.Play(beep.Seq(streamer, beep.Callback(func() {
# 		done <- true
# 	})))

# 	// Wait for playback to finish
# 	<-done
# 	return nil
# }
