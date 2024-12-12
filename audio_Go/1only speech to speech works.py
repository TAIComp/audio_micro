# package main

# import (
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
# 	"time"
# 	"syscall"

# 	speech "cloud.google.com/go/speech/apiv1"
# 	speechpb "cloud.google.com/go/speech/apiv1/speechpb"
# 	texttospeech "cloud.google.com/go/texttospeech/apiv1"
# 	tts "cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
# 	"github.com/gordonklaus/portaudio"
# 	"github.com/joho/godotenv"
# 	"github.com/sashabaranov/go-openai"
# 	"google.golang.org/api/option"
# 	"github.com/faiface/beep"
# 	"github.com/faiface/beep/mp3"
# 	"github.com/faiface/beep/speaker"
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
# 		} else {
# 			fmt.Printf("Found context audio file: %d - %s\n", idx, path)
# 		}
# 	}

# 	// Initialize the speaker
# 	if err := transcriber.initSpeaker(); err != nil {
# 		log.Printf("Warning: Failed to initialize speaker: %v", err)
# 	}

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
# func (at *AudioTranscriber) textToSpeech(text string) string {
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
# 		log.Printf("Text-to-Speech error: %v", err)
# 		return ""
# 	}

# 	outputDir := "ai_responses"
# 	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
# 		log.Printf("Error creating output directory: %v", err)
# 		return ""
# 	}

# 	outputPath := filepath.Join(outputDir, "latest_response.mp3")
# 	if err := os.WriteFile(outputPath, resp.AudioContent, 0644); err != nil {
# 		log.Printf("Error saving TTS audio: %v", err)
# 		return ""
# 	}

# 	return outputPath
# }

# // Process audio stream
# func (at *AudioTranscriber) processAudioStream(ctx context.Context) error {
# 	stream, err := at.speechClient.StreamingRecognize(ctx)
# 	if err != nil {
# 		return fmt.Errorf("could not create stream: %v", err)
# 	}

# 	go func() {
# 		for {
# 			select {
# 			case data := <-at.audioQueue:
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

# 		currentTime := time.Now()

# 		// Check for interrupt commands
# 		if currentTime.Sub(at.lastInterruptTime) >= at.interruptCooldown {
# 			for cmd := range at.interruptCommands {
# 				if strings.Contains(transcript, cmd) {
# 					at.handleVoiceInterrupt()
# 					return
# 				}
# 			}
# 		}

# 		// Skip processing if not in FULL_LISTENING state
# 		if at.listeningState != FULL_LISTENING {
# 			continue
# 		}

# 		if !result.IsFinal {
# 			if transcript != at.LastTranscript && time.Since(at.lastInterimTimestamp) >= at.interimCooldown {
# 				fmt.Printf(`Interim: "%s"`+"\n", transcript)
# 				at.LastTranscript = transcript
# 				at.lastInterimTimestamp = time.Now()
# 			}
# 		} else {
# 			// Skip interrupt commands
# 			if _, exists := at.interruptCommands[transcript]; exists {
# 				return
# 			}

# 			// Update current sentence
# 			if at.CurrentSentence == "" {
# 				at.CurrentSentence = transcript
# 			} else {
# 				newWords := strings.Fields(transcript)
# 				for _, word := range newWords {
# 					if !strings.Contains(at.CurrentSentence, word) {
# 						at.CurrentSentence += " " + word
# 					}
# 				}
# 			}

# 			index, isComplete := at.getContextAndCompletion(at.CurrentSentence)
# 			fmt.Printf(`Final: "%s" (Complete: %t, Audio Index: %d)`+"\n", at.CurrentSentence, isComplete, index)

# 			// Only process if not currently speaking and enough silence has passed
# 			silenceDuration := time.Since(at.lastInterruptTime).Seconds()
# 			if !at.isSpeaking && ((isComplete && silenceDuration >= 1.0) || (!isComplete && silenceDuration >= 5.0)) {
# 				fmt.Println("\nProcessing: Sentence complete and 1 second silence passed")
# 				go at.processCompleteSentence(at.CurrentSentence, index)
# 				at.resetState()
# 			}
# 		}
# 	}
# }

# // Play audio response
# func (at *AudioTranscriber) playAudioResponse(audioPath string) {
# 	at.mu.Lock()
# 	at.isSpeaking = true
# 	at.listeningState = INTERRUPT_ONLY
# 	at.mu.Unlock()

# 	defer func() {
# 		at.mu.Lock()
# 		at.isSpeaking = false
# 		at.listeningState = FULL_LISTENING
# 		at.mu.Unlock()
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
# }

# // handleKeyboardInterrupt handles the backtick key interrupt
# func (at *AudioTranscriber) handleKeyboardInterrupt() {
# 	currentTime := time.Now()
# 	if currentTime.Sub(at.lastInterruptTime) >= at.interruptCooldown {
# 		fmt.Println("\nBacktick interrupt detected!")
# 		at.handleInterrupt()
# 		at.lastInterruptTime = currentTime
# 	}
# }

# // handleInterrupt handles interrupt logic
# func (at *AudioTranscriber) handleInterrupt() {
# 	at.mu.Lock()
# 	defer at.mu.Unlock()

# 	// Stop any playing audio
# 	at.stopAudio()

# 	// Signal to stop AI response generation
# 	at.stopGeneration = true

# 	// Reset all audio and processing states
# 	at.isSpeaking = false
# 	at.listeningState = FULL_LISTENING

# 	// Clear all transcript variables
# 	at.CurrentSentence = ""
# 	at.LastTranscript = ""
# 	at.pendingResponse = ""
# 	at.stopGeneration = false

# 	fmt.Println("\nListening... (Press ` to interrupt)")
# }

# // playAcknowledgment plays the interruption acknowledgment audio
# func (at *AudioTranscriber) playAcknowledgment() {
# 	if _, err := os.Stat(at.interruptAudioPath); os.IsNotExist(err) {
# 		log.Printf("Interruption audio file not found: %s", at.interruptAudioPath)
# 		return
# 	}

# 	// Open the acknowledgment audio file
# 	file, err := os.Open(at.interruptAudioPath)
# 	if err != nil {
# 		log.Printf("Error playing acknowledgment: %v", err)
# 		return
# 	}
# 	defer file.Close()

# 	// Create a new PortAudio stream for playback
# 	outputStream, err := portaudio.OpenDefaultStream(0, 2, 44100, 1024, make([]float32, 1024))
# 	if err != nil {
# 		log.Printf("Error opening playback stream: %v", err)
# 		return
# 	}
# 	defer outputStream.Close()

# 	if err := outputStream.Start(); err != nil {
# 		log.Printf("Error starting playback stream: %v", err)
# 		return
# 	}
# 	defer outputStream.Stop()

# 	// Read and play the audio data
# 	buffer := make([]byte, 4096)
# 	for {
# 		n, err := file.Read(buffer)
# 		if err == io.EOF {
# 			break
# 		}
# 		if err != nil {
# 			log.Printf("Error reading audio file: %v", err)
# 			break
# 		}

# 		// Convert bytes to float32 samples
# 		samples := make([]float32, n/2)
# 		for i := 0; i < n/2; i++ {
# 			sample := float32(int16(buffer[i*2])|int16(buffer[i*2+1])<<8) / 32768.0
# 			samples[i] = sample
# 		}

# 		err = outputStream.Write()
# 		if err != nil {
# 			log.Printf("Error playing audio: %v", err)
# 			break
# 		}
# 	}
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
# 		numSamples := n / 2  // number of samples we can convert
# 		if numSamples > len(samples)/2 {
# 			numSamples = len(samples)/2
# 		}
		
# 		for i := 0; i < numSamples; i++ {
# 			if i*2+1 >= n {
# 				break
# 			}
# 			sample := float32(int16(buffer[i*2]) | int16(buffer[i*2+1])<<8) / 32768.0
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
# 	at.CurrentSentence = ""
# 	at.LastTranscript = ""
# 	at.pendingResponse = ""
# 	at.isSpeaking = false
# 	at.listeningState = FULL_LISTENING
# 	at.lastInterimTimestamp = time.Now()
# }

# // processCompleteSentence handles the processing of a complete sentence
# func (at *AudioTranscriber) processCompleteSentence(sentence string, initialAudioIndex int) {
# 	defer func() {
# 		at.stopGeneration = false
# 	}()

# 	at.stopGeneration = false

# 	// 1. Generate AI response first
# 	response := at.getAIResponse(sentence)
# 	if response == "" || at.stopGeneration {
# 		return
# 	}
# 	fmt.Printf("AI Response: %s\n", response)

# 	// 2. Play context audio
# 	if !at.stopGeneration {
# 		contextFile := at.contextFileMapping[initialAudioIndex]
# 		fmt.Printf("Playing context audio: %s.mp3\n", contextFile)
# 		at.playContextAudio(initialAudioIndex)
# 		// Wait for context audio to finish
# 		time.Sleep(500 * time.Millisecond)
# 	}

# 	// 3. Convert and play AI response
# 	if response != "" && !at.stopGeneration {
# 		audioPath := at.textToSpeech(response)
# 		if audioPath != "" {
# 			fmt.Println("Playing ai response audio")
# 			at.playAudioResponse(audioPath)
# 		}
# 	}
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

# 	// Create context with cancellation
# 	ctx, cancel := context.WithCancel(context.Background())
# 	defer cancel()

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
# 					SampleRateHertz:           16000,
# 					LanguageCode:              "en-US",
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
# 		transcriber.processResponses([]*speechpb.StreamingRecognizeResponse{resp})
# 	}
# }
