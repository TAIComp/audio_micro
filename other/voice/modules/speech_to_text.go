package modules

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"cloud.google.com/go/speech/apiv1/speechpb"
	"github.com/gordonklaus/portaudio"
	"github.com/faiface/beep/speaker"
)

// ProcessAudioStream manages the streaming recognition process
func (at *AudioTranscriber) ProcessAudioStream(ctx context.Context) (err error) {
	// Add panic recovery
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Recovered from panic in ProcessAudioStream: %v", r)
			err = fmt.Errorf("panic recovered: %v", r)
		}
	}()

	log.Println("Starting ProcessAudioStream")
	
	// Initialize audio input
	inputChannels := 1
	sampleRate := 16000
	framesPerBuffer := make([]float32, 1024)

	// Open default input stream
	log.Println("Opening audio stream...")
	stream, err := portaudio.OpenDefaultStream(inputChannels, 0, float64(sampleRate), len(framesPerBuffer), framesPerBuffer)
	if err != nil {
		log.Printf("Failed to open stream: %v", err)
		return fmt.Errorf("failed to open stream: %v", err)
	}
	defer func() {
		log.Println("Closing audio stream...")
		if err := stream.Close(); err != nil {
			log.Printf("Error closing stream: %v", err)
		}
	}()

	log.Println("Starting audio stream...")
	if err := stream.Start(); err != nil {
		log.Printf("Failed to start stream: %v", err)
		return fmt.Errorf("failed to start stream: %v", err)
	}
	defer func() {
		log.Println("Stopping audio stream...")
		if err := stream.Stop(); err != nil {
			log.Printf("Error stopping stream: %v", err)
		}
	}()

	// Create speech recognition stream
	log.Println("Creating speech recognition stream...")
	speechStream, err := at.speechClient.StreamingRecognize(ctx)
	if err != nil {
		log.Printf("Could not create speech stream: %v", err)
		return fmt.Errorf("could not create stream: %v", err)
	}
	defer func() {
		log.Println("Closing speech stream...")
		if err := speechStream.CloseSend(); err != nil {
			log.Printf("Error closing speech stream: %v", err)
		}
	}()

	// Send the configuration request
	log.Println("Sending config request...")
	if err := speechStream.Send(&speechpb.StreamingRecognizeRequest{
		StreamingRequest: &speechpb.StreamingRecognizeRequest_StreamingConfig{
			StreamingConfig: &speechpb.StreamingRecognitionConfig{
				Config: &speechpb.RecognitionConfig{
					Encoding:        speechpb.RecognitionConfig_LINEAR16,
					SampleRateHertz: int32(sampleRate),
					LanguageCode:    "en-US",
				},
				InterimResults: true,
			},
		},
	}); err != nil {
		log.Printf("Could not send config: %v", err)
		return fmt.Errorf("could not send config: %v", err)
	}

	// Create channels for goroutine communication
	errChan := make(chan error, 1)
	doneChan := make(chan struct{})

	// Start goroutine for handling audio input
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Recovered from panic in audio input goroutine: %v", r)
				errChan <- fmt.Errorf("panic in audio input: %v", r)
			}
			close(doneChan)
		}()

		log.Println("Starting audio input goroutine...")
		
		for {
			select {
			case <-ctx.Done():
				return
			default:
				// Read audio data
				err := stream.Read()
				if err != nil {
					log.Printf("Failed to read from stream: %v", err)
					errChan <- fmt.Errorf("failed to read from stream: %v", err)
					return
				}

				// Convert audio data to bytes
				audioData := make([]byte, len(framesPerBuffer)*2)
				for i, sample := range framesPerBuffer {
					binary.LittleEndian.PutUint16(audioData[i*2:], uint16(sample*32767.0))
				}

				// Send audio data
				if err := speechStream.Send(&speechpb.StreamingRecognizeRequest{
					StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{
						AudioContent: audioData,
					},
				}); err != nil {
					log.Printf("Could not send audio: %v", err)
					errChan <- fmt.Errorf("could not send audio: %v", err)
					return
				}
			}
		}
	}()

	// Start goroutine for handling responses
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Recovered from panic in response handling goroutine: %v", r)
				errChan <- fmt.Errorf("panic in response handling: %v", r)
			}
		}()

		var lastState ListeningState = FULL_LISTENING
		var lastAudioPlaying bool = false
		log.Println("Starting response handling goroutine...")
		
		clearTranscripts := func() {
			at.mu.Lock()
			defer at.mu.Unlock()
			at.InterimResult = ""
			at.FinalResult = ""
			at.InterruptTranscript = ""
			at.LastTranscript = ""
			at.CurrentSentence = ""
			time.Sleep(100 * time.Millisecond)
		}
		
		for {
			select {
			case <-ctx.Done():
				return
			case <-doneChan:
				return
			default:
				resp, err := speechStream.Recv()
				if err == io.EOF {
					log.Println("Received EOF from speech stream")
					return
				}
				if err != nil {
					log.Printf("Cannot receive: %v", err)
					errChan <- fmt.Errorf("cannot receive: %v", err)
					return
				}

				currentState := at.getListeningState()
				currentAudioPlaying := at.isAudioPlaying()

				// Clear transcripts in these scenarios:
				if currentState != lastState || currentAudioPlaying != lastAudioPlaying {
					clearTranscripts()
					lastState = currentState
					lastAudioPlaying = currentAudioPlaying
					continue
				}

				for _, result := range resp.Results {
					if len(result.Alternatives) == 0 {
						continue
					}
					
					transcript := result.Alternatives[0].Transcript
					
					// Check for interrupts during audio playback
					if currentAudioPlaying {
						at.mu.Lock()
						at.InterruptTranscript = transcript
						at.mu.Unlock()
						
						fmt.Printf("Checking Interrupt: %s\n", transcript)
						
						// Create a channel to synchronize interrupt handling
						interruptDone := make(chan struct{})
						
						go func() {
							if at.containsInterruptCommand(transcript) {
								
								// Stop audio immediately
								speaker.Clear()
								
								// Handle the interrupt
								at.handleVoiceInterrupt()
								
								// Clear transcripts
								clearTranscripts()
							}
							close(interruptDone)
						}()
						
						// Wait for interrupt handling to complete
						<-interruptDone
						
					} else if currentState == FULL_LISTENING {
						if result.IsFinal {
							at.handleFinalResult(transcript)
						} else {
							at.handleInterimResult(transcript)
						}
					}
				}
			}
		}
	}()

	// Wait for any errors or context cancellation
	select {
	case err := <-errChan:
		log.Printf("Error encountered: %v", err)
		return err
	case <-ctx.Done():
		log.Println("Context cancelled")
		return nil
	}
}

// handleAudioInput processes incoming audio data
func (at *AudioTranscriber) handleAudioInput(ctx context.Context,
	stream speechpb.Speech_StreamingRecognizeClient,
	createNewStream func() error,
	errChan chan error,
	doneChan chan struct{}) {

	for {
		select {
		case <-ctx.Done():
			return
		case <-doneChan:
			return
		case data, ok := <-at.AudioQueue:
			if !ok {
				return
			}

			// Only send audio data when in FULL_LISTENING state or checking for interrupts
			if at.getListeningState() == FULL_LISTENING || 
			   (at.getListeningState() == INTERRUPT_ONLY && at.isAudioPlaying()) {
				
				err := stream.Send(&speechpb.StreamingRecognizeRequest{
					StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{
						AudioContent: data,
					},
				})

				if err != nil {
					log.Printf("Error sending audio data: %v", err)
					if err := createNewStream(); err != nil {
						errChan <- err
						return
					}
				}
			}
		}
	}
}

// handleStreamResponses processes responses from the speech recognition stream
func (at *AudioTranscriber) handleStreamResponses(ctx context.Context,
	stream speechpb.Speech_StreamingRecognizeClient,
	createNewStream func() error,
	errChan chan error,
	doneChan chan struct{}) {

	for {
		select {
		case <-ctx.Done():
			return
		case <-doneChan:
			return
		default:
			resp, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				log.Printf("Error receiving from stream: %v", err)
				if err := createNewStream(); err != nil {
					errChan <- err
					return
				}
				continue
			}

			for _, result := range resp.Results {
				if result.IsFinal {
					at.handleFinalResult(result.Alternatives[0].Transcript)
				} else {
					at.handleInterimResult(result.Alternatives[0].Transcript)
				}
			}
		}
	}
}

// monitorStateChanges monitors for state changes that require stream recreation
func (at *AudioTranscriber) monitorStateChanges(ctx context.Context,
	createNewStream func() error,
	errChan chan error) {

	var lastState ListeningState
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			currentState := at.getListeningState()
			if currentState != lastState {
				// Print state change message
				fmt.Printf("\nListening .... (press ` or say \"shut up\" to interrupt)\n")
				fmt.Printf("Current State: %d, Audio Playing: %v\n", currentState, at.isAudioPlaying())

				// Recreate stream on state change
				if err := createNewStream(); err != nil {
					errChan <- err
					return
				}
				lastState = currentState
			}
		}
	}
}

func (at *AudioTranscriber) processResponses(responses []*speechpb.StreamingRecognizeResponse) {
	clearTranscripts := func() {
		at.mu.Lock()
		defer at.mu.Unlock()
		at.InterruptTranscript = ""
		at.InterimResult = ""
		at.FinalResult = ""
		at.LastTranscript = ""
		at.CurrentSentence = ""
		// Add a longer delay to ensure buffers are cleared
		time.Sleep(200 * time.Millisecond)
	}

	for _, response := range responses {
		if len(response.Results) == 0 {
			continue
		}

		at.mu.Lock()
		currentState := at.getListeningState()
		isPlaying := at.isAudioPlaying()
		stateChanged := currentState != at.lastState
		playingChanged := isPlaying != at.lastAudioPlaying
		at.mu.Unlock()

		// Always clear transcripts when state or playing status changes
		if stateChanged || playingChanged {
			clearTranscripts()
			at.lastState = currentState
			at.lastAudioPlaying = isPlaying
			// Skip this response to ensure clean state
			continue
		}

		// Additional clearing when transitioning from audio playback to listening
		if !isPlaying && at.lastAudioPlaying {
			clearTranscripts()
			at.lastAudioPlaying = isPlaying
			continue
		}

		result := response.Results[0]
		transcript := strings.TrimSpace(result.Alternatives[0].Transcript)
		
		if transcript == "" {
			continue
		}

		// Handle different states
		switch currentState {
		case INTERRUPT_ONLY:
			if isPlaying {
				// Clear before processing new interrupt check
				clearTranscripts()
				at.InterruptTranscript = transcript
				fmt.Printf("Interrupt Check: %s\n", transcript)
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
			} else {
				// Clear transcripts when audio is playing in FULL_LISTENING
				clearTranscripts()
			}
		}
	}
}












