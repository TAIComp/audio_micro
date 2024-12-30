package modules

import (
	"fmt"
	"log"
	"strings"
	"time"
)

const (
	COMPLETE_WAIT   = 1 * time.Second
	INCOMPLETE_WAIT = 2 * time.Second
)

// handleInterimResult processes interim transcription results
func (at *AudioTranscriber) handleInterimResult(transcript string) {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	// Only update interim result in FULL_LISTENING state
	if at.listeningState == FULL_LISTENING && !at.audioPlaying {
		cleanTranscript := strings.TrimSpace(transcript)
		fmt.Printf("Full Listening - Interim: %s\n", cleanTranscript)
		at.InterimResult = cleanTranscript
		at.lastInterimTimestamp = time.Now()
	}
}

// handleFinalResult processes final transcription results
func (at *AudioTranscriber) handleFinalResult(transcript string) {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	// Only process final results in FULL_LISTENING state
	if at.listeningState == FULL_LISTENING && !at.audioPlaying {
		fmt.Printf("Full Listening - Final: %s\n", transcript)
		at.FinalResult = transcript

		// Get sentence completion status and context index
		contextIndex, isComplete := at.getContextAndCompletion(transcript)

		// Set waiting time based on completion status
		waitTime := INCOMPLETE_WAIT
		if isComplete {
			waitTime = COMPLETE_WAIT
			fmt.Println("Processing: Complete sentence detected")
		} else {
			fmt.Println("Processing: Incomplete sentence detected")
		}

		// Start or reset timer
		if !at.sentenceState.isTimerActive {
			at.sentenceState.isTimerActive = true
			
			// Move transcript clearing to processSentenceWithTimer
			go at.processSentenceWithTimer(transcript, contextIndex, waitTime)
		} else {
			// Reset timer
			at.sentenceState.lastUpdateTime = time.Now()
		}
	}
}

// processSentenceWithTimer handles the timing logic for sentence processing
func (at *AudioTranscriber) processSentenceWithTimer(text string, contextIndex int, waitTime time.Duration) {
	startTime := time.Now()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			at.sentenceState.Lock()
			if !at.sentenceState.isTimerActive {
				at.sentenceState.Unlock()
				return
			}

			elapsed := time.Since(startTime)
			remaining := waitTime - elapsed

			fmt.Printf("Waiting for more input: %.1f/%.1f seconds\n", 
				elapsed.Seconds(), waitTime.Seconds())

			if remaining <= 0 {
				// Timer expired, process the sentence
				at.sentenceState.isTimerActive = false
				at.sentenceState.timerExpired = true
				at.sentenceState.Unlock()

				// Clear transcripts after timer expires but before processing
				at.aggressiveClearTranscripts()

				// Process the completed sentence
				at.processCompletedSentence(text, contextIndex)
				return
			}

			// Check if there's been new input
			if time.Since(at.lastInterimTimestamp) < elapsed {
				at.sentenceState.isTimerActive = false
				at.sentenceState.Unlock()
				return
			}

			at.sentenceState.Unlock()
		}
	}
}

// processCompletedSentence handles the completed sentence
func (at *AudioTranscriber) processCompletedSentence(text string, contextIndex int) {
	// Change state to interrupt only
	at.setListeningState(INTERRUPT_ONLY)
	fmt.Printf("Current State: %d, Audio Playing: %v\n", at.getListeningState(), at.isAudioPlaying())

	// Aggressive clearing before processing
	at.clearTranscripts()
	time.Sleep(100 * time.Millisecond)
	at.clearTranscripts()

	// Play context audio and generate GPT response simultaneously
	responseChan := make(chan string)
	go func() {
		response := at.getAIResponse(text)
		responseChan <- response
	}()

	// Play context audio while waiting for GPT response
	at.playContextAudio(contextIndex)
	fmt.Printf("Processing responses - State: %d, Audio Playing: %v\n", at.getListeningState(), at.isAudioPlaying())

	// Get GPT response and convert to speech
	response := <-responseChan
	fmt.Println(response)

	// Clear transcripts again before playing audio
	at.clearTranscripts()
	time.Sleep(100 * time.Millisecond)
	at.clearTranscripts()

	// Convert response to speech and play it
	audioData, err := at.textToSpeech(response)
	if err != nil {
		log.Printf("Error converting text to speech: %v", err)
		return
	}

	if err := at.streamAudioResponse(audioData); err != nil {
		log.Printf("Error streaming audio response: %v", err)
	}

	// Reset state and prepare for next input with aggressive clearing
	at.resetState()
	time.Sleep(200 * time.Millisecond)
	at.clearTranscripts()
	at.setListeningState(FULL_LISTENING)
	
	fmt.Printf("\nListening .... (press ` or say \"shut up\" to interrupt)\n")
	fmt.Printf("Current State: %d, Audio Playing: %v\n", at.getListeningState(), at.isAudioPlaying())
}

// Audio state management
func (at *AudioTranscriber) setAudioPlaying(playing bool) {
	at.audioPlayingMutex.Lock()
	defer at.audioPlayingMutex.Unlock()
	at.audioPlaying = playing
}

func (at *AudioTranscriber) isAudioPlaying() bool {
	at.audioPlayingMutex.Lock()
	defer at.audioPlayingMutex.Unlock()
	return at.audioPlaying
}

// Listening state management
func (at *AudioTranscriber) setListeningState(state ListeningState) {
	at.mu.Lock()
	defer at.mu.Unlock()
	at.listeningState = state
}

func (at *AudioTranscriber) getListeningState() ListeningState {
	at.mu.Lock()
	defer at.mu.Unlock()
	return at.listeningState
}

// Audio queue management
func (at *AudioTranscriber) clearAudioQueue() {
	for len(at.AudioQueue) > 0 {
		<-at.AudioQueue
	}
}

// Add this helper function to AudioTranscriber
func (at *AudioTranscriber) clearTranscripts() {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	at.InterimResult = ""
	at.FinalResult = ""
	at.InterruptTranscript = ""
	at.LastTranscript = ""
	at.CurrentSentence = ""
	
	// Add a small delay to ensure buffers are cleared
	time.Sleep(100 * time.Millisecond)
}
