package modules

import (
	"fmt"
	"log"
	"time"
)

// Cleanup cleans up resources when the transcriber is done
func (at *AudioTranscriber) Cleanup() {
	at.cleanup.Once.Do(func() {
		close(at.cleanup.done)
		at.stopAudio()
		at.clearAudioQueue()
		at.resetState()
		
		// Clean up clients
		if at.speechClient != nil {
			at.speechClient.Close()
		}
		if at.ttsClient != nil {
			at.ttsClient.Close()
		}
	})
}

// resetState resets the transcription state
func (at *AudioTranscriber) resetState() {
	at.mu.Lock()
	defer at.mu.Unlock()

	at.CurrentSentence = ""
	at.LastTranscript = ""
	at.lastInterimTimestamp = time.Now()
	at.lastInterruptTime = time.Now()
	at.InterruptTranscript = ""
	at.InterimResult = ""
	at.FinalResult = ""

	at.sentenceState.Lock()
	at.sentenceState.currentSentence = ""
	at.sentenceState.pendingWords = nil
	at.sentenceState.isTimerActive = false
	at.sentenceState.timerExpired = false
	at.sentenceState.Unlock()

	at.audioState.Lock()
	at.audioState.lastProcessedText = ""
	at.audioState.isProcessing = false
	at.audioState.feedbackBuffer = make([]string, 0, 10)
	at.audioState.Unlock()

	at.clearAudioQueue()
}

// logDebugEvent logs debug events with timestamps
func (at *AudioTranscriber) logDebugEvent(event string) {
	at.debug.Lock()
	defer at.debug.Unlock()
	
	at.debug.events = append(at.debug.events, struct {
		timestamp time.Time
		event     string
	}{
		timestamp: time.Now(),
		event:     event,
	})
	
	if len(at.debug.events) > 100 {
		at.debug.events = at.debug.events[1:]
	}
	
	log.Printf("Debug: %s", event)
}

// logAudioStateChange logs audio state changes
func (at *AudioTranscriber) logAudioStateChange(state string, playing bool) {
	at.logDebugEvent(fmt.Sprintf("Audio State Change: %s, Playing: %v", state, playing))
}
