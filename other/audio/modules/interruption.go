package modules

import (
	"fmt"
	"log"
	"regexp"
	"strings"

	"github.com/eiannone/keyboard"
	"github.com/faiface/beep/speaker"
)

// HandleInterrupts listens for keyboard events
func (at *AudioTranscriber) HandleInterrupts() {
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
			at.cancelFunc()
			return
		}

		if char == InterruptKeyChar {
			log.Println("Backtick interrupt detected!")
			at.handleKeyboardInterrupt()
			continue
		}
	}
}

// handleKeyboardInterrupt handles the backtick key interrupt
func (at *AudioTranscriber) handleKeyboardInterrupt() {
	fmt.Println("\nInterrupt: Backtick keyboard detected!")
	at.aggressiveClearTranscripts()
	at.handleInterrupt()
}

// handleVoiceInterrupt handles voice-based interrupts
func (at *AudioTranscriber) handleVoiceInterrupt() {
	fmt.Println("\nInterrupt: Voice command detected!")
	at.handleInterrupt()
}

// handleInterrupt handles both keyboard and voice interrupts
func (at *AudioTranscriber) handleInterrupt() {
	// Stop any ongoing audio
	speaker.Clear()
	
	// Set state to interrupt only
	at.setListeningState(INTERRUPT_ONLY)
	
	// Clear audio queue and reset state
	at.clearAudioQueue()
	
	// First round of aggressive transcript clearing
	at.aggressiveClearTranscripts()
	
	// Play acknowledgment and reset state
	at.playAcknowledgment()
	
	// Second round of aggressive clearing after acknowledgment
	at.aggressiveClearTranscripts()
	
	at.setListeningState(FULL_LISTENING)
	at.setAudioPlaying(false)

	// Reset audio stream if it exists
	if at.audioStream != nil {
		at.audioStream.Stop()
		at.audioStream.Start()
	}

	// Final round of aggressive clearing
	at.aggressiveClearTranscripts()
	
	// Add extra newlines for better visibility
	fmt.Printf("\nListening .... (press ` or say \"shut up\" to interrupt)\n")
}

// containsInterruptCommand checks if the transcript contains an interrupt command
func (at *AudioTranscriber) containsInterruptCommand(transcript string) bool {
	at.mu.Lock()
	defer at.mu.Unlock()

	log.Printf("DEBUG: Raw interrupt check transcript: '%s'", transcript)

	// Clean the text more aggressively
	re := regexp.MustCompile(`[^\w\s]`)
	cleanedText := re.ReplaceAllString(transcript, "")
	// Remove extra spaces (including leading/trailing) and convert to lowercase
	normalizedText := strings.Join(strings.Fields(strings.ToLower(cleanedText)), " ")

	log.Printf("DEBUG: Normalized interrupt check transcript: '%s'", normalizedText)

	// Check exact matches first
	if _, exists := at.interruptCommands[normalizedText]; exists {
		log.Printf("DEBUG: Found exact interrupt command match: '%s'", normalizedText)
		return true
	}

	return false
}

// playAcknowledgment plays the interruption acknowledgment audio
func (at *AudioTranscriber) playAcknowledgment() {
	at.setAudioPlaying(true)
	at.playAudioResponse(at.interruptAudioPath)
	at.setAudioPlaying(false)
	// Clear transcripts after acknowledgment
	at.aggressiveClearTranscripts()
}

// stopAudio interrupts any ongoing audio playback
func (at *AudioTranscriber) stopAudio() {
	speaker.Clear()
	// Clear transcripts after stopping audio
	at.aggressiveClearTranscripts()
}
