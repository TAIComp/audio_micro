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

// handleVoiceInterrupt should be as simple as keyboard interrupt
func (at *AudioTranscriber) handleVoiceInterrupt() {
	fmt.Println("\nInterrupt: Voice command detected!")
	at.aggressiveClearTranscripts()
	at.handleInterrupt()
}

// handleInterrupt handles both keyboard and voice interrupts
func (at *AudioTranscriber) handleInterrupt() {
	// Stop any ongoing audio
	speaker.Clear()
	at.setListeningState(INTERRUPT_ONLY)
	at.clearAudioQueue()
	at.aggressiveClearTranscripts()
	
	// Play acknowledgment and reset state
	at.playAcknowledgment()
	at.aggressiveClearTranscripts()
	at.setListeningState(FULL_LISTENING)
	at.setAudioPlaying(false)

	// Reset audio stream if it exists
	if at.audioStream != nil {
		at.audioStream.Stop()
		at.audioStream.Start()
	}

	at.aggressiveClearTranscripts()
	
	fmt.Printf("\nListening .... (press ` or say \"shut up\" to interrupt)\n")
}

// containsInterruptCommand checks if the transcript contains an interrupt command
func (at *AudioTranscriber) containsInterruptCommand(transcript string) bool {

	re := regexp.MustCompile(`[^\w\s]`)
	cleanedText := re.ReplaceAllString(transcript, "")
	normalizedText := strings.ToLower(strings.TrimSpace(cleanedText))

	// Only check exact matches
	if _, exists := at.interruptCommands[normalizedText]; exists {
		fmt.Printf("EXACT MATCH FOUND: \"%s\"\n", normalizedText)
		return true
	}

	return false
}

// playAcknowledgment plays the interruption acknowledgment audio
func (at *AudioTranscriber) playAcknowledgment() {
	at.setAudioPlaying(true)
	at.playAudioResponse(at.interruptAudioPath)
	at.setAudioPlaying(false)
	at.aggressiveClearTranscripts()
}

