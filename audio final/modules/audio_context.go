package modules

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/faiface/beep"
	"github.com/faiface/beep/mp3"
	"github.com/faiface/beep/speaker"
)

// playContextAudio plays the context-based audio file
func (at *AudioTranscriber) playContextAudio(index int) bool {
	filename, exists := at.contextFileMapping[index]
	if !exists {
		return false
	}
	
	audioFile := filepath.Join(at.contextFolder, filename+".mp3")
	
	// Create a WaitGroup to ensure audio completes
	var wg sync.WaitGroup
	wg.Add(1)
	
	go func() {
		defer wg.Done()
		at.playAudioResponse(audioFile)
	}()
	
	// Wait for audio to complete with timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	
	select {
	case <-done:
		return true
	case <-time.After(10 * time.Second):
		log.Println("Context audio playback timed out")
		return false
	}
}

// playAudioResponse plays an audio file and manages state
func (at *AudioTranscriber) playAudioResponse(audioPath string) {
	// Set audio playing state
	at.setAudioPlaying(true)
	defer at.setAudioPlaying(false)

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

	// Initialize the speaker if needed
	if err := speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10)); err != nil {
		log.Printf("Error initializing speaker: %v", err)
		return
	}

	// Create done channel
	done := make(chan bool)

	// Play the audio
	speaker.Play(beep.Seq(streamer, beep.Callback(func() {
		done <- true
	})))

	// Wait for playback to finish
	<-done
}

// initSpeaker initializes the speaker with a sample audio file
func (at *AudioTranscriber) initSpeaker() error {
	sampleFile := filepath.Join(at.contextFolder, at.contextFileMapping[0]+".mp3")
	f, err := os.Open(sampleFile)
	if err != nil {
		return fmt.Errorf("error opening sample file: %v", err)
	}
	defer f.Close()

	_, format, err := mp3.Decode(f)
	if err != nil {
		return fmt.Errorf("error decoding sample file: %v", err)
	}

	return speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))
}
