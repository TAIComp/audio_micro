package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"syscall"
	"time"

	"audio/modules"

	speech "cloud.google.com/go/speech/apiv1"
	texttospeech "cloud.google.com/go/texttospeech/apiv1"
	"github.com/gordonklaus/portaudio"
	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/api/option"
)

func main() {
	// Define suppressAlsaWarnings function inside main
	suppressAlsaWarnings := func() {
		// Create a pipe
		reader, writer, err := os.Pipe()
		if err != nil {
			log.Printf("Failed to create pipe: %v", err)
			return
		}

		// Save the original stderr
		stderr := os.Stderr

		// Redirect stderr to the pipe
		os.Stderr = writer
		
		// Close the original stderr
		stderr.Close()

		// Redirect stderr at the syscall level
		syscall.Dup2(int(writer.Fd()), int(stderr.Fd()))

		// Start a goroutine to drain the pipe
		go func() {
			buffer := make([]byte, 1024)
			for {
				_, err := reader.Read(buffer)
				if err != nil {
					return
				}
			}
		}()
	}

	// Call suppressAlsaWarnings at the start
	suppressAlsaWarnings()

	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("Failed to initialize PortAudio: %v", err)
	}
	defer portaudio.Terminate()

	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Fatalf("Error loading .env file: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Use your existing service account key file
	credentialsFile := "tutorai-438115-f72989e460df.json"

	// Check if credentials file exists
	if _, err := os.Stat(credentialsFile); os.IsNotExist(err) {
		log.Fatalf("Credentials file %s not found", credentialsFile)
	}

	// Initialize Speech-to-Text client
	speechClient, err := speech.NewClient(ctx, option.WithCredentialsFile(credentialsFile))
	if err != nil {
		log.Fatalf("Failed to create speech client: %v", err)
	}
	defer speechClient.Close()

	// Initialize Text-to-Speech client
	ttsClient, err := texttospeech.NewClient(ctx, option.WithCredentialsFile(credentialsFile))
	if err != nil {
		log.Fatalf("Failed to create TTS client: %v", err)
	}
	defer ttsClient.Close()

	// Initialize OpenAI client
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}
	openaiClient := openai.NewClient(openaiKey)

	// Create new AudioTranscriber instance
	transcriber, err := modules.NewAudioTranscriber(ctx, speechClient, ttsClient, openaiClient)
	if err != nil {
		log.Fatalf("Failed to create audio transcriber: %v", err)
	}
	defer transcriber.Cleanup()

	// Start keyboard interrupt handler in a goroutine
	go transcriber.HandleInterrupts()

	// Print initial listening message
	fmt.Printf("\nListening .... (press ` or say \"shut up\" to interrupt)\n")

	// Start audio processing in a goroutine
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				if err := transcriber.ProcessAudioStream(ctx); err != nil {
					log.Printf("Error processing audio stream: %v", err)
					time.Sleep(time.Second) // Wait before retrying
				}
			}
		}
	}()

	// Wait for program termination
	<-ctx.Done()
}
 