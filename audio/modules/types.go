package modules

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"time"

	speech "cloud.google.com/go/speech/apiv1"
	"cloud.google.com/go/speech/apiv1/speechpb"
	texttospeech "cloud.google.com/go/texttospeech/apiv1"
	tts "cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
	"github.com/gordonklaus/portaudio"
	"github.com/sashabaranov/go-openai"
)

type ListeningState int32

const (
	FULL_LISTENING ListeningState = iota
	INTERRUPT_ONLY
)

const (
	InterruptKeyChar       = '`'
	StateTransitionTimeout = 3 * time.Second
)

type StateTransition struct {
	From      ListeningState
	To        ListeningState
	Timestamp time.Time
}

type AudioTranscriber struct {
	// Audio parameters
	Rate            int
	Chunk           int
	CurrentSentence string
	LastTranscript  string
	InterruptTranscript string
	InterimResult   string
	FinalResult     string

	// Thread-safe queue for audio data
	AudioQueue chan []byte

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
	audioPlaying      bool
	audioPlayingMutex sync.Mutex

	// Transcript tracking
	lastInterimTimestamp time.Time
	interimCooldown      time.Duration

	// Interrupt handling
	lastInterruptTime   time.Time
	interruptCooldown   time.Duration

	// Paths
	interruptAudioPath string
	contextFolder      string
	randomFolder       string

	// Audio mappings
	contextFileMapping map[int]string
	randomFileNames    []string
	randomAudioFiles  []string
	usedRandomFiles   map[string]struct{}

	// Synchronization
	mu sync.Mutex

	// New fields for better control
	audioInputBuffer []float32
	processingLock   sync.Mutex
	interruptChan    chan struct{}

	// Audio state management
	isProcessingAudio bool
	audioStateMutex   sync.Mutex

	// Channel for coordinating audio playback
	audioPlaybackDone chan struct{}

	// Audio state management
	audioState struct {
		sync.Mutex
		lastSpeechTime    time.Time
		lastProcessedText string
		isProcessing      bool
		feedbackBuffer    []string
	}

	// Audio processing control
	processingControl struct {
		sync.Mutex
		noiseGate       float64
		volumeThreshold float64
		cooldownPeriod  time.Duration
	}

	// Cancellation function
	cancelFunc context.CancelFunc

	// State transitions
	stateTransitions chan StateTransition

	// Debugging
	debug struct {
		sync.Mutex
		events []struct {
			timestamp time.Time
			event     string
		}
	}

	// Cleanup synchronization
	cleanup struct {
		sync.Once
		done chan struct{}
	}

	// Sentence state management
	sentenceState struct {
		sync.Mutex
		currentSentence string
		lastUpdateTime  time.Time
		isTimerActive   bool
		timerExpired    bool
		pendingWords    []string
		isProcessing    bool
		baseTranscript  string
	}

	// Additional fields
	micVolume          float64
	noiseGateThreshold float64
	stopGeneration     bool
	pendingResponse    string
	muteThreshold     float64
}

// NewAudioTranscriber creates and initializes a new AudioTranscriber instance
func NewAudioTranscriber(ctx context.Context, speechClient *speech.Client, ttsClient *texttospeech.Client, openaiClient *openai.Client) (*AudioTranscriber, error) {
	ctx, transcriberCancel := context.WithCancel(ctx)
	
	at := &AudioTranscriber{
		Rate:         16000,
		Chunk:        800,
		AudioQueue:   make(chan []byte, 100),
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
		Model:        "gpt-4o-mini",
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
			"shut up":          {},
			"stop please":      {},
			"shut up please":   {},
			"please shut up":   {},
		},
		lastInterimTimestamp: time.Now(),
		interimCooldown:      500 * time.Millisecond,
		lastInterruptTime:    time.Now(),
		interruptCooldown:    2 * time.Second,
		interruptAudioPath:   "interruption.mp3",
		contextFileMapping:    map[int]string{
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
		cleanup: struct {
			sync.Once
			done chan struct{}
		}{
			done: make(chan struct{}),
		},
		stateTransitions: make(chan StateTransition, 100),
	}

	// Set audio folders
	at.contextFolder = filepath.Join("voiceCashingSys", "contextBased")
	at.randomFolder = filepath.Join("voiceCashingSys", "random")

	// Create directories if they don't exist
	os.MkdirAll(at.contextFolder, 0755)
	os.MkdirAll(at.randomFolder, 0755)

	return at, nil
}
