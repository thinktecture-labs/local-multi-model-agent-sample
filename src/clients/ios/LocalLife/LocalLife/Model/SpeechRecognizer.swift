import Foundation
import Speech
import AVFoundation

@MainActor @Observable
final class SpeechRecognizer {
    var transcript: String = ""
    var isListening: Bool = false
    var isAvailable: Bool = false
    var error: String? = nil

    private var recognizer: SFSpeechRecognizer?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private let audioEngine = AVAudioEngine()

    init(locale: Locale = Locale.current) {
        recognizer = SFSpeechRecognizer(locale: locale)
        recognizer?.supportsOnDeviceRecognition = true
        isAvailable = recognizer?.isAvailable ?? false
    }

    /// Request microphone + speech recognition permissions.
    func requestAuthorization() async -> Bool {
        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }

        guard speechStatus == .authorized else {
            error = "Speech recognition not authorized"
            return false
        }

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            self.error = "Audio session error: \(error.localizedDescription)"
            return false
        }

        isAvailable = true
        return true
    }

    /// Start listening. Transcription streams into `transcript` in real time.
    func startListening() {
        guard let recognizer, recognizer.isAvailable else {
            error = "Speech recognizer not available"
            return
        }

        // Cancel any ongoing task
        stopListening()

        transcript = ""
        error = nil

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true // Force on-device for airplane mode

        self.recognitionRequest = request

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            request.append(buffer)
        }

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }

            MainActor.assumeIsolated {
                if let result {
                    self.transcript = result.bestTranscription.formattedString
                }

                if let error {
                    // Ignore cancellation errors
                    let nsError = error as NSError
                    if nsError.domain != "kAFAssistantErrorDomain" || nsError.code != 216 {
                        self.error = error.localizedDescription
                    }
                    self.stopListening()
                }

                if result?.isFinal == true {
                    self.stopListening()
                }
            }
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
            isListening = true
        } catch {
            self.error = "Audio engine error: \(error.localizedDescription)"
            stopListening()
        }
    }

    /// Stop listening and finalize transcription.
    func stopListening() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        isListening = false
    }
}
