import Foundation
import Observation
import os

import LeapSDK
import LeapModelDownloader

/// Centralized loggers for the app.
enum Log {
    static let model = Logger(subsystem: "com.thinktecture.genai.LocalLife", category: "Model")
    static let agent = Logger(subsystem: "com.thinktecture.genai.LocalLife", category: "Agent")
    static let chat = Logger(subsystem: "com.thinktecture.genai.LocalLife", category: "Chat")
}

public enum ModelState: Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case downloaded
    case loading
    case ready
    case error(String)
}

enum ModelError: Error {
    case notLoaded
}

@Observable
public final class ModelManager {
    public var state: ModelState = .notDownloaded

    /// HuggingFace model coordinates for LFM2.5-1.2B-Instruct GGUF.
    private let hfOwner = "LiquidAI"
    private let hfRepo = "LFM2.5-1.2B-Instruct-GGUF"
    private let hfFilename = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"

    private var modelRunner: (any ModelRunner)?
    private(set) var conversation: Conversation?

    private var downloader: ModelDownloader?
    private var modelURL: URL?

    /// The model descriptor — constructed directly, no network resolve needed.
    private var downloadableModel: HuggingFaceDownloadableModel {
        HuggingFaceDownloadableModel(
            ownerName: hfOwner,
            repoName: hfRepo,
            filename: hfFilename
        )
    }

    /// Lazily create the ModelDownloader off the cooperative pool to avoid
    /// the SDK's internal `unsafeForcedSync` blocking a Swift Concurrency thread.
    private func ensureDownloader() async -> ModelDownloader {
        if let downloader { return downloader }
        let dl = await Task.detached { ModelDownloader() }.value
        self.downloader = dl
        return dl
    }

    public init() {}

    /// Check if model files are already cached on disk.
    public func checkModelAvailability() async {
        let dl = await ensureDownloader()
        let model = downloadableModel

        let status = await Task.detached {
            await dl.queryStatus(model)
        }.value
        Log.model.debug("queryStatus: \(String(describing: status))")
        switch status {
        case .downloaded:
            self.modelURL = dl.getModelFile(model)
            Log.model.info("Model cached at: \(self.modelURL?.path ?? "nil")")
            state = .downloaded
        case .downloadInProgress(let progress):
            Log.model.info("Partial download: \(Int(progress * 100))%")
            state = .downloading(progress: progress)
        case .notOnLocal:
            state = .notDownloaded
        @unknown default:
            Log.model.warning("Unknown download status: \(String(describing: status))")
            state = .notDownloaded
        }
    }

    /// Download model from HuggingFace with progress polling.
    func downloadModel() async {
        state = .downloading(progress: 0.0)
        let dl = await ensureDownloader()
        let model = downloadableModel

        // Fire off the download (non-blocking)
        dl.requestDownloadModel(model)

        // Poll for progress until complete, with cancellation support
        // and a guard against indefinite .notOnLocal states.
        var notOnLocalCount = 0
        let maxNotOnLocalPolls = 60 // 30 seconds of no activity before giving up

        while !Task.isCancelled {
            let status = await Task.detached {
                await dl.queryStatus(model)
            }.value

            switch status {
            case .downloaded:
                self.modelURL = dl.getModelFile(model)
                Log.model.info("Download complete: \(self.modelURL?.path ?? "")")
                state = .downloaded
                return
            case .downloadInProgress(let progress):
                notOnLocalCount = 0
                state = .downloading(progress: progress)
            case .notOnLocal:
                notOnLocalCount += 1
                if notOnLocalCount >= maxNotOnLocalPolls {
                    Log.model.error("Download stalled — no progress after \(maxNotOnLocalPolls) polls")
                    state = .error("Download failed — no progress detected")
                    return
                }
            @unknown default:
                break
            }

            try? await Task.sleep(for: .milliseconds(500))
        }
    }

    /// Load model into memory and create conversation.
    public func loadModel(systemPrompt: String) async throws {
        state = .loading

        guard let modelURL else {
            Log.model.error("modelURL is nil — cannot load model")
            throw ModelError.notLoaded
        }

        // Leap.load is synchronous and CPU-intensive — run off the cooperative pool.
        Log.model.info("Loading model from: \(modelURL.path)")
        let path = modelURL.path
        let runner = try await Task.detached {
            let options = LiquidInferenceEngineOptions(bundlePath: path)
            return try Leap.load(options: options)
        }.value
        modelRunner = runner
        conversation = runner.createConversation(systemPrompt: systemPrompt)

        Log.model.info("Model loaded successfully")
        state = .ready
    }

    public var isReady: Bool {
        state == .ready
    }

    /// Registered tool functions — kept so we can re-register on conversation reset.
    private var registeredFunctions: [LeapFunction] = []

    /// Register a tool function with the LEAP conversation.
    public func registerFunction(_ function: LeapFunction) {
        registeredFunctions.append(function)
        conversation?.registerFunction(function)
    }

    /// Create a fresh conversation with the same system prompt and tools.
    /// Call this before each agent loop run to guarantee zero history bleed.
    public func resetConversation() {
        guard let runner = modelRunner else { return }
        let conv = runner.createConversation(systemPrompt: AgentConfiguration.systemPrompt)
        for fn in registeredFunctions {
            conv.registerFunction(fn)
        }
        self.conversation = conv
    }

    /// Generate a response (streaming).
    func generateResponse(message: String, options: GenerationOptions) -> AsyncThrowingStream<MessageResponse, Error> {
        guard let conversation else {
            return AsyncThrowingStream { $0.finish(throwing: ModelError.notLoaded) }
        }
        return conversation.generateResponse(userTextMessage: message, generationOptions: options)
    }
}
