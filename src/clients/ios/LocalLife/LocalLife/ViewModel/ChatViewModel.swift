import EventKit
import Foundation
import Observation
import os

@MainActor @Observable
final class ChatViewModel {
    // MARK: - UI State
    var messages: [ChatMessage] = []
    var inputText: String = ""
    var isGenerating: Bool = false
    var isOffline: Bool = false
    var activeToolCalls: [ToolCallInfo] = []
    var modelReady: Bool = false
    var showModelDownload: Bool = false
    var downloadProgress: Double? = nil
    var isLoadingModel: Bool = false
    var demoMode: Bool = true // HealthKit demo data by default

    // MARK: - Configuration
    var calendarName: String = "SDD Demo" // Configurable calendar filter

    // MARK: - Speech
    let speechRecognizer = SpeechRecognizer()
    var isListening: Bool { speechRecognizer.isListening }

    // MARK: - Dependencies
    private let modelManager = ModelManager()
    private let toolRegistry = ToolRegistry()
    private var agentLoop: AgentLoop?
    private let connectivityMonitor = ConnectivityMonitor()

    func setup() async {
        // When running inside a test host process, skip model loading so
        // that tests can manage their own ModelManager instance without
        // competing for GPU/Metal resources.
        if ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil {
            Log.chat.info("Running inside test host — skipping app model setup")
            return
        }

        isLoadingModel = true

        connectivityMonitor.start { [weak self] isConnected in
            self?.isOffline = !isConnected
        }

        // Set up speech recognition
        _ = await speechRecognizer.requestAuthorization()

        // Register tools (share a single EKEventStore per Apple guidelines)
        let sharedEventStore = EKEventStore()
        let healthKit = HealthKitTool()
        healthKit.demoMode = demoMode
        let calendar = CalendarTool(eventStore: sharedEventStore)
        calendar.calendarFilter = calendarName
        let reminders = RemindersTool(eventStore: sharedEventStore)

        toolRegistry.register(healthKit)
        toolRegistry.register(calendar)
        toolRegistry.register(reminders)

        // Request permissions (non-fatal if denied)
        try? await healthKit.requestAuthorization()
        _ = try? await calendar.requestAccess()
        _ = try? await reminders.requestAccess()

        // Check model availability and load
        await modelManager.checkModelAvailability()
        Log.chat.debug("After checkModelAvailability, state=\(String(describing: self.modelManager.state))")
        if case .downloaded = modelManager.state {
            Log.chat.info("Model is downloaded, loading...")
            isLoadingModel = true
            await loadAndWireModel()
            isLoadingModel = false
        } else {
            // Not downloaded or only partially downloaded — show download UI
            Log.chat.info("Model not ready, showing download UI")
            isLoadingModel = false
            showModelDownload = true
        }
    }

    /// Toggle voice input. Tap to start, tap again to stop and send.
    func toggleVoiceInput() {
        if speechRecognizer.isListening {
            speechRecognizer.stopListening()
            // Transfer transcript to input and auto-send
            let transcript = speechRecognizer.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            if !transcript.isEmpty {
                inputText = transcript
                Task { await sendMessage() }
            }
        } else {
            speechRecognizer.startListening()
        }
    }

    func downloadModel() async {
        // Start a background task to poll modelManager.state for progress updates
        let progressTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { break }
                if case .downloading(let progress) = self.modelManager.state {
                    self.downloadProgress = progress
                }
                try? await Task.sleep(for: .milliseconds(300))
            }
        }

        await modelManager.downloadModel()
        progressTask.cancel()
        downloadProgress = nil

        if case .downloaded = modelManager.state {
            // Dismiss the download overlay immediately, then load in background
            showModelDownload = false
            isLoadingModel = true
            await loadAndWireModel()
            isLoadingModel = false
        } else {
            let notice = ChatMessage(
                role: .assistant,
                content: "Download failed. Please check your connection and try again.",
                timestamp: Date(),
                toolCalls: nil,
                isStreaming: false
            )
            messages.append(notice)
        }
    }

    func sendMessage() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating, modelReady else { return }

        inputText = ""
        isGenerating = true
        activeToolCalls = []

        let userMessage = ChatMessage(
            role: .user, content: text, timestamp: Date(),
            toolCalls: nil, isStreaming: false
        )
        messages.append(userMessage)

        // Placeholder assistant message for streaming
        let assistantMessage = ChatMessage(
            role: .assistant, content: "", timestamp: Date(),
            toolCalls: nil, isStreaming: true
        )
        messages.append(assistantMessage)
        let assistantIndex = messages.count - 1

        if let agentLoop {
            await runAgentLoop(agentLoop: agentLoop, text: text, at: assistantIndex)
        } else {
            // Fallback to simulation if LEAP SDK not loaded
            await simulateAgentResponse(for: text, at: assistantIndex)
        }

        isGenerating = false
    }

    func clearChat() {
        messages = []
        activeToolCalls = []
    }

    func toggleDemoMode() {
        demoMode.toggle()
        // Update the HealthKit tool's demo mode
        if let healthKit = toolRegistry.get("query_health_data") as? HealthKitTool {
            healthKit.demoMode = demoMode
        }
        let mode = demoMode ? "ON" : "OFF"
        let notice = ChatMessage(
            role: .assistant,
            content: "Demo mode: \(mode)",
            timestamp: Date(),
            toolCalls: nil,
            isStreaming: false
        )
        messages.append(notice)
    }

    // MARK: - Private

    private func loadAndWireModel() async {
        do {
            try await modelManager.loadModel(systemPrompt: AgentConfiguration.systemPrompt)
            // Register tool functions with LEAP conversation
            for fn in toolRegistry.asLeapFunctions() {
                modelManager.registerFunction(fn)
            }
            agentLoop = AgentLoop(toolRegistry: toolRegistry, modelManager: modelManager)
            modelReady = true
        } catch {
            let notice = ChatMessage(
                role: .assistant,
                content: "Failed to load model: \(error.localizedDescription)",
                timestamp: Date(),
                toolCalls: nil,
                isStreaming: false
            )
            messages.append(notice)
        }
    }

    /// Regex to strip leaked tool call syntax like [tool_name(args)] from model output.
    private static let toolCallSyntaxPattern = try! NSRegularExpression(
        pattern: "\\[\\w+\\([^\\]]*\\)(?:,\\s*\\w+\\([^\\]]*\\))*\\]\\s*"
    )

    private func runAgentLoop(agentLoop: AgentLoop, text: String, at index: Int) async {
        do {
            let finalText = try await agentLoop.run(message: text) { [weak self] event in
                guard let self else { return }
                MainActor.assumeIsolated {
                    switch event {
                    case .textChunk(let chunk):
                        // Accumulate into content, then strip any tool call syntax
                        self.messages[index].content += chunk
                        let content = self.messages[index].content
                        let range = NSRange(content.startIndex..., in: content)
                        let cleaned = Self.toolCallSyntaxPattern.stringByReplacingMatches(
                            in: content, range: range, withTemplate: ""
                        )
                        if cleaned != content {
                            self.messages[index].content = cleaned
                        }

                    case .toolCallsReceived(let names):
                        // Keep any already-completed tools from prior rounds,
                        // then add the new round's tools.
                        let completed = self.activeToolCalls.filter { $0.status.isFinished }
                        let newCalls = names.map {
                            ToolCallInfo(toolName: $0, status: .executing, durationMs: nil, resultSummary: nil, arguments: nil, resultData: nil)
                        }
                        self.activeToolCalls = completed + newCalls

                    case .toolExecuting(let name):
                        if let idx = self.activeToolCalls.firstIndex(where: { $0.toolName == name && $0.status == .pending }) {
                            self.activeToolCalls[idx] = ToolCallInfo(
                                toolName: name, status: .executing, durationMs: nil, resultSummary: nil, arguments: nil, resultData: nil
                            )
                        }

                    case .toolCompleted(let name, let success, let summary, let ms, let arguments, let resultData):
                        if let idx = self.activeToolCalls.firstIndex(where: { $0.toolName == name && $0.status == .executing }) {
                            self.activeToolCalls[idx] = ToolCallInfo(
                                toolName: name,
                                status: success ? .completed : .failed(summary),
                                durationMs: ms,
                                resultSummary: summary,
                                arguments: arguments,
                                resultData: resultData
                            )
                        }

                    case .generationComplete:
                        break

                    case .warning(let msg):
                        Log.agent.warning("Agent warning: \(msg)")
                    }
                }
            }
            messages[index].content = finalText
            messages[index].isStreaming = false
            messages[index].toolCalls = activeToolCalls
        } catch {
            messages[index].content = "Error: \(error.localizedDescription)"
            messages[index].isStreaming = false
        }
        activeToolCalls = []
    }

    // MARK: - Simulated agent response (fallback when LEAP SDK not available)

    private func simulateAgentResponse(for query: String, at index: Int) async {
        let lowered = query.lowercased()
        let isMeetingQuery = lowered.contains("meeting") || lowered.contains("muller")
            || lowered.contains("müller") || lowered.contains("prepare")

        if isMeetingQuery {
            // Simulate 3 tool calls
            activeToolCalls = [
                ToolCallInfo(toolName: "search_calendar", status: .executing, durationMs: nil, resultSummary: nil, arguments: nil, resultData: nil)
            ]
            try? await Task.sleep(for: .milliseconds(400))
            activeToolCalls[0] = ToolCallInfo(
                toolName: "search_calendar", status: .completed, durationMs: 45,
                resultSummary: "1 event: Dr. Muller - Cardiology, tomorrow 14:00",
                arguments: ["query": "muller", "days_ahead": "7"],
                resultData: "{\"count\":1,\"events\":[{\"title\":\"Dr. Muller - Cardiology\",\"start_time\":\"14:00\",\"location\":\"Klinikum Stuttgart\"}]}"
            )

            activeToolCalls.append(ToolCallInfo(toolName: "query_health_data", status: .executing, durationMs: nil, resultSummary: nil, arguments: nil, resultData: nil))
            try? await Task.sleep(for: .milliseconds(300))
            activeToolCalls[1] = ToolCallInfo(
                toolName: "query_health_data", status: .completed, durationMs: 32,
                resultSummary: "heart_rate: avg 72 bpm, 30 days",
                arguments: ["metric": "heart_rate", "days": "30"],
                resultData: "{\"metric\":\"heart_rate\",\"average\":72,\"unit\":\"bpm\",\"samples\":30,\"trend\":\"slightly_increasing\"}"
            )

            activeToolCalls.append(ToolCallInfo(toolName: "search_reminders", status: .executing, durationMs: nil, resultSummary: nil, arguments: nil, resultData: nil))
            try? await Task.sleep(for: .milliseconds(250))
            activeToolCalls[2] = ToolCallInfo(
                toolName: "search_reminders", status: .completed, durationMs: 18,
                resultSummary: "5 reminders in 'Health Questions'",
                arguments: ["query": "health"],
                resultData: "{\"count\":5,\"reminders\":[{\"title\":\"Review blood pressure readings\"},{\"title\":\"Heart rate variability\"},{\"title\":\"Exercise routine\"},{\"title\":\"Metoprolol alternatives\"},{\"title\":\"Elevated HR Feb 15-18\"}]}"
            )

            // Simulate streaming response
            let response = """
            Tomorrow at 2:00 PM you have an appointment with Dr. Muller at Klinikum Stuttgart (Cardiology).

            Your health summary over the past 30 days:
            - Resting heart rate: avg 72 bpm (up from 65 three months ago)
            - Blood pressure: 138/85 mmHg, stable
            - Daily steps: avg 6,500 (below your 8,000 target)

            You have 5 reminders in your "Health Questions" list:
            - Review blood pressure readings
            - Ask about heart rate variability with current medication
            - Discuss exercise routine adjustments
            - Switching from Metoprolol to newer beta-blocker
            - Elevated resting heart rate Feb 15-18

            Would you like me to compile these into a summary for Dr. Muller?
            """

            for char in response {
                messages[index].content.append(char)
                if char == "\n" || char == "." || char == ":" {
                    try? await Task.sleep(for: .milliseconds(15))
                }
            }
            messages[index].isStreaming = false
            messages[index].toolCalls = activeToolCalls
        } else {
            // Simple direct answer
            let response = "I'm LocalLife, your on-device personal AI agent. I can access your Calendar, HealthKit data, and Reminders to help you prepare for appointments, review health trends, and stay organized. Everything runs locally on this device — no data leaves your phone.\n\nTry asking: \"Prepare me for my meeting with Dr. Muller tomorrow\""

            for char in response {
                messages[index].content.append(char)
                if char == "." || char == "\n" {
                    try? await Task.sleep(for: .milliseconds(10))
                }
            }
            messages[index].isStreaming = false
        }

        activeToolCalls = []
    }
}
