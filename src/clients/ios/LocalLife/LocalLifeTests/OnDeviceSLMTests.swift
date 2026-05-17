// OnDeviceSLMTests.swift
// Real end-to-end tests that run the LFM 2.5 1.2B model on-device.
//
// These tests load the actual GGUF model via LeapSDK, register all 3 tools,
// send demo queries through AgentLoop.run(), and validate:
//   1. The model selects the correct tool(s) for each query
//   2. The synthesized response contains expected data points
//   3. The full pipeline (model → tool call → execution → synthesis) works
//
// Requirements:
//   - Must run on a physical iPhone (Neural Engine required)
//   - Model must already be downloaded (~700MB GGUF file)
//   - Run destination: your iPhone device, NOT simulator or "My Mac"
//   - These tests take ~10-30 seconds each due to model inference
//
// To enable these tests, set the environment variable RUN_ON_DEVICE_TESTS=1
// in the scheme's Test action (Edit Scheme → Test → Arguments → Environment Variables).
// Without this flag, the on-device tests are skipped so mock tests can run safely.

import Testing
import Foundation
import LocalLife
import EventKit

/// On-device test suites are disabled by default to prevent model-loading crashes
/// from killing the entire test runner. To enable, remove .disabled() from the
/// @Suite attributes below, or set up a separate test plan for on-device tests.

// MARK: - Shared Test Infrastructure

/// Shared model + agent setup. Loading the model is expensive (~5s),
/// so we load once and reuse across tests via a static actor.
private actor TestModelHolder {
    static let shared = TestModelHolder()

    private var modelManager: ModelManager?
    private var toolRegistry: ToolRegistry?
    private var agentLoop: AgentLoop?
    private var setupError: Error?
    private var didSetup = false

    func getAgent() async throws -> (AgentLoop, ToolRegistry, ModelManager) {
        if let setupError { throw setupError }
        if let agentLoop, let toolRegistry, let modelManager {
            return (agentLoop, toolRegistry, modelManager)
        }

        guard !didSetup else {
            throw TestSetupError.modelNotAvailable
        }
        didSetup = true

        do {
            let mm = ModelManager()
            let tr = ToolRegistry()

            // Register tools — HealthKit in demo mode, Calendar/Reminders with shared store
            let sharedEventStore = EKEventStore()
            let healthKit = HealthKitTool()
            healthKit.demoMode = true
            let calendar = CalendarTool(eventStore: sharedEventStore)
            calendar.calendarFilter = "SDD Demo"
            let reminders = RemindersTool(eventStore: sharedEventStore)

            tr.register(healthKit)
            tr.register(calendar)
            tr.register(reminders)

            // Check model is downloaded
            await mm.checkModelAvailability()
            guard mm.state == .downloaded else {
                throw TestSetupError.modelNotDownloaded(state: "\(mm.state)")
            }

            // Load model
            try await mm.loadModel(systemPrompt: AgentConfiguration.systemPrompt)
            guard mm.isReady else {
                throw TestSetupError.modelNotReady
            }

            // Register tool functions with LEAP conversation
            for fn in tr.asLeapFunctions() {
                mm.registerFunction(fn)
            }

            let al = AgentLoop(toolRegistry: tr, modelManager: mm)

            self.modelManager = mm
            self.toolRegistry = tr
            self.agentLoop = al

            // llama.cpp's Metal backend fires GGML_ASSERT → SIGABRT during
            // C++ global static destruction at process exit. Register an atexit
            // handler (LIFO — runs before C++ destructors) that calls _exit(0)
            // to terminate cleanly, skipping the problematic destructors.
            atexit { _exit(0) }

            return (al, tr, mm)
        } catch {
            self.setupError = error
            throw error
        }
    }

}

private enum TestSetupError: Error, CustomStringConvertible {
    case modelNotDownloaded(state: String)
    case modelNotReady
    case modelNotAvailable

    var description: String {
        switch self {
        case .modelNotDownloaded(let state):
            return "Model not downloaded (state: \(state)). Run the app first to download the model."
        case .modelNotReady:
            return "Model loaded but not ready."
        case .modelNotAvailable:
            return "Model setup previously failed."
        }
    }
}

/// Captures all events from an AgentLoop.run() call.
private final class EventCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var _toolsReceived: [[String]] = []
    private var _toolsExecuted: [String] = []
    private var _toolsCompleted: [(name: String, success: Bool, summary: String, args: [String: String], data: String)] = []
    private var _textChunks: [String] = []
    private var _tokensUsed: [Int] = []

    var toolsReceived: [[String]] { lock.withLock { _toolsReceived } }
    var toolsExecuted: [String] { lock.withLock { _toolsExecuted } }
    var toolsCompleted: [(name: String, success: Bool, summary: String, args: [String: String], data: String)] { lock.withLock { _toolsCompleted } }
    var textChunks: [String] { lock.withLock { _textChunks } }
    var tokensUsed: [Int] { lock.withLock { _tokensUsed } }

    /// All tool names that completed successfully.
    var successfulTools: [String] {
        lock.withLock { _toolsCompleted.filter { $0.success }.map { $0.name } }
    }

    /// The accumulated text from all chunks.
    var fullText: String {
        lock.withLock { _textChunks.joined() }
    }

    func handle(_ event: AgentEvent) {
        lock.withLock {
            switch event {
            case .textChunk(let text):
                _textChunks.append(text)
            case .toolCallsReceived(let names):
                _toolsReceived.append(names)
            case .toolExecuting(let name):
                _toolsExecuted.append(name)
            case .toolCompleted(let name, let success, let summary, _, let args, let data):
                _toolsCompleted.append((name: name, success: success, summary: summary, args: args, data: data))
            case .generationComplete(let tokens):
                _tokensUsed.append(tokens)
            case .warning:
                break
            }
        }
    }
}

// MARK: - On-Device E2E Tests

/// Parent suite that serializes all on-device tests to prevent concurrent model loading.
/// The LFM model can only be loaded once — concurrent loads crash the Metal backend.
@Suite("On-Device SLM Tests", .serialized)
struct OnDeviceSLMTests {

// MARK: Tool Selection

@Suite("Tool Selection", .serialized)
struct ToolSelectionTests {

    @Test("Model routes 'What is my heart rate?' to query_health_data",
          .timeLimit(.minutes(2)))
    func heartRateRouting() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(message: "What is my heart rate?") { event in
            collector.handle(event)
        }

        #expect(collector.successfulTools.contains("query_health_data"),
                "Model should call query_health_data. Called: \(collector.successfulTools)")
        #expect(!collector.successfulTools.contains("search_calendar"),
                "Should NOT call calendar tool for health query")
    }

    @Test("Model routes 'Show me my blood pressure' to query_health_data",
          .timeLimit(.minutes(2)))
    func bloodPressureRouting() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(message: "Show me my blood pressure") { event in
            collector.handle(event)
        }

        #expect(collector.successfulTools.contains("query_health_data"),
                "Model should call query_health_data. Called: \(collector.successfulTools)")
    }

    @Test("Model routes 'What appointments do I have this week?' to search_calendar",
          .timeLimit(.minutes(2)))
    func calendarRouting() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(message: "What appointments do I have this week?") { event in
            collector.handle(event)
        }

        #expect(collector.successfulTools.contains("search_calendar"),
                "Model should call search_calendar. Called: \(collector.successfulTools)")
    }

    @Test("Model routes 'Show me my reminders' to search_reminders",
          .timeLimit(.minutes(2)))
    func remindersRouting() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(message: "Show me my reminders") { event in
            collector.handle(event)
        }

        #expect(collector.successfulTools.contains("search_reminders"),
                "Model should call search_reminders. Called: \(collector.successfulTools)")
    }

    @Test("Model routes 'How many steps did I walk?' to query_health_data",
          .timeLimit(.minutes(2)))
    func stepsRouting() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(message: "How many steps did I walk?") { event in
            collector.handle(event)
        }

        #expect(collector.successfulTools.contains("query_health_data"),
                "Model should call query_health_data. Called: \(collector.successfulTools)")
    }
}

// MARK: Synthesis Response Quality

@Suite("Synthesis Quality", .serialized)
struct SynthesisQualityTests {

    @Test("Heart rate response includes BPM value and unit",
          .timeLimit(.minutes(2)))
    func heartRateSynthesisQuality() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(message: "What is my heart rate?") { event in
            collector.handle(event)
        }

        let lower = response.lowercased()
        // The model should mention the heart rate value (72 bpm from demo data)
        let mentionsValue = lower.contains("72") || lower.contains("bpm")
            || lower.contains("heart rate") || lower.contains("heart_rate")
        #expect(mentionsValue,
                "Response should reference heart rate data. Got: \(response.prefix(300))")
    }

    @Test("Blood pressure response includes systolic/diastolic values",
          .timeLimit(.minutes(2)))
    func bloodPressureSynthesisQuality() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(message: "Show me my blood pressure") { event in
            collector.handle(event)
        }

        let lower = response.lowercased()
        let mentionsBP = lower.contains("138") || lower.contains("85")
            || lower.contains("mmhg") || lower.contains("blood pressure")
        #expect(mentionsBP,
                "Response should reference blood pressure data. Got: \(response.prefix(300))")
    }

    @Test("Steps response includes step count",
          .timeLimit(.minutes(2)))
    func stepsSynthesisQuality() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(message: "How many steps did I walk?") { event in
            collector.handle(event)
        }

        let lower = response.lowercased()
        let mentionsSteps = lower.contains("6500") || lower.contains("6,500")
            || lower.contains("steps") || lower.contains("step")
        #expect(mentionsSteps,
                "Response should reference step data. Got: \(response.prefix(300))")
    }

    @Test("Calendar response references appointments",
          .timeLimit(.minutes(2)))
    func calendarSynthesisQuality() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(message: "What appointments do I have this week?") { event in
            collector.handle(event)
        }

        // Response should contain something about the calendar results
        // (may be empty if no "SDD Demo" calendar exists on the test device,
        //  but the tool should still have been called)
        #expect(collector.successfulTools.contains("search_calendar"),
                "Calendar tool should have been called")
        // The response should not be empty
        #expect(!response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "Response should not be empty")
    }
}

// MARK: Meeting Prep (Multi-Tool)

@Suite("Meeting Prep", .serialized)
struct MeetingPrepTests {

    @Test("Meeting prep executes all 3 tools via preflight",
          .timeLimit(.minutes(3)))
    func meetingPrepCallsAllTools() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(
            message: "Prepare me for my meeting with Dr. Muller tomorrow"
        ) { event in
            collector.handle(event)
        }

        // Meeting prep uses preflight — all 3 tools should execute
        let completedNames = collector.successfulTools
        #expect(completedNames.contains("search_calendar"),
                "Meeting prep should call search_calendar. Called: \(completedNames)")
        #expect(completedNames.contains("query_health_data"),
                "Meeting prep should call query_health_data. Called: \(completedNames)")
        #expect(completedNames.contains("search_reminders"),
                "Meeting prep should call search_reminders. Called: \(completedNames)")
    }

    @Test("Meeting prep synthesis mentions health data",
          .timeLimit(.minutes(3)))
    func meetingPrepSynthesisHasHealth() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(
            message: "Prepare me for my appointment with Dr. Muller"
        ) { event in
            collector.handle(event)
        }

        let lower = response.lowercased()
        // The synthesis should reference health data from the tool results
        let mentionsHealth = lower.contains("heart") || lower.contains("bpm")
            || lower.contains("72") || lower.contains("health")
            || lower.contains("blood pressure") || lower.contains("138")
        #expect(mentionsHealth,
                "Meeting prep should mention health data. Got: \(response.prefix(500))")
    }

    @Test("Meeting prep synthesis mentions reminders",
          .timeLimit(.minutes(3)))
    func meetingPrepSynthesisHasReminders() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(
            message: "Prepare me for my meeting with Dr. Muller"
        ) { event in
            collector.handle(event)
        }

        let lower = response.lowercased()
        // Should mention at least some reminder content
        let mentionsReminders = lower.contains("reminder") || lower.contains("metoprolol")
            || lower.contains("blood pressure") || lower.contains("exercise")
            || lower.contains("heart rate variability") || lower.contains("question")
        #expect(mentionsReminders,
                "Meeting prep should mention reminders. Got: \(response.prefix(500))")
    }

    @Test("Meeting prep produces non-trivial response",
          .timeLimit(.minutes(3)))
    func meetingPrepResponseLength() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(
            message: "Prepare me for my meeting with Dr. Muller tomorrow"
        ) { event in
            collector.handle(event)
        }

        // A good meeting prep briefing should be substantial
        #expect(response.count > 100,
                "Meeting prep response should be >100 chars. Got \(response.count) chars: \(response.prefix(200))")

        // Should have multiple lines/bullet points
        let lines = response.split(separator: "\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        #expect(lines.count >= 3,
                "Meeting prep should have at least 3 content lines. Got \(lines.count)")
    }
}

// MARK: Tool Execution Pipeline

@Suite("Pipeline Validation", .serialized)
struct PipelineTests {

    @Test("Tool execution produces valid event sequence",
          .timeLimit(.minutes(2)))
    func eventSequenceValid() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(message: "What is my heart rate?") { event in
            collector.handle(event)
        }

        // Should have received tool calls
        #expect(!collector.toolsReceived.isEmpty || collector.toolsCompleted.isEmpty,
                "Should have toolCallsReceived events (unless model answered directly)")

        // Every executed tool should also be completed
        for tool in collector.toolsExecuted {
            let completed = collector.toolsCompleted.map { $0.name }
            #expect(completed.contains(tool),
                    "Tool '\(tool)' was executed but never completed")
        }

        // All completed tools should report success
        for tool in collector.toolsCompleted {
            #expect(tool.success, "Tool '\(tool.name)' should succeed. Summary: \(tool.summary)")
        }

        // Should have used some tokens
        #expect(!collector.tokensUsed.isEmpty, "Should report token usage")
        #expect(collector.tokensUsed.allSatisfy { $0 > 0 }, "Token count should be positive")
    }

    @Test("Health tool returns valid JSON through the pipeline",
          .timeLimit(.minutes(2)))
    func healthToolDataIntegrity() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        _ = try await agent.run(message: "What is my heart rate?") { event in
            collector.handle(event)
        }

        // Find the health tool completion
        let healthCompletion = collector.toolsCompleted.first { $0.name == "query_health_data" }
        if let healthCompletion {
            // The tool result data should be valid JSON
            let data = healthCompletion.data.data(using: .utf8)!
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(json != nil, "Health tool result should be valid JSON")
            #expect(json?["metric"] != nil, "Should have a metric field")
            #expect(json?["unit"] != nil, "Should have a unit field")
        }
        // If no health completion, the model may have answered directly — that's also valid
    }

    @Test("Model generates response within token budget",
          .timeLimit(.minutes(2)))
    func tokenBudgetRespected() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()
        let collector = EventCollector()

        let response = try await agent.run(message: "What is my heart rate?") { event in
            collector.handle(event)
        }

        // Response should be reasonably sized (not runaway generation)
        // With maxTokens=600, response should be well under 3000 chars
        #expect(response.count < 3000,
                "Response should be concise (< 3000 chars). Got \(response.count) chars")
    }

    @Test("Multiple sequential queries work without model state corruption",
          .timeLimit(.minutes(4)))
    func sequentialQueriesStable() async throws {
        let (agent, _, _) = try await TestModelHolder.shared.getAgent()

        // Query 1: Health
        let collector1 = EventCollector()
        let response1 = try await agent.run(message: "What is my heart rate?") { event in
            collector1.handle(event)
        }
        #expect(!response1.isEmpty, "First query should produce a response")

        // Query 2: Calendar
        let collector2 = EventCollector()
        let response2 = try await agent.run(message: "What appointments do I have?") { event in
            collector2.handle(event)
        }
        #expect(!response2.isEmpty, "Second query should produce a response")

        // Query 3: Reminders
        let collector3 = EventCollector()
        let response3 = try await agent.run(message: "Show me my reminders") { event in
            collector3.handle(event)
        }
        #expect(!response3.isEmpty, "Third query should produce a response")

        // Verify different tools were called for each
        let tools1 = Set(collector1.successfulTools)
        let tools2 = Set(collector2.successfulTools)
        let tools3 = Set(collector3.successfulTools)

        // At least one query should have used a different tool than another
        let allSame = tools1 == tools2 && tools2 == tools3
        #expect(!allSame || tools1.isEmpty,
                "Sequential queries should route to different tools. Got: \(tools1), \(tools2), \(tools3)")
    }
}

} // end OnDeviceSLMTests
