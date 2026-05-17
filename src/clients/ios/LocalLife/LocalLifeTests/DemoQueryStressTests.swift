// DemoQueryStressTests.swift
// Runs each of the 4 app demo queries 5 times consecutively, logging
// tool choices and final answers to assess SLM consistency/flakiness.

import Testing
import Foundation
import LocalLife
import EventKit

// MARK: - Shared Infrastructure (reuse model from OnDeviceSLMTests)

private actor StressTestModelHolder {
    static let shared = StressTestModelHolder()

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
            throw StressSetupError.modelNotAvailable
        }
        didSetup = true
        do {
            let mm = ModelManager()
            let tr = ToolRegistry()
            let sharedEventStore = EKEventStore()
            let healthKit = HealthKitTool()
            healthKit.demoMode = true
            let calendar = CalendarTool(eventStore: sharedEventStore)
            calendar.calendarFilter = "SDD Demo"
            let reminders = RemindersTool(eventStore: sharedEventStore)
            tr.register(healthKit)
            tr.register(calendar)
            tr.register(reminders)
            await mm.checkModelAvailability()
            guard mm.state == .downloaded else {
                throw StressSetupError.modelNotDownloaded
            }
            try await mm.loadModel(systemPrompt: AgentConfiguration.systemPrompt)
            guard mm.isReady else { throw StressSetupError.modelNotReady }
            for fn in tr.asLeapFunctions() { mm.registerFunction(fn) }
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

private enum StressSetupError: Error, CustomStringConvertible {
    case modelNotDownloaded, modelNotReady, modelNotAvailable
    var description: String {
        switch self {
        case .modelNotDownloaded: return "Model not downloaded"
        case .modelNotReady: return "Model not ready"
        case .modelNotAvailable: return "Model setup previously failed"
        }
    }
}

/// Captures tool calls and text for a single query run.
private final class RunCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var _toolsCompleted: [(name: String, success: Bool, args: [String: String])] = []
    private var _textChunks: [String] = []

    var toolsCalled: [(name: String, success: Bool, args: [String: String])] {
        lock.withLock { _toolsCompleted }
    }
    var toolNames: [String] { lock.withLock { _toolsCompleted.map { $0.name } } }
    var fullText: String { lock.withLock { _textChunks.joined() } }

    func handle(_ event: AgentEvent) {
        lock.withLock {
            switch event {
            case .textChunk(let text):
                _textChunks.append(text)
            case .toolCompleted(let name, let success, _, _, let args, _):
                _toolsCompleted.append((name: name, success: success, args: args))
            default:
                break
            }
        }
    }
}

// MARK: - Stress Tests

@Suite("Demo Query Stress Tests", .serialized)
struct DemoQueryStressTests {

    static let iterations = 5

    static let demoQueries: [(label: String, query: String, expectedTool: String)] = [
        ("Calendar", "What appointments do I have this week?", "search_calendar"),
        ("HealthKit", "Show my heart rate trends for the last 30 days", "query_health_data"),
        ("Reminders", "Search my reminders for health questions", "search_reminders"),
        ("MeetingPrep", "Prepare me for my meeting with Dr. Pepper tomorrow", "search_calendar"),
    ]

    @Test("Calendar query x5 — tool choices and answers",
          .timeLimit(.minutes(5)))
    func calendarStress() async throws {
        try await runStress(queryIndex: 0)
    }

    @Test("HealthKit query x5 — tool choices and answers",
          .timeLimit(.minutes(5)))
    func healthKitStress() async throws {
        try await runStress(queryIndex: 1)
    }

    @Test("Reminders query x5 — tool choices and answers",
          .timeLimit(.minutes(5)))
    func remindersStress() async throws {
        try await runStress(queryIndex: 2)
    }

    @Test("MeetingPrep query x5 — tool choices and answers",
          .timeLimit(.minutes(10)))
    func meetingPrepStress() async throws {
        try await runStress(queryIndex: 3)
    }

    // MARK: - Core runner

    private func runStress(queryIndex: Int) async throws {
        let (agent, _, _) = try await StressTestModelHolder.shared.getAgent()
        let q = Self.demoQueries[queryIndex]

        var results: [(run: Int, tools: [String], args: [[String: String]], response: String, correct: Bool)] = []

        for i in 1...Self.iterations {
            let collector = RunCollector()
            let response = try await agent.run(message: q.query) { event in
                collector.handle(event)
            }

            let tools = collector.toolNames
            let args = collector.toolsCalled.map { $0.args }
            let correct = tools.contains(q.expectedTool)
            results.append((run: i, tools: tools, args: args, response: response, correct: correct))
        }

        let correctCount = results.filter { $0.correct }.count
        let consistency = Set(results.map { $0.tools.description }).count == 1

        // Build detailed report
        var lines: [String] = []
        lines.append("")
        lines.append("═══════════════════════════════════════════════════")
        lines.append("  [\(q.label)] \(correctCount)/\(Self.iterations) correct | \(consistency ? "CONSISTENT" : "INCONSISTENT")")
        lines.append("  Query: \"\(q.query)\"")
        lines.append("  Expected tool: \(q.expectedTool)")
        lines.append("═══════════════════════════════════════════════════")
        for r in results {
            let status = r.correct ? "✅" : "❌"
            let toolStr = r.tools.isEmpty ? "(none)" : r.tools.joined(separator: ", ")
            let argsStr = r.args.map { a in
                a.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
            }.joined(separator: " | ")
            lines.append("")
            lines.append("  Run \(r.run) \(status)  tool=[\(toolStr)]  args=[\(argsStr)]")
            lines.append("  ───────────────────────────────────────────────")
            // Full response, indented
            let responseLines = r.response.split(separator: "\n", omittingEmptySubsequences: false)
            for rl in responseLines {
                lines.append("    \(rl)")
            }
            lines.append("  ───────────────────────────────────────────────")
        }
        lines.append("")
        let report = lines.joined(separator: "\n")

        // Always print so it's visible in Xcode console
        print(report)

        // Pass/fail assertion
        #expect(correctCount >= 3,
                "STRESS_REPORT:\n\(report)")
    }
}
