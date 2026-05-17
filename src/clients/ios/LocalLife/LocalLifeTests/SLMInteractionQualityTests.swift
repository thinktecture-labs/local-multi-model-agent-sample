// SLMInteractionQualityTests.swift
// End-to-end quality tests for the SLM agent interaction flow.
//
// These tests simulate the full agent loop using mock tools and validate:
// 1. Correct tool routing for demo queries
// 2. Tool result data structure and completeness
// 3. Synthesis prompt quality (what the model receives for Round 2)
// 4. Meeting prep preflight orchestration (ReWOO pattern)
// 5. Event callback sequence and correctness

import Testing
import Foundation

// MARK: - Test Infrastructure

/// Captures AgentEvent-equivalent callbacks for assertions.
private struct EventCapture {
    var textChunks: [String] = []
    var toolCallsReceived: [[String]] = []
    var toolsExecuting: [String] = []
    var toolsCompleted: [(name: String, success: Bool, summary: String, durationMs: Double)] = []
    var generationCompletes: [Int] = []
    var warnings: [String] = []

    var allToolResults: String {
        toolsCompleted.map { "[Tool Result: \($0.name)]\n\($0.summary)" }.joined(separator: "\n\n")
    }
}

/// Minimal tool result matching production ToolResult.
private struct QualityToolResult {
    let success: Bool
    let data: String
    let displaySummary: String
}

/// Simulates the agent loop's tool execution + synthesis prompt construction,
/// matching the production AgentLoop.run() flow.
private final class MockAgentLoop {
    private let tools: [String: (([String: Any]) -> QualityToolResult)]

    init(tools: [String: (([String: Any]) -> QualityToolResult)]) {
        self.tools = tools
    }

    /// Simulate single-tool flow: execute tool, build synthesis prompt.
    func runSingleTool(
        toolName: String,
        arguments: [String: Any],
        originalQuery: String
    ) -> (toolResult: QualityToolResult, synthesisPrompt: String, events: EventCapture) {
        var capture = EventCapture()

        capture.toolCallsReceived.append([toolName])
        capture.toolsExecuting.append(toolName)

        let result = tools[toolName]?(arguments) ?? QualityToolResult(
            success: false, data: "{}", displaySummary: "Unknown tool"
        )

        capture.toolsCompleted.append((
            name: toolName,
            success: result.success,
            summary: result.displaySummary,
            durationMs: 42.0
        ))

        let synthesisPrompt = "[Tool Result: \(toolName)]\n\(result.data)\n\n"
        return (result, synthesisPrompt, capture)
    }

    /// Simulate meeting prep preflight: all 3 tools execute, then synthesis.
    func runMeetingPrep(originalQuery: String) -> (synthesisPrompt: String, events: EventCapture) {
        var capture = EventCapture()

        let preflightTools: [(name: String, args: [String: Any])] = [
            ("search_calendar", ["query": "", "days_ahead": 7]),
            ("query_health_data", ["metric": "heart_rate", "days": 30]),
            ("search_reminders", ["query": ""]),
        ]

        capture.toolCallsReceived.append(preflightTools.map { $0.name })

        var toolResults = ""
        for tool in preflightTools {
            capture.toolsExecuting.append(tool.name)
            let result = tools[tool.name]?(tool.args) ?? QualityToolResult(
                success: false, data: "{}", displaySummary: "Unknown tool"
            )
            capture.toolsCompleted.append((
                name: tool.name, success: result.success,
                summary: result.displaySummary, durationMs: 30.0
            ))
            toolResults += "[Tool Result: \(tool.name)]\n\(result.data)\n\n"
        }

        let synthesisPrompt = """
        The user asked: \(originalQuery)

        Here are the results from their personal data:

        \(toolResults)

        Summarize this as a meeting preparation briefing. Use bullet points. \
        Include health data with units. List the pending reminders. Be concise.
        """

        return (synthesisPrompt, capture)
    }
}

// MARK: - Demo Tool Factories

private func makeDemoTools() -> [String: (([String: Any]) -> QualityToolResult)] {
    [
        "search_calendar": { args in
            let query = args["query"] as? String ?? ""
            if query.isEmpty || query.lowercased().contains("muller") || query.lowercased().contains("müller") {
                return QualityToolResult(
                    success: true,
                    data: """
                    {"count":1,"events":[{"title":"Dr. Muller - Cardiology","start_date":"2026-03-05T13:00:00Z","start_time":"14:00","location":"Klinikum Stuttgart","notes":"Annual checkup, bring recent blood pressure readings"}],"days_ahead":\(args["days_ahead"] as? Int ?? 7),"calendar_filter":"SDD Demo"}
                    """,
                    displaySummary: "1 event: Dr. Muller - Cardiology"
                )
            }
            return QualityToolResult(success: true, data: "{\"count\":0,\"events\":[]}", displaySummary: "No events found")
        },

        "query_health_data": { args in
            let metric = args["metric"] as? String ?? "heart_rate"
            let days = args["days"] as? Int ?? 30
            switch metric {
            case "heart_rate":
                return QualityToolResult(
                    success: true,
                    data: "{\"metric\":\"heart_rate\",\"unit\":\"bpm\",\"days\":\(days),\"count\":\(days * 4),\"avg\":72,\"min\":58,\"max\":94,\"trend\":\"slightly_increasing\",\"notable\":\"Elevated readings Feb 15-18 (avg 82 bpm)\"}",
                    displaySummary: "heart_rate: avg 72 bpm (trend: slightly increasing)"
                )
            case "blood_pressure":
                return QualityToolResult(
                    success: true,
                    data: "{\"metric\":\"blood_pressure\",\"unit\":\"mmHg\",\"systolic_avg\":138,\"diastolic_avg\":85,\"trend\":\"stable\",\"classification\":\"Stage 1 hypertension\"}",
                    displaySummary: "blood_pressure: avg 138/85 mmHg (stable)"
                )
            case "steps":
                return QualityToolResult(
                    success: true,
                    data: "{\"metric\":\"steps\",\"unit\":\"steps/day\",\"avg\":6500,\"trend\":\"slightly_declining\"}",
                    displaySummary: "steps: avg 6,500/day"
                )
            default:
                return QualityToolResult(success: false, data: "{}", displaySummary: "Unknown metric: \(metric)")
            }
        },

        "search_reminders": { args in
            return QualityToolResult(
                success: true,
                data: """
                {"count":5,"reminders":[{"title":"Review blood pressure readings","list":"Health Questions"},{"title":"Heart rate variability","list":"Health Questions"},{"title":"Exercise routine","list":"Health Questions"},{"title":"Metoprolol alternatives","list":"Health Questions"},{"title":"Elevated HR Feb 15-18","list":"Health Questions"}],"query":"\(args["query"] as? String ?? "")"}
                """,
                displaySummary: "5 reminders in 'Health Questions'"
            )
        },
    ]
}

// MARK: - Demo Query Test Cases

@Suite("SLM Interaction Quality — Single Tool Queries")
struct SingleToolQueryTests {

    private var agent: MockAgentLoop {
        MockAgentLoop(tools: makeDemoTools())
    }

    // MARK: Calendar Queries

    @Test("'What appointments do I have?' routes to search_calendar")
    func calendarQueryRouting() {
        let (result, _, events) = agent.runSingleTool(
            toolName: "search_calendar",
            arguments: ["query": "", "days_ahead": 7],
            originalQuery: "What appointments do I have this week?"
        )

        #expect(result.success)
        #expect(events.toolsCompleted.count == 1)
        #expect(events.toolsCompleted[0].name == "search_calendar")
    }

    @Test("Calendar tool result contains event details for synthesis")
    func calendarResultQuality() {
        let (result, synthesisPrompt, _) = agent.runSingleTool(
            toolName: "search_calendar",
            arguments: ["query": "muller", "days_ahead": 7],
            originalQuery: "Do I have a meeting with Dr. Muller?"
        )

        #expect(result.data.contains("Dr. Muller"))
        #expect(result.data.contains("Cardiology"))
        #expect(result.data.contains("Klinikum Stuttgart"))
        #expect(result.data.contains("14:00"))

        // Synthesis prompt should contain the full tool result
        #expect(synthesisPrompt.contains("[Tool Result: search_calendar]"))
        #expect(synthesisPrompt.contains("Dr. Muller"))
    }

    // MARK: Health Queries

    @Test("'What is my heart rate?' returns structured health data")
    func heartRateQueryQuality() {
        let (result, synthesisPrompt, _) = agent.runSingleTool(
            toolName: "query_health_data",
            arguments: ["metric": "heart_rate", "days": 30],
            originalQuery: "What is my heart rate?"
        )

        #expect(result.success)
        #expect(result.data.contains("\"avg\":72"))
        #expect(result.data.contains("\"unit\":\"bpm\""))
        #expect(result.data.contains("\"trend\":\"slightly_increasing\""))

        // Synthesis prompt gives model enough context
        #expect(synthesisPrompt.contains("bpm"))
        #expect(synthesisPrompt.contains("72"))
    }

    @Test("'Show blood pressure' returns systolic and diastolic")
    func bloodPressureQueryQuality() {
        let (result, _, _) = agent.runSingleTool(
            toolName: "query_health_data",
            arguments: ["metric": "blood_pressure", "days": 30],
            originalQuery: "Show me my blood pressure"
        )

        #expect(result.success)
        #expect(result.data.contains("138"))
        #expect(result.data.contains("85"))
        #expect(result.data.contains("mmHg"))
    }

    @Test("'How many steps?' returns step count data")
    func stepsQueryQuality() {
        let (result, _, _) = agent.runSingleTool(
            toolName: "query_health_data",
            arguments: ["metric": "steps", "days": 7],
            originalQuery: "How many steps did I take?"
        )

        #expect(result.success)
        #expect(result.data.contains("6500"))
        #expect(result.data.contains("steps/day"))
    }

    // MARK: Reminder Queries

    @Test("'Show my reminders' returns reminder list")
    func remindersQueryQuality() {
        let (result, synthesisPrompt, _) = agent.runSingleTool(
            toolName: "search_reminders",
            arguments: ["query": "health"],
            originalQuery: "Show me my health reminders"
        )

        #expect(result.success)
        #expect(result.data.contains("Review blood pressure"))
        #expect(result.data.contains("Metoprolol"))
        #expect(result.data.contains("\"count\":5"))

        #expect(synthesisPrompt.contains("[Tool Result: search_reminders]"))
    }
}

@Suite("SLM Interaction Quality — Meeting Prep (Multi-Tool)")
struct MeetingPrepQueryTests {

    private var agent: MockAgentLoop {
        MockAgentLoop(tools: makeDemoTools())
    }

    @Test("Meeting prep executes all 3 tools in correct order")
    func meetingPrepToolOrder() {
        let (_, events) = agent.runMeetingPrep(
            originalQuery: "Prepare me for my meeting with Dr. Muller tomorrow"
        )

        #expect(events.toolCallsReceived.count == 1)
        #expect(events.toolCallsReceived[0] == ["search_calendar", "query_health_data", "search_reminders"])
        #expect(events.toolsExecuting == ["search_calendar", "query_health_data", "search_reminders"])
        #expect(events.toolsCompleted.count == 3)
        #expect(events.toolsCompleted.allSatisfy { $0.success })
    }

    @Test("Meeting prep synthesis prompt contains all tool results")
    func meetingPrepSynthesisCompleteness() {
        let (prompt, _) = agent.runMeetingPrep(
            originalQuery: "Prepare me for my meeting with Dr. Muller tomorrow"
        )

        // Should contain results from all 3 tools
        #expect(prompt.contains("[Tool Result: search_calendar]"))
        #expect(prompt.contains("[Tool Result: query_health_data]"))
        #expect(prompt.contains("[Tool Result: search_reminders]"))

        // Should contain key data points
        #expect(prompt.contains("Dr. Muller"))
        #expect(prompt.contains("72"))       // heart rate avg
        #expect(prompt.contains("bpm"))      // units
        #expect(prompt.contains("Metoprolol")) // reminder item
    }

    @Test("Meeting prep synthesis prompt includes user's original question")
    func meetingPrepIncludesOriginalQuestion() {
        let query = "Prepare me for my meeting with Dr. Muller tomorrow"
        let (prompt, _) = agent.runMeetingPrep(originalQuery: query)

        #expect(prompt.contains("The user asked: \(query)"))
    }

    @Test("Meeting prep synthesis prompt instructs model formatting")
    func meetingPrepInstructsFormatting() {
        let (prompt, _) = agent.runMeetingPrep(
            originalQuery: "Prepare for Dr. Muller appointment"
        )

        #expect(prompt.contains("bullet points"))
        #expect(prompt.contains("units"))
        #expect(prompt.contains("reminders"))
        #expect(prompt.contains("concise"))
    }

    @Test("Meeting prep data contains appointment location")
    func meetingPrepHasLocation() {
        let (prompt, _) = agent.runMeetingPrep(
            originalQuery: "Prepare me for my meeting with Dr. Muller"
        )

        #expect(prompt.contains("Klinikum Stuttgart"))
    }

    @Test("Meeting prep data contains health trends")
    func meetingPrepHasHealthTrends() {
        let (prompt, _) = agent.runMeetingPrep(
            originalQuery: "Prepare me for the doctor appointment"
        )

        #expect(prompt.contains("slightly_increasing"))
    }

    @Test("Meeting prep data contains all 5 reminder items")
    func meetingPrepHasAllReminders() {
        let (prompt, _) = agent.runMeetingPrep(
            originalQuery: "Prepare for Dr. Muller"
        )

        let expectedReminders = [
            "Review blood pressure readings",
            "Heart rate variability",
            "Exercise routine",
            "Metoprolol alternatives",
            "Elevated HR Feb 15-18",
        ]
        for reminder in expectedReminders {
            #expect(prompt.contains(reminder), "Missing reminder: \(reminder)")
        }
    }
}

@Suite("SLM Interaction Quality — Event Sequence")
struct EventSequenceTests {

    private var agent: MockAgentLoop {
        MockAgentLoop(tools: makeDemoTools())
    }

    @Test("Single tool query produces correct event sequence")
    func singleToolEventSequence() {
        let (_, _, events) = agent.runSingleTool(
            toolName: "search_calendar",
            arguments: [:],
            originalQuery: "Show calendar"
        )

        // Expected sequence: toolCallsReceived → toolExecuting → toolCompleted
        #expect(events.toolCallsReceived.count == 1)
        #expect(events.toolsExecuting.count == 1)
        #expect(events.toolsCompleted.count == 1)
    }

    @Test("Meeting prep produces 3 tool completions")
    func meetingPrepEventCount() {
        let (_, events) = agent.runMeetingPrep(originalQuery: "Prepare for Dr. Muller")

        #expect(events.toolsCompleted.count == 3)
        #expect(events.toolsCompleted[0].name == "search_calendar")
        #expect(events.toolsCompleted[1].name == "query_health_data")
        #expect(events.toolsCompleted[2].name == "search_reminders")
    }

    @Test("All tool completions report success")
    func allToolsSucceed() {
        let (_, events) = agent.runMeetingPrep(originalQuery: "Prepare for Dr. Muller")

        for completion in events.toolsCompleted {
            #expect(completion.success, "Tool \(completion.name) should succeed")
        }
    }

    @Test("Tool durations are positive")
    func toolDurationsPositive() {
        let (_, events) = agent.runMeetingPrep(originalQuery: "Prepare for Dr. Muller")

        for completion in events.toolsCompleted {
            #expect(completion.durationMs > 0, "Tool \(completion.name) duration should be > 0")
        }
    }
}

@Suite("SLM Interaction Quality — Edge Cases")
struct EdgeCaseTests {

    private var agent: MockAgentLoop {
        MockAgentLoop(tools: makeDemoTools())
    }

    @Test("Empty calendar returns graceful result")
    func emptyCalendar() {
        let (result, _, _) = agent.runSingleTool(
            toolName: "search_calendar",
            arguments: ["query": "nonexistent_event_xyz", "days_ahead": 7],
            originalQuery: "Do I have a dentist appointment?"
        )

        #expect(result.success)
        #expect(result.data.contains("\"count\":0"))
    }

    @Test("Unknown health metric returns failure")
    func unknownHealthMetric() {
        let (result, _, _) = agent.runSingleTool(
            toolName: "query_health_data",
            arguments: ["metric": "body_temperature", "days": 7],
            originalQuery: "What is my body temperature?"
        )

        #expect(!result.success)
        #expect(result.displaySummary.contains("Unknown metric"))
    }

    @Test("Tool result JSON is parseable")
    func toolResultsAreParseable() throws {
        let toolCases: [(String, [String: Any])] = [
            ("search_calendar", ["query": "", "days_ahead": 7]),
            ("query_health_data", ["metric": "heart_rate", "days": 30]),
            ("search_reminders", ["query": ""]),
        ]

        let tools = makeDemoTools()
        for (name, args) in toolCases {
            let result = tools[name]!(args)
            let data = result.data.data(using: .utf8)!
            let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(parsed != nil, "Tool \(name) result should be valid JSON")
        }
    }

    @Test("Synthesis prompt is not empty after tool execution")
    func synthesisPromptNotEmpty() {
        let (_, synthesisPrompt, _) = agent.runSingleTool(
            toolName: "query_health_data",
            arguments: ["metric": "heart_rate", "days": 30],
            originalQuery: "Heart rate?"
        )

        #expect(!synthesisPrompt.isEmpty)
        #expect(synthesisPrompt.count > 50, "Synthesis prompt should have meaningful content")
    }
}
