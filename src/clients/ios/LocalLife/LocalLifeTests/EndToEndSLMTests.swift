// EndToEndSLMTests.swift
// End-to-end tests for the full SLM interaction pipeline.
//
// These tests exercise the complete agent orchestration path:
//   User query → isMeetingPrepQuery → tool selection (bestMatch) →
//   tool execution → synthesis prompt construction → response validation
//
// The tests use production-equivalent types and real demo data to validate
// what the on-device LFM 2.5 1.2B model would receive and produce.
//
// Future: Once LeapSDK is linked to the test target and the test bundle
// is configured as a hosted test (with LocalLife.app as host), these can
// be extended to run actual model inference on-device. See MARK: Model-in-the-Loop.

import Testing
import Foundation
import LocalLife

// MARK: - Production-Equivalent Types

/// Mirrors production ToolResult exactly.
private struct E2EToolResult {
    let success: Bool
    let data: String
    let displaySummary: String
    let error: String?
}

/// Mirrors production ToolParameterSchema.
private struct E2EToolParameterSchema {
    enum ParamType: String { case string, integer }
    let name: String
    let type: ParamType
    let description: String
    let isOptional: Bool
}

/// Mirrors production Tool protocol.
private protocol E2ETool {
    var name: String { get }
    var description: String { get }
    var parameters: [E2EToolParameterSchema] { get }
    func execute(arguments: [String: Any]) async -> E2EToolResult
}

/// Mirrors production ToolRegistry with NSLock thread safety.
private final class E2EToolRegistry: @unchecked Sendable {
    private var tools: [String: E2ETool] = [:]
    private let lock = NSLock()

    func register(_ tool: E2ETool) {
        lock.withLock { tools[tool.name] = tool }
    }

    func get(_ name: String) -> E2ETool? {
        lock.withLock { tools[name] }
    }

    var allTools: [E2ETool] {
        lock.withLock { Array(tools.values) }
    }

    func execute(name: String, arguments: [String: Any]) async -> E2EToolResult {
        let tool = lock.withLock { tools[name] }
        guard let tool else {
            let available = lock.withLock { tools.keys.joined(separator: ", ") }
            return E2EToolResult(
                success: false, data: "{}",
                displaySummary: "Unknown tool: \(name)",
                error: "Tool '\(name)' not registered. Available: \(available)"
            )
        }
        return await tool.execute(arguments: arguments)
    }
}

/// Mirrors production AgentEvent.
private enum E2EAgentEvent {
    case textChunk(String)
    case toolCallsReceived([String])
    case toolExecuting(name: String)
    case toolCompleted(name: String, success: Bool, summary: String, durationMs: Double, arguments: [String: String], resultData: String)
    case generationComplete(tokensUsed: Int)
    case warning(String)
}

// MARK: - Production-Equivalent Agent Loop

/// Mirrors production AgentLoop orchestration without LeapSDK model inference.
/// Exercises the exact same tool selection, execution, and synthesis prompt
/// construction logic as the real agent.
private final class E2EAgentLoop {
    private let toolRegistry: E2EToolRegistry

    init(toolRegistry: E2EToolRegistry) {
        self.toolRegistry = toolRegistry
    }

    /// Full pipeline for a single-tool query.
    func runSingleToolQuery(
        query: String,
        selectedTool: String,
        arguments: [String: Any],
        onEvent: ((E2EAgentEvent) -> Void)? = nil
    ) async -> (toolResult: E2EToolResult, synthesisPrompt: String) {
        onEvent?(.toolCallsReceived([selectedTool]))
        onEvent?(.toolExecuting(name: selectedTool))

        let start = CFAbsoluteTimeGetCurrent()
        let result = await toolRegistry.execute(name: selectedTool, arguments: arguments)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let argStrings = arguments.mapValues { "\($0)" }
        onEvent?(.toolCompleted(
            name: selectedTool, success: result.success,
            summary: result.displaySummary, durationMs: elapsed,
            arguments: argStrings, resultData: result.data
        ))

        let synthesisPrompt = "[Tool Result: \(selectedTool)]\n\(result.data)\n\n"
        return (result, synthesisPrompt)
    }

    /// Full pipeline for meeting prep (ReWOO pattern): preflight all 3 tools.
    func runMeetingPrep(
        query: String,
        onEvent: ((E2EAgentEvent) -> Void)? = nil
    ) async -> (synthesisPrompt: String, toolResults: [(name: String, result: E2EToolResult)]) {
        let preflightTools: [(name: String, args: [String: Any])] = [
            ("search_calendar", ["query": "", "days_ahead": 7]),
            ("query_health_data", ["metric": "heart_rate", "days": 30]),
            ("search_reminders", ["query": ""]),
        ]

        onEvent?(.toolCallsReceived(preflightTools.map { $0.name }))

        var results: [(name: String, result: E2EToolResult)] = []
        var toolResultsMessage = ""

        for tool in preflightTools {
            onEvent?(.toolExecuting(name: tool.name))
            let start = CFAbsoluteTimeGetCurrent()
            let result = await toolRegistry.execute(name: tool.name, arguments: tool.args)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let argStrings = tool.args.mapValues { "\($0)" }
            onEvent?(.toolCompleted(
                name: tool.name, success: result.success,
                summary: result.displaySummary, durationMs: elapsed,
                arguments: argStrings, resultData: result.data
            ))
            results.append((name: tool.name, result: result))
            toolResultsMessage += "[Tool Result: \(tool.name)]\n\(result.data)\n\n"
        }

        // Build the exact same synthesis prompt as production AgentLoop
        let synthesisPrompt = """
        The user asked: \(query)

        Here are the results from their personal data:

        \(toolResultsMessage)

        Summarize this as a meeting preparation briefing. Use bullet points. \
        Include health data with units. List the pending reminders. Be concise.
        """

        return (synthesisPrompt, results)
    }

    /// Production-equivalent bestMatch: pick the tool whose metadata best matches the query.
    func bestMatch(toolNames: [String], forQuery query: String) -> String {
        let queryWords = Set(query.lowercased().split(whereSeparator: { !$0.isLetter }))
        var best = toolNames[0]
        var bestScore = -1

        for name in toolNames {
            if let tool = toolRegistry.get(name) {
                let metadata = "\(tool.name) \(tool.description) \(tool.parameters.map(\.description).joined(separator: " "))"
                let metaWords = Set(metadata.lowercased().split(whereSeparator: { !$0.isLetter }))
                let score = queryWords.intersection(metaWords).count
                if score > bestScore {
                    bestScore = score
                    best = name
                }
            }
        }
        return best
    }

    /// Delegates to the production AgentLoop.isMeetingPrepQuery.
    static func isMeetingPrepQuery(_ message: String) -> Bool {
        AgentLoop.isMeetingPrepQuery(message)
    }
}

// MARK: - Production-Equivalent Demo Tools

/// HealthKit demo tool with exact same data as production HealthKitTool.demoData().
private final class E2EHealthKitTool: E2ETool {
    let name = "query_health_data"
    let description = "Query the user's health data from HealthKit. Can retrieve heart rate, blood pressure, steps, active energy, sleep, and weight for a specified number of past days."

    var parameters: [E2EToolParameterSchema] {
        [
            E2EToolParameterSchema(name: "metric", type: .string, description: "The health metric to query (heart_rate, blood_pressure, steps, active_energy, sleep, weight)", isOptional: false),
            E2EToolParameterSchema(name: "days", type: .integer, description: "Number of past days to query (default 30)", isOptional: true),
        ]
    }

    func execute(arguments: [String: Any]) async -> E2EToolResult {
        let metric = arguments["metric"] as? String ?? "heart_rate"
        let days = arguments["days"] as? Int ?? 30
        return demoData(metric: metric, days: days)
    }

    private func demoData(metric: String, days: Int) -> E2EToolResult {
        let data: [String: Any]
        let summary: String

        switch metric {
        case "heart_rate":
            data = [
                "metric": "heart_rate", "unit": "bpm", "days": days,
                "count": days * 4, "avg": 72, "min": 58, "max": 94,
                "trend": "slightly_increasing",
                "trend_detail": "Average increased from 65 bpm to 72 bpm over past 3 months",
                "notable": "Elevated readings Feb 15-18 (avg 82 bpm)",
                "recent_7day_avg": 74,
            ]
            summary = "heart_rate: avg 72 bpm (trend: slightly increasing), \(days * 4) readings over \(days) days"

        case "blood_pressure":
            data = [
                "metric": "blood_pressure", "unit": "mmHg", "days": days,
                "count": days, "systolic_avg": 138, "diastolic_avg": 85,
                "systolic_range": "130-148", "diastolic_range": "78-92",
                "trend": "stable", "classification": "Stage 1 hypertension",
            ]
            summary = "blood_pressure: avg 138/85 mmHg (stable, Stage 1 hypertension)"

        case "steps":
            data = [
                "metric": "steps", "unit": "steps/day", "days": days,
                "count": days, "avg": 6500, "min": 2100, "max": 12400,
                "trend": "slightly_declining",
                "trend_detail": "Down from 7200/day to 6500/day over past month",
                "total": 6500 * days, "days_above_8000": 8,
            ]
            summary = "steps: avg 6,500/day (declining, target 8,000)"

        case "active_energy":
            data = [
                "metric": "active_energy", "unit": "kcal/day", "days": days,
                "count": days, "avg": 380, "min": 120, "max": 650, "trend": "stable",
            ]
            summary = "active_energy: avg 380 kcal/day"

        case "weight":
            data = [
                "metric": "weight", "unit": "kg", "days": days,
                "count": 8, "avg": 82.0, "min": 81.2, "max": 82.8,
                "trend": "stable", "latest": 82.1,
            ]
            summary = "weight: 82.1 kg (stable)"

        case "sleep":
            data = [
                "metric": "sleep", "unit": "hours/night", "days": days,
                "count": days, "avg": 6.8, "min": 4.5, "max": 8.2, "trend": "stable",
            ]
            summary = "sleep: avg 6.8 hours/night"

        default:
            return E2EToolResult(success: false, data: "{}", displaySummary: "Unknown metric: \(metric)", error: "Unsupported metric")
        }

        let jsonData = (try? JSONSerialization.data(withJSONObject: data)) ?? Data()
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
        return E2EToolResult(success: true, data: jsonString, displaySummary: summary, error: nil)
    }
}

/// Calendar demo tool returning curated events.
private final class E2ECalendarTool: E2ETool {
    let name = "search_calendar"
    let description = "Search the user's calendar for upcoming appointments and events. Returns title, date, time, location, and notes for matching appointments."

    var parameters: [E2EToolParameterSchema] {
        [
            E2EToolParameterSchema(name: "query", type: .string, description: "Optional search term to filter events by title, location, or notes", isOptional: true),
            E2EToolParameterSchema(name: "days_ahead", type: .integer, description: "Number of days ahead to search (default 7)", isOptional: true),
        ]
    }

    func execute(arguments: [String: Any]) async -> E2EToolResult {
        let query = (arguments["query"] as? String)?.lowercased() ?? ""
        let daysAhead = arguments["days_ahead"] as? Int ?? 7

        // Demo: always return Dr. Muller appointment for relevant queries
        if query.isEmpty || query.contains("muller") || query.contains("müller") || query.contains("dr") {
            let tomorrow = Calendar.current.date(byAdding: .day, value: 1, to: Date()) ?? Date()
            let dateStr = ISO8601DateFormatter().string(from: tomorrow)
            let data: [String: Any] = [
                "count": 1,
                "events": [[
                    "title": "Dr. Muller - Cardiology",
                    "start_date": dateStr,
                    "start_time": "14:00",
                    "location": "Klinikum Stuttgart",
                    "notes": "Annual checkup, bring recent blood pressure readings",
                    "is_all_day": false,
                ]],
                "search_query": query,
                "days_ahead": daysAhead,
                "calendar_filter": "SDD Demo",
            ]
            let jsonData = (try? JSONSerialization.data(withJSONObject: data)) ?? Data()
            let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
            return E2EToolResult(success: true, data: jsonString, displaySummary: "1 event: Dr. Muller - Cardiology", error: nil)
        }

        let data: [String: Any] = ["count": 0, "events": [] as [[String: Any]], "search_query": query, "days_ahead": daysAhead, "calendar_filter": "SDD Demo"]
        let jsonData = (try? JSONSerialization.data(withJSONObject: data)) ?? Data()
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
        return E2EToolResult(success: true, data: jsonString, displaySummary: "No events found", error: nil)
    }
}

/// Reminders demo tool returning curated health questions.
private final class E2ERemindersTool: E2ETool {
    let name = "search_reminders"
    let description = "Search the user's Reminders for tasks and notes. Returns matching reminders with title, notes, completion status, and due date."

    var parameters: [E2EToolParameterSchema] {
        [
            E2EToolParameterSchema(name: "query", type: .string, description: "Search term to find relevant reminders by title or notes", isOptional: false),
            E2EToolParameterSchema(name: "list_name", type: .string, description: "Optional: restrict search to a specific Reminders list", isOptional: true),
        ]
    }

    func execute(arguments: [String: Any]) async -> E2EToolResult {
        let query = (arguments["query"] as? String) ?? ""
        let reminders: [[String: Any]] = [
            ["title": "Review blood pressure readings", "list": "Health Questions", "is_completed": false],
            ["title": "Heart rate variability", "list": "Health Questions", "is_completed": false],
            ["title": "Exercise routine", "list": "Health Questions", "is_completed": false],
            ["title": "Metoprolol alternatives", "list": "Health Questions", "is_completed": false],
            ["title": "Elevated HR Feb 15-18", "list": "Health Questions", "is_completed": false],
        ]
        let data: [String: Any] = [
            "count": reminders.count,
            "reminders": reminders,
            "query": query,
            "list_filter": "all",
        ]
        let jsonData = (try? JSONSerialization.data(withJSONObject: data)) ?? Data()
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
        return E2EToolResult(
            success: true, data: jsonString,
            displaySummary: "5 reminders in 'Health Questions'", error: nil
        )
    }
}

// MARK: - Helper: Build a fully wired E2E agent

private func makeE2EAgent() -> (agent: E2EAgentLoop, registry: E2EToolRegistry) {
    let registry = E2EToolRegistry()
    registry.register(E2ECalendarTool())
    registry.register(E2EHealthKitTool())
    registry.register(E2ERemindersTool())
    let agent = E2EAgentLoop(toolRegistry: registry)
    return (agent, registry)
}

// MARK: - E2E: Full Pipeline — Demo Query 1: "What is my heart rate?"

@Suite("E2E Pipeline — Heart Rate Query")
struct E2EHeartRateTests {

    @Test("Full pipeline: heart rate query → tool execution → synthesis prompt")
    func heartRateFullPipeline() async throws {
        let (agent, _) = makeE2EAgent()

        // Step 1: Query classification
        let query = "What is my heart rate?"
        #expect(!E2EAgentLoop.isMeetingPrepQuery(query), "Should NOT be classified as meeting prep")

        // Step 2: Tool selection via bestMatch
        let allTools = ["search_calendar", "query_health_data", "search_reminders"]
        let selected = agent.bestMatch(toolNames: allTools, forQuery: query)
        #expect(selected == "query_health_data", "bestMatch should route to health tool")

        // Step 3: Tool execution
        var events: [String] = []
        let (result, synthesisPrompt) = await agent.runSingleToolQuery(
            query: query,
            selectedTool: selected,
            arguments: ["metric": "heart_rate", "days": 30]
        ) { event in
            switch event {
            case .toolCallsReceived(let names): events.append("received:\(names.joined(separator: ","))")
            case .toolExecuting(let name): events.append("executing:\(name)")
            case .toolCompleted(let name, _, _, _, _, _): events.append("completed:\(name)")
            default: break
            }
        }

        // Step 4: Validate tool result
        #expect(result.success)
        let jsonData = try #require(result.data.data(using: .utf8))
        let json = try #require(JSONSerialization.jsonObject(with: jsonData) as? [String: Any])
        #expect(json["metric"] as? String == "heart_rate")
        #expect(json["avg"] as? Int == 72)
        #expect(json["unit"] as? String == "bpm")
        #expect(json["trend"] as? String == "slightly_increasing")

        // Step 5: Validate synthesis prompt
        #expect(synthesisPrompt.contains("[Tool Result: query_health_data]"))
        #expect(synthesisPrompt.contains("72"))
        #expect(synthesisPrompt.contains("bpm"))

        // Step 6: Validate event sequence
        #expect(events == ["received:query_health_data", "executing:query_health_data", "completed:query_health_data"])
    }

    @Test("Heart rate data has enough detail for SLM synthesis")
    func heartRateDataRichness() async throws {
        let (agent, _) = makeE2EAgent()
        let (result, _) = await agent.runSingleToolQuery(
            query: "What is my heart rate?",
            selectedTool: "query_health_data",
            arguments: ["metric": "heart_rate", "days": 30]
        )

        let jsonData = try #require(result.data.data(using: .utf8))
        let json = try #require(JSONSerialization.jsonObject(with: jsonData) as? [String: Any])

        // The SLM needs these data points to generate a good response:
        #expect(json["avg"] != nil, "Average HR needed for summary")
        #expect(json["min"] != nil, "Min HR shows range")
        #expect(json["max"] != nil, "Max HR shows range")
        #expect(json["trend"] != nil, "Trend needed for health context")
        #expect(json["notable"] != nil, "Notable events provide talking points")
        #expect(json["days"] != nil, "Time period gives context")
    }
}

// MARK: - E2E: Full Pipeline — Demo Query 2: "Show me my blood pressure"

@Suite("E2E Pipeline — Blood Pressure Query")
struct E2EBloodPressureTests {

    @Test("Full pipeline: blood pressure query → tool → synthesis")
    func bloodPressureFullPipeline() async throws {
        let (agent, _) = makeE2EAgent()

        let query = "Show me my blood pressure"
        #expect(!E2EAgentLoop.isMeetingPrepQuery(query))

        let selected = agent.bestMatch(
            toolNames: ["search_calendar", "query_health_data", "search_reminders"],
            forQuery: query
        )
        #expect(selected == "query_health_data")

        let (result, synthesisPrompt) = await agent.runSingleToolQuery(
            query: query,
            selectedTool: "query_health_data",
            arguments: ["metric": "blood_pressure", "days": 30]
        )

        #expect(result.success)
        let bpData = try #require(result.data.data(using: .utf8))
        let json = try #require(JSONSerialization.jsonObject(with: bpData) as? [String: Any])
        #expect(json["systolic_avg"] as? Int == 138)
        #expect(json["diastolic_avg"] as? Int == 85)
        #expect(json["unit"] as? String == "mmHg")
        #expect(json["classification"] as? String == "Stage 1 hypertension")

        #expect(synthesisPrompt.contains("138"))
        #expect(synthesisPrompt.contains("85"))
        #expect(synthesisPrompt.contains("mmHg"))
    }
}

// MARK: - E2E: Full Pipeline — Demo Query 3: "Prepare me for my meeting with Dr. Muller"

@Suite("E2E Pipeline — Meeting Prep Query")
struct E2EMeetingPrepTests {

    @Test("Full pipeline: meeting prep query → 3 tools → synthesis prompt")
    func meetingPrepFullPipeline() async {
        let (agent, _) = makeE2EAgent()
        let query = "Prepare me for my meeting with Dr. Muller tomorrow"

        // Step 1: Query classification
        #expect(E2EAgentLoop.isMeetingPrepQuery(query), "Should be classified as meeting prep")

        // Step 2: Execute preflight (all 3 tools)
        var eventLog: [String] = []
        let (synthesisPrompt, toolResults) = await agent.runMeetingPrep(query: query) { event in
            switch event {
            case .toolCallsReceived(let names): eventLog.append("received:\(names.joined(separator: ","))")
            case .toolExecuting(let name): eventLog.append("exec:\(name)")
            case .toolCompleted(let name, let ok, _, _, _, _): eventLog.append("done:\(name):\(ok)")
            default: break
            }
        }

        // Step 3: Validate all tools succeeded
        #expect(toolResults.count == 3)
        #expect(toolResults.allSatisfy { $0.result.success })

        // Step 4: Validate individual tool results
        let calendarResult = toolResults[0]
        #expect(calendarResult.name == "search_calendar")
        #expect(calendarResult.result.data.contains("Dr. Muller"))
        #expect(calendarResult.result.data.contains("Klinikum Stuttgart"))

        let healthResult = toolResults[1]
        #expect(healthResult.name == "query_health_data")
        #expect(healthResult.result.data.contains("72"))
        #expect(healthResult.result.data.contains("bpm"))

        let remindersResult = toolResults[2]
        #expect(remindersResult.name == "search_reminders")
        #expect(remindersResult.result.data.contains("Metoprolol"))

        // Step 5: Validate synthesis prompt structure
        #expect(synthesisPrompt.contains("The user asked: \(query)"))
        #expect(synthesisPrompt.contains("[Tool Result: search_calendar]"))
        #expect(synthesisPrompt.contains("[Tool Result: query_health_data]"))
        #expect(synthesisPrompt.contains("[Tool Result: search_reminders]"))
        #expect(synthesisPrompt.contains("bullet points"))
        #expect(synthesisPrompt.contains("concise"))

        // Step 6: Validate event sequence matches production
        #expect(eventLog == [
            "received:search_calendar,query_health_data,search_reminders",
            "exec:search_calendar", "done:search_calendar:true",
            "exec:query_health_data", "done:query_health_data:true",
            "exec:search_reminders", "done:search_reminders:true",
        ])
    }

    @Test("Meeting prep synthesis has enough data for comprehensive briefing")
    func meetingPrepSynthesisCompleteness() async {
        let (agent, _) = makeE2EAgent()
        let (prompt, _) = await agent.runMeetingPrep(
            query: "Prepare me for my appointment with Dr. Muller"
        )

        // Calendar data for briefing
        #expect(prompt.contains("Dr. Muller - Cardiology"))
        #expect(prompt.contains("14:00"))
        #expect(prompt.contains("Klinikum Stuttgart"))
        #expect(prompt.contains("blood pressure readings"))  // from event notes

        // Health data for discussion points
        #expect(prompt.contains("heart_rate"))
        #expect(prompt.contains("72"))
        #expect(prompt.contains("slightly_increasing"))

        // Reminders for question list
        #expect(prompt.contains("Review blood pressure readings"))
        #expect(prompt.contains("Heart rate variability"))
        #expect(prompt.contains("Exercise routine"))
        #expect(prompt.contains("Metoprolol alternatives"))
        #expect(prompt.contains("Elevated HR Feb 15-18"))
    }

    @Test("Meeting prep with German query variant")
    func meetingPrepGermanVariant() async {
        let (agent, _) = makeE2EAgent()

        // German query with English meeting keyword triggers meeting prep
        let query = "Prepare me for the Arzt appointment"
        #expect(E2EAgentLoop.isMeetingPrepQuery(query))

        let (prompt, results) = await agent.runMeetingPrep(query: query)
        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.result.success })
        #expect(prompt.contains("The user asked: \(query)"))
    }
}

// MARK: - E2E: Full Pipeline — Demo Query 4: "Show my reminders"

@Suite("E2E Pipeline — Reminders Query")
struct E2ERemindersTests {

    @Test("Full pipeline: reminders query → tool → synthesis")
    func remindersFullPipeline() async throws {
        let (agent, _) = makeE2EAgent()

        // Use a query where "reminders" / "tasks" clearly route to search_reminders
        let query = "Show me my reminders and tasks"
        #expect(!E2EAgentLoop.isMeetingPrepQuery(query))

        let selected = agent.bestMatch(
            toolNames: ["search_calendar", "query_health_data", "search_reminders"],
            forQuery: query
        )
        #expect(selected == "search_reminders")

        let (result, synthesisPrompt) = await agent.runSingleToolQuery(
            query: query,
            selectedTool: "search_reminders",
            arguments: ["query": "health"]
        )

        #expect(result.success)
        let jsonData = try #require(result.data.data(using: .utf8))
        let json = try #require(JSONSerialization.jsonObject(with: jsonData) as? [String: Any])
        #expect(json["count"] as? Int == 5)
        let reminders = json["reminders"] as? [[String: Any]]
        #expect(reminders?.count == 5)

        #expect(synthesisPrompt.contains("[Tool Result: search_reminders]"))
        #expect(synthesisPrompt.contains("Metoprolol"))
    }
}

// MARK: - E2E: Full Pipeline — Demo Query 5: "What appointments do I have?"

@Suite("E2E Pipeline — Calendar Query")
struct E2ECalendarTests {

    @Test("Full pipeline: calendar query → tool → synthesis")
    func calendarFullPipeline() async {
        let (agent, _) = makeE2EAgent()

        let query = "What appointments do I have this week?"
        #expect(!E2EAgentLoop.isMeetingPrepQuery(query))

        let selected = agent.bestMatch(
            toolNames: ["search_calendar", "query_health_data", "search_reminders"],
            forQuery: query
        )
        #expect(selected == "search_calendar")

        let (result, synthesisPrompt) = await agent.runSingleToolQuery(
            query: query,
            selectedTool: "search_calendar",
            arguments: ["query": "", "days_ahead": 7]
        )

        #expect(result.success)
        #expect(result.data.contains("Dr. Muller"))
        #expect(synthesisPrompt.contains("[Tool Result: search_calendar]"))
    }

    @Test("Calendar query with no matching events")
    func calendarNoResults() async throws {
        let (agent, _) = makeE2EAgent()

        let (result, _) = await agent.runSingleToolQuery(
            query: "Do I have a dentist appointment?",
            selectedTool: "search_calendar",
            arguments: ["query": "dentist", "days_ahead": 7]
        )

        #expect(result.success)
        let jsonData = try #require(result.data.data(using: .utf8))
        let json = try #require(JSONSerialization.jsonObject(with: jsonData) as? [String: Any])
        #expect(json["count"] as? Int == 0)
        #expect(result.displaySummary == "No events found")
    }
}

// MARK: - E2E: Tool Result JSON Contract Validation

@Suite("E2E — Tool Result JSON Contracts")
struct E2EToolResultContractTests {

    @Test("All demo tool results produce valid, parseable JSON")
    func allToolResultsValidJSON() async throws {
        let (agent, _) = makeE2EAgent()

        let queries: [(tool: String, args: [String: Any])] = [
            ("search_calendar", ["query": "", "days_ahead": 7]),
            ("search_calendar", ["query": "dentist", "days_ahead": 7]),
            ("query_health_data", ["metric": "heart_rate", "days": 30]),
            ("query_health_data", ["metric": "blood_pressure", "days": 30]),
            ("query_health_data", ["metric": "steps", "days": 7]),
            ("query_health_data", ["metric": "active_energy", "days": 14]),
            ("query_health_data", ["metric": "weight", "days": 30]),
            ("query_health_data", ["metric": "sleep", "days": 7]),
            ("search_reminders", ["query": ""]),
            ("search_reminders", ["query": "health"]),
        ]

        for (tool, args) in queries {
            let (result, _) = await agent.runSingleToolQuery(
                query: "test", selectedTool: tool, arguments: args
            )
            let data = try #require(result.data.data(using: .utf8), "Tool \(tool) result should be valid UTF-8")
            let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(parsed != nil, "Tool \(tool) with args \(args) should return valid JSON")
        }
    }

    @Test("Health metric JSON always includes metric name and unit")
    func healthMetricsHaveRequiredFields() async throws {
        let (agent, _) = makeE2EAgent()

        let metrics = ["heart_rate", "blood_pressure", "steps", "active_energy", "weight", "sleep"]

        for metric in metrics {
            let (result, _) = await agent.runSingleToolQuery(
                query: "test", selectedTool: "query_health_data",
                arguments: ["metric": metric, "days": 30]
            )
            #expect(result.success, "Metric \(metric) should succeed")
            let metricData = try #require(result.data.data(using: .utf8), "Metric \(metric) should be valid UTF-8")
            let json = try #require(JSONSerialization.jsonObject(with: metricData) as? [String: Any], "Metric \(metric) should be valid JSON")
            #expect(json["metric"] as? String == metric, "JSON should contain metric name")
            #expect(json["unit"] != nil, "Metric \(metric) should have a unit")
        }
    }

    @Test("Calendar JSON includes event structure fields")
    func calendarEventStructure() async throws {
        let (agent, _) = makeE2EAgent()
        let (result, _) = await agent.runSingleToolQuery(
            query: "test", selectedTool: "search_calendar",
            arguments: ["query": "", "days_ahead": 7]
        )

        let calData = try #require(result.data.data(using: .utf8))
        let json = try #require(JSONSerialization.jsonObject(with: calData) as? [String: Any])
        #expect(json["count"] != nil)
        #expect(json["events"] != nil)
        #expect(json["days_ahead"] != nil)
        #expect(json["calendar_filter"] != nil)

        let events = json["events"] as? [[String: Any]]
        #expect(events != nil)
        if let event = events?.first {
            #expect(event["title"] != nil)
            #expect(event["start_time"] != nil)
            #expect(event["location"] != nil)
        }
    }

    @Test("Reminders JSON includes reminder structure fields")
    func reminderStructure() async throws {
        let (agent, _) = makeE2EAgent()
        let (result, _) = await agent.runSingleToolQuery(
            query: "test", selectedTool: "search_reminders",
            arguments: ["query": ""]
        )

        let remData = try #require(result.data.data(using: .utf8))
        let json = try #require(JSONSerialization.jsonObject(with: remData) as? [String: Any])
        #expect(json["count"] != nil)
        #expect(json["reminders"] != nil)

        let reminders = json["reminders"] as? [[String: Any]]
        #expect(reminders != nil)
        if let reminder = reminders?.first {
            #expect(reminder["title"] != nil)
            #expect(reminder["list"] != nil)
            #expect(reminder["is_completed"] != nil)
        }
    }
}

// MARK: - E2E: Error Handling Pipeline

@Suite("E2E — Error Handling")
struct E2EErrorHandlingTests {

    @Test("Unknown tool name returns descriptive error")
    func unknownToolError() async {
        let (agent, _) = makeE2EAgent()
        let (result, _) = await agent.runSingleToolQuery(
            query: "test", selectedTool: "nonexistent_tool",
            arguments: [:]
        )

        #expect(!result.success)
        #expect(result.displaySummary.contains("Unknown tool"))
        #expect(result.error!.contains("not registered"))
        #expect(result.error!.contains("search_calendar"))  // lists available tools
    }

    @Test("Unknown health metric returns failure with metric name")
    func unknownHealthMetric() async {
        let (agent, _) = makeE2EAgent()
        let (result, _) = await agent.runSingleToolQuery(
            query: "test", selectedTool: "query_health_data",
            arguments: ["metric": "body_temperature", "days": 7]
        )

        #expect(!result.success)
        #expect(result.displaySummary.contains("body_temperature"))
    }
}

// MARK: - E2E: bestMatch Tool Selection Accuracy

@Suite("E2E — Tool Selection Accuracy")
struct E2EToolSelectionTests {

    @Test("bestMatch routes health queries to health tool",
          arguments: [
            "What is my heart rate?",
            "Show me my blood pressure",
            "How many steps did I take today?",
            "What's my weight?",
            "Sleep data for the past week",
            "active energy burned",
          ])
    func healthQueriesRouteCorrectly(query: String) {
        let (agent, _) = makeE2EAgent()
        let selected = agent.bestMatch(
            toolNames: ["search_calendar", "query_health_data", "search_reminders"],
            forQuery: query
        )
        #expect(selected == "query_health_data", "Query '\(query)' should route to health tool")
    }

    @Test("bestMatch routes reminder queries to reminders tool",
          arguments: [
            "Show me my reminders",
            "What tasks do I have?",
            "Search my reminders for health questions",
          ])
    func reminderQueriesRouteCorrectly(query: String) {
        let (agent, _) = makeE2EAgent()
        let selected = agent.bestMatch(
            toolNames: ["search_calendar", "query_health_data", "search_reminders"],
            forQuery: query
        )
        #expect(selected == "search_reminders", "Query '\(query)' should route to reminders tool")
    }

    @Test("bestMatch routes calendar queries to calendar tool",
          arguments: [
            "What appointments do I have?",
            "Search my calendar for next week",
            "Do I have any upcoming events?",
          ])
    func calendarQueriesRouteCorrectly(query: String) {
        let (agent, _) = makeE2EAgent()
        let selected = agent.bestMatch(
            toolNames: ["search_calendar", "query_health_data", "search_reminders"],
            forQuery: query
        )
        #expect(selected == "search_calendar", "Query '\(query)' should route to calendar tool")
    }
}

// MARK: - E2E: Synthesis Prompt Quality

@Suite("E2E — Synthesis Prompt Quality")
struct E2ESynthesisPromptTests {

    @Test("Single-tool synthesis prompt follows expected format")
    func singleToolSynthesisFormat() async {
        let (agent, _) = makeE2EAgent()
        let (_, prompt) = await agent.runSingleToolQuery(
            query: "heart rate", selectedTool: "query_health_data",
            arguments: ["metric": "heart_rate", "days": 30]
        )

        // Format: [Tool Result: <name>]\n<json>\n\n
        #expect(prompt.hasPrefix("[Tool Result: query_health_data]\n"))
        #expect(prompt.hasSuffix("\n\n"))

        // The JSON in the prompt should be parseable
        let jsonPart = prompt
            .replacingOccurrences(of: "[Tool Result: query_health_data]\n", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let parsed = try? JSONSerialization.jsonObject(with: jsonPart.data(using: .utf8)!)
        #expect(parsed != nil, "JSON portion of synthesis prompt should be parseable")
    }

    @Test("Meeting prep synthesis prompt has all required sections")
    func meetingPrepSynthesisStructure() async {
        let (agent, _) = makeE2EAgent()
        let query = "Prepare me for my meeting with Dr. Muller"
        let (prompt, _) = await agent.runMeetingPrep(query: query)

        // Must have: user question, 3 tool results, formatting instructions
        let requiredSections = [
            "The user asked:",
            "[Tool Result: search_calendar]",
            "[Tool Result: query_health_data]",
            "[Tool Result: search_reminders]",
            "Summarize this as a meeting preparation briefing",
            "bullet points",
            "health data with units",
            "pending reminders",
            "Be concise",
        ]
        for section in requiredSections {
            #expect(prompt.contains(section), "Synthesis prompt missing: '\(section)'")
        }
    }
}

// MARK: - Model-in-the-Loop (Future)
//
// The tests below are designed to run once the test target is configured
// to link against LeapSDK and host the LocalLife.app. They exercise the
// actual LFM 2.5 1.2B model's tool calling and synthesis capabilities.
//
// To enable:
// 1. In Xcode, select LocalLifeTests target → General → Host Application → LocalLife
// 2. Add LeapSDK to LocalLifeTests target's "Link Binary With Libraries"
// 3. Uncomment the tests below
// 4. Run on a physical device (model requires Neural Engine)
//
// @Suite("E2E — Model-in-the-Loop (On-Device)")
// struct ModelInTheLoopTests {
//
//     @Test("Model loads successfully on device")
//     func modelLoads() async throws {
//         let modelManager = ModelManager()
//         await modelManager.checkModelAvailability()
//         try await modelManager.loadModel(systemPrompt: AgentConfiguration.systemPrompt)
//         #expect(modelManager.isReady)
//     }
//
//     @Test("Model routes 'What is my heart rate?' to query_health_data")
//     func modelRoutesHeartRate() async throws {
//         // Setup
//         let modelManager = ModelManager()
//         let toolRegistry = ToolRegistry()
//         let healthKit = HealthKitTool()
//         healthKit.demoMode = true
//         toolRegistry.register(healthKit)
//         toolRegistry.register(CalendarTool())
//         toolRegistry.register(RemindersTool())
//
//         await modelManager.checkModelAvailability()
//         try await modelManager.loadModel(systemPrompt: AgentConfiguration.systemPrompt)
//         for fn in toolRegistry.asLeapFunctions() {
//             modelManager.registerFunction(fn)
//         }
//
//         let agentLoop = AgentLoop(toolRegistry: toolRegistry, modelManager: modelManager)
//         var toolsUsed: [String] = []
//
//         _ = try await agentLoop.run(message: "What is my heart rate?") { event in
//             if case .toolCompleted(let name, _, _, _, _, _) = event {
//                 toolsUsed.append(name)
//             }
//         }
//
//         #expect(toolsUsed.contains("query_health_data"))
//         #expect(!toolsUsed.contains("search_calendar"))
//     }
//
//     @Test("Model synthesis includes heart rate value and unit")
//     func modelSynthesisQuality() async throws {
//         // Similar setup as above, then:
//         // let response = try await agentLoop.run(message: "What is my heart rate?") { _ in }
//         // #expect(response.contains("72") || response.contains("bpm"))
//     }
//
//     @Test("Meeting prep produces comprehensive briefing")
//     func modelMeetingPrep() async throws {
//         // Full meeting prep with real model:
//         // let response = try await agentLoop.run(message: "Prepare me for my meeting with Dr. Muller") { _ in }
//         // #expect(response.contains("Dr. Muller") || response.contains("Cardiology"))
//         // #expect(response.contains("heart rate") || response.contains("72"))
//         // #expect(response.contains("reminder") || response.contains("Metoprolol"))
//     }
// }
