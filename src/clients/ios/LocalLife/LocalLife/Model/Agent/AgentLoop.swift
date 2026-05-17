import Foundation
import LeapSDK

/// Events emitted during the agent loop for real-time UI updates.
public enum AgentEvent {
    case textChunk(String)
    case toolCallsReceived([String])
    case toolExecuting(name: String)
    case toolCompleted(name: String, success: Bool, summary: String, durationMs: Double, arguments: [String: String], resultData: String)
    case generationComplete(tokensUsed: Int)
    case warning(String)
}

public final class AgentLoop {
    private let toolRegistry: ToolRegistry
    private let modelManager: ModelManager

    public init(toolRegistry: ToolRegistry, modelManager: ModelManager) {
        self.toolRegistry = toolRegistry
        self.modelManager = modelManager
    }

    /// Run the full agent loop: send message, handle tool calls, feed results back, synthesize.
    ///
    /// The `onEvent` callback provides real-time UI updates during execution.
    public func run(
        message: String,
        onEvent: @escaping (AgentEvent) -> Void
    ) async throws -> String {
        // Fresh conversation per query — the 1.2B model's tool selection degrades
        // when prior conversation history is present, even with resetHistory=true.
        // Chat history is maintained at the app layer (ChatViewModel.messages).
        modelManager.resetConversation()

        var options = GenerationOptions(
            temperature: AgentConfiguration.temperature
        )
        options.repetitionPenalty = 1.05

        var currentMessage = message
        var preflightToolResults = ""

        // Pre-flight: for "prepare for meeting" queries, programmatically call all 3
        // tools since the 1.2B model can't reliably plan multi-tool orchestration.
        // Uses ReWOO pattern: plan in code, execute, then synthesize.
        let isMeetingPrep = Self.isMeetingPrepQuery(message)
        if isMeetingPrep {
            let toolResults = await executePreflightTools(userMessage: message, onEvent: onEvent)
            preflightToolResults = toolResults
            currentMessage = """
            The user asked: \(message)

            Here are the ONLY facts from their personal data:

            \(toolResults)

            Write a concise meeting preparation briefing.
            RULES:
            - ONLY state facts that appear in the tool results above.
            - Do NOT invent medications, diagnoses, or values not shown above.
            - Focus on the specific appointment the user asked about.
            - Use bullet points. Include units for all health values.
            - List any pending reminders.
            """
        }

        // --- Round 1: Model generates, may call tools ---
        var accumulatedText = ""
        var pendingToolCalls: [LeapFunctionCall] = []

        let stream = modelManager.generateResponse(message: currentMessage, options: options)
        for try await response in stream {
            switch response {
            case .chunk(let text):
                accumulatedText += text
                onEvent(.textChunk(text))
            case .functionCall(let calls):
                pendingToolCalls = calls
            case .complete(let completion):
                onEvent(.generationComplete(tokensUsed: Int(completion.stats?.totalTokens ?? 0)))
            default:
                break
            }
        }

        // Meeting prep: model already has all data from preflight, skip Round 2.
        // This eliminates the variance from optional model tool calls in Round 1
        // and ensures consistent tool-pill rendering (always 3 preflight pills).
        if isMeetingPrep {
            return Self.sanitizeSynthesis(accumulatedText, toolResults: preflightToolResults)
        }

        // No tool calls → model answered directly, done
        if pendingToolCalls.isEmpty {
            return accumulatedText
        }

        // Guard: pick the single best-matching tool when model over-fires
        if pendingToolCalls.count > 1 {
            pendingToolCalls = [bestMatch(pendingToolCalls, forQuery: message)]
        }

        onEvent(.toolCallsReceived(pendingToolCalls.map { $0.name }))

        // --- Execute tool(s) ---
        var toolResultsMessage = ""
        for call in pendingToolCalls {
            onEvent(.toolExecuting(name: call.name))
            let start = CFAbsoluteTimeGetCurrent()
            let args: [String: Any] = call.arguments.compactMapValues { value -> Any? in
                guard let value else { return nil }
                if let str = value as? String {
                    if let intValue = Int(str) { return intValue }
                    if let doubleValue = Double(str) { return doubleValue }
                    return str
                }
                return value
            }
            let result = await toolRegistry.execute(name: call.name, arguments: args)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let argStrings = args.mapValues { "\($0)" }
            onEvent(.toolCompleted(
                name: call.name,
                success: result.success,
                summary: result.displaySummary,
                durationMs: elapsed,
                arguments: argStrings,
                resultData: result.data
            ))
            toolResultsMessage += "[Tool Result: \(call.name)]\n\(result.data)\n\n"
        }

        // --- Round 2: Synthesis — model sees tool results, generates answer ---
        // Keep history (user question + tool call) so the model has full context.
        // Ignore any further tool calls to prevent badge accumulation.
        var synthesisOptions = options
        synthesisOptions.resetHistory = false
        var synthesisText = ""
        let synthStream = modelManager.generateResponse(message: toolResultsMessage, options: synthesisOptions)
        for try await response in synthStream {
            switch response {
            case .chunk(let text):
                synthesisText += text
                onEvent(.textChunk(text))
            case .complete(let completion):
                onEvent(.generationComplete(tokensUsed: Int(completion.stats?.totalTokens ?? 0)))
            default:
                break // Ignore .functionCall — synthesis only
            }
        }

        return synthesisText
    }

    /// Execute a single tool call (used for testing tools independently).
    func executeTool(name: String, arguments: [String: Any]) async -> ToolResult {
        return await toolRegistry.execute(name: name, arguments: arguments)
    }

    // MARK: - Pre-flight orchestration for multi-tool queries

    /// Pick the tool call whose registered metadata best matches the query.
    /// Compares query words against each tool's name, description, and parameter
    /// descriptions from the registry — adapts automatically if tools change.
    private func bestMatch(
        _ calls: [LeapFunctionCall], forQuery query: String
    ) -> LeapFunctionCall {
        let queryWords = Set(query.lowercased().split(whereSeparator: { !$0.isLetter }))
        var best = calls[0]
        var bestScore = -1
        for call in calls {
            if let tool = toolRegistry.get(call.name) {
                // Build word set from tool name, description, and param descriptions
                let metadata = "\(tool.name) \(tool.description) \(tool.parameters.map(\.description).joined(separator: " "))"
                let metaWords = Set(metadata.lowercased().split(whereSeparator: { !$0.isLetter }))
                let score = queryWords.intersection(metaWords).count
                if score > bestScore {
                    bestScore = score
                    best = call
                }
            }
        }
        return best
    }

    public static func isMeetingPrepQuery(_ message: String) -> Bool {
        let lower = message.lowercased()
        let hasMeetingIntent = lower.contains("prepare") || lower.contains("meeting")
            || lower.contains("appointment")
        let hasDoctorContext = lower.contains("dr.") || lower.contains("dr ")
            || lower.contains("doctor") || lower.contains("arzt")
        return hasMeetingIntent && hasDoctorContext
    }

    /// Extract a person/event name from the user query for targeted calendar search.
    /// Strips periods so "dr pepper" matches both "Dr Pepper" and "Dr. Pepper".
    static func extractCalendarQuery(from message: String) -> String {
        let lower = message.lowercased()
        // "meeting with Dr. Pepper tomorrow" → "dr pepper"
        if let withRange = lower.range(of: "with ") {
            var nameEnd = lower.endIndex
            for stop in ["tomorrow", "today", "next ", "this ", "on "] {
                if let r = lower.range(of: stop, range: withRange.upperBound..<lower.endIndex),
                   r.lowerBound < nameEnd {
                    nameEnd = r.lowerBound
                }
            }
            let name = String(lower[withRange.upperBound..<nameEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: ".", with: "")
            if !name.isEmpty { return name }
        }
        // Fallback: "Dr. Pepper" without a preceding "with"
        if let drRange = lower.range(of: "dr. ") ?? lower.range(of: "dr ") {
            let words = String(lower[drRange.lowerBound...]).split(separator: " ")
            if words.count >= 2 {
                return "\(words[0]) \(words[1])".replacingOccurrences(of: ".", with: "")
            }
        }
        return ""
    }

    private static let confabulationTerms = [
        "metoprolol", "lisinopril", "amlodipine", "atorvastatin", "metformin",
        "losartan", "hydrochlorothiazide", "simvastatin", "warfarin", "clopidogrel",
        "beta-blocker", "beta blocker", "ace inhibitor",
        "medication change", "medication switch", "switching from",
    ]

    /// Strip lines containing clinical terms that don't appear in tool results.
    static func sanitizeSynthesis(_ text: String, toolResults: String) -> String {
        let toolLower = toolResults.lowercased()
        return text.components(separatedBy: "\n")
            .filter { line in
                let lineLower = line.lowercased()
                return !confabulationTerms.contains { term in
                    lineLower.contains(term) && !toolLower.contains(term)
                }
            }
            .joined(separator: "\n")
    }

    private func executePreflightTools(
        userMessage: String,
        onEvent: @escaping (AgentEvent) -> Void
    ) async -> String {
        let calendarQuery = Self.extractCalendarQuery(from: userMessage)
        let daysAhead = 7

        onEvent(.toolCallsReceived(["search_calendar", "query_health_data", "search_reminders"]))
        var results = ""

        // Calendar — targeted query first, fallback to broad if no results
        onEvent(.toolExecuting(name: "search_calendar"))
        var start = CFAbsoluteTimeGetCurrent()
        var result = await toolRegistry.execute(
            name: "search_calendar",
            arguments: ["query": calendarQuery, "days_ahead": daysAhead]
        )
        // Fallback: if targeted query found nothing, retry without query filter
        if !calendarQuery.isEmpty && result.data.contains("\"count\":0") {
            result = await toolRegistry.execute(
                name: "search_calendar",
                arguments: ["query": "", "days_ahead": daysAhead]
            )
        }
        var elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        onEvent(.toolCompleted(
            name: "search_calendar", success: result.success,
            summary: result.displaySummary, durationMs: elapsed,
            arguments: ["query": calendarQuery.isEmpty ? "(all)" : calendarQuery, "days_ahead": "\(daysAhead)"],
            resultData: result.data
        ))
        results += "[Tool Result: search_calendar]\n\(result.data)\n\n"

        // Health data — query all relevant metrics to prevent confabulation
        onEvent(.toolExecuting(name: "query_health_data"))
        start = CFAbsoluteTimeGetCurrent()
        var healthData = ""
        for metric in ["heart_rate", "blood_pressure", "weight", "steps"] {
            let r = await toolRegistry.execute(
                name: "query_health_data",
                arguments: ["metric": metric, "days": 30]
            )
            healthData += r.data + "\n"
        }
        elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        onEvent(.toolCompleted(
            name: "query_health_data", success: true,
            summary: "Health: heart rate, blood pressure, weight, steps",
            durationMs: elapsed,
            arguments: ["metrics": "heart_rate,blood_pressure,weight,steps", "days": "30"],
            resultData: healthData
        ))
        results += "[Tool Result: query_health_data]\n\(healthData)\n\n"

        // Reminders
        onEvent(.toolExecuting(name: "search_reminders"))
        start = CFAbsoluteTimeGetCurrent()
        result = await toolRegistry.execute(
            name: "search_reminders", arguments: ["query": ""]
        )
        elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        onEvent(.toolCompleted(
            name: "search_reminders", success: result.success,
            summary: result.displaySummary, durationMs: elapsed,
            arguments: ["query": ""],
            resultData: result.data
        ))
        results += "[Tool Result: search_reminders]\n\(result.data)\n\n"

        return results
    }
}
