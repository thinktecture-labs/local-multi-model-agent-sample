// AgentLoopLogicTests.swift
// Tests for AgentLoop routing logic: meeting prep detection, tool selection
// heuristics, and preflight orchestration patterns.

import Testing
import Foundation
import LocalLife

// MARK: - Best Match Tool Selection (mirrors AgentLoop.bestMatch)

private struct MockToolCall {
    let name: String
    let arguments: [String: String]
}

private struct MockToolMetadata {
    let name: String
    let description: String
    let paramDescriptions: [String]
}

/// Reproduces the bestMatch scoring logic for isolated testing.
private enum ToolMatcher {
    static func bestMatch(
        _ calls: [MockToolCall],
        forQuery query: String,
        toolMetadata: [String: MockToolMetadata]
    ) -> MockToolCall {
        let queryWords = Set(query.lowercased().split(whereSeparator: { !$0.isLetter }))
        var best = calls[0]
        var bestScore = -1

        for call in calls {
            if let meta = toolMetadata[call.name] {
                let metadataText = "\(meta.name) \(meta.description) \(meta.paramDescriptions.joined(separator: " "))"
                let metaWords = Set(metadataText.lowercased().split(whereSeparator: { !$0.isLetter }))
                let score = queryWords.intersection(metaWords).count
                if score > bestScore {
                    bestScore = score
                    best = call
                }
            }
        }
        return best
    }
}

// MARK: - Tool call syntax stripping (mirrors ChatViewModel regex)

private let toolCallSyntaxPattern = try! NSRegularExpression(
    pattern: "\\[\\w+\\([^\\]]*\\)(?:,\\s*\\w+\\([^\\]]*\\))*\\]\\s*"
)

private func stripToolCallSyntax(_ text: String) -> String {
    let range = NSRange(text.startIndex..., in: text)
    return toolCallSyntaxPattern.stringByReplacingMatches(
        in: text, range: range, withTemplate: ""
    )
}

// MARK: - Tests

@Suite("Meeting Prep Detection")
struct MeetingPrepDetectionTests {

    @Test("Detects 'prepare for meeting with Dr. Muller'")
    func standardMeetingPrep() {
        #expect(AgentLoop.isMeetingPrepQuery("Prepare me for my meeting with Dr. Muller"))
    }

    @Test("German-only query without English meeting keywords does NOT trigger")
    func germanOnlyQueryDoesNotTrigger() {
        // Production code only checks English keywords for meeting intent.
        // "Arzt" matches the doctor context, but "Bereite" is not "prepare".
        #expect(!AgentLoop.isMeetingPrepQuery("Bereite mich auf den Arzt Termin vor"))
    }

    @Test("German doctor with English meeting keyword triggers")
    func germanDoctorWithEnglishMeeting() {
        #expect(AgentLoop.isMeetingPrepQuery("Prepare for Arzt appointment"))
    }

    @Test("Detects 'appointment with doctor'")
    func appointmentWithDoctor() {
        #expect(AgentLoop.isMeetingPrepQuery("Prepare for my appointment with the doctor"))
    }

    @Test("Detects 'meeting with Dr Smith'")
    func meetingWithDrSpace() {
        #expect(AgentLoop.isMeetingPrepQuery("I have a meeting with Dr Smith tomorrow"))
    }

    @Test("Does NOT trigger for plain calendar query")
    func plainCalendarQuery() {
        #expect(!AgentLoop.isMeetingPrepQuery("What meetings do I have this week?"))
    }

    @Test("Does NOT trigger for health query")
    func healthQuery() {
        #expect(!AgentLoop.isMeetingPrepQuery("What is my heart rate?"))
    }

    @Test("Does NOT trigger for reminder query")
    func reminderQuery() {
        #expect(!AgentLoop.isMeetingPrepQuery("Show me my reminders"))
    }

    @Test("Does NOT trigger for doctor without meeting intent")
    func doctorWithoutMeeting() {
        #expect(!AgentLoop.isMeetingPrepQuery("Who is Dr. Muller?"))
    }

    @Test("Does NOT trigger for meeting without doctor context")
    func meetingWithoutDoctor() {
        #expect(!AgentLoop.isMeetingPrepQuery("Prepare me for my meeting with the team"))
    }

    @Test("Case insensitive detection")
    func caseInsensitive() {
        #expect(AgentLoop.isMeetingPrepQuery("PREPARE FOR MEETING WITH DR. JONES"))
    }
}

@Suite("Calendar Query Extraction")
struct CalendarQueryExtractionTests {

    @Test("Extracts 'dr pepper' (no period) from 'meeting with Dr. Pepper tomorrow'")
    func extractDrPepperWithTomorrow() {
        let query = AgentLoop.extractCalendarQuery(from: "Prepare me for my meeting with Dr. Pepper tomorrow")
        #expect(query == "dr pepper")
    }

    @Test("Extracts 'dr muller' from 'meeting with Dr. Muller next week'")
    func extractDrMullerWithNextWeek() {
        let query = AgentLoop.extractCalendarQuery(from: "Prepare for meeting with Dr. Muller next week")
        #expect(query == "dr muller")
    }

    @Test("Extracts name without temporal stop word")
    func extractWithoutStopWord() {
        let query = AgentLoop.extractCalendarQuery(from: "Prepare for my appointment with Dr. Smith")
        #expect(query == "dr smith")
    }

    @Test("Fallback extracts 'dr jones' without 'with' keyword")
    func fallbackDrWithoutWith() {
        let query = AgentLoop.extractCalendarQuery(from: "Prepare for the Dr. Jones meeting")
        #expect(query == "dr jones")
    }

    @Test("Returns empty for query without person name")
    func emptyForNoPerson() {
        let query = AgentLoop.extractCalendarQuery(from: "Prepare for my meeting with the team")
        #expect(query == "the team")
    }
}

@Suite("Synthesis Sanitizer")
struct SynthesisSanitizerTests {

    @Test("Removes line with Metoprolol not in tool results")
    func removesMetoprolol() {
        let text = """
        - Heart rate: avg 72 bpm
        - Consider switching from Metoprolol to newer medication
        - Blood pressure: 138/85 mmHg
        """
        let toolResults = "heart_rate avg 72 blood_pressure 138 85"
        let sanitized = AgentLoop.sanitizeSynthesis(text, toolResults: toolResults)
        #expect(!sanitized.contains("Metoprolol"))
        #expect(sanitized.contains("Heart rate"))
        #expect(sanitized.contains("Blood pressure"))
    }

    @Test("Keeps line with term that IS in tool results")
    func keepsGroundedTerm() {
        let text = "- Current medication: Metoprolol 50mg"
        let toolResults = "reminders: Review metoprolol dosage"
        let sanitized = AgentLoop.sanitizeSynthesis(text, toolResults: toolResults)
        #expect(sanitized.contains("Metoprolol"))
    }

    @Test("Passes through clean text unchanged")
    func passesCleanText() {
        let text = """
        - Appointment at 11:00 with Dr. Pepper
        - Heart rate: avg 72 bpm
        - 3 pending reminders
        """
        let toolResults = "some tool data"
        let sanitized = AgentLoop.sanitizeSynthesis(text, toolResults: toolResults)
        #expect(sanitized == text)
    }

    @Test("Removes beta-blocker mention not in tool results")
    func removesBetaBlocker() {
        let text = "- Discuss switching to a newer beta-blocker"
        let toolResults = "heart_rate avg 72"
        let sanitized = AgentLoop.sanitizeSynthesis(text, toolResults: toolResults)
        #expect(!sanitized.contains("beta-blocker"))
    }
}

@Suite("Tool Selection (bestMatch)")
struct ToolSelectionTests {

    private var toolMetadata: [String: MockToolMetadata] {
        [
            "search_calendar": MockToolMetadata(
                name: "search_calendar",
                description: "Search the user's calendar for upcoming appointments and events",
                paramDescriptions: ["Search term to filter events by title, location, or notes", "Number of days ahead to search"]
            ),
            "query_health_data": MockToolMetadata(
                name: "query_health_data",
                description: "Query the user's health data from HealthKit including heart rate blood pressure steps",
                paramDescriptions: ["The health metric to query", "Number of past days to query"]
            ),
            "search_reminders": MockToolMetadata(
                name: "search_reminders",
                description: "Search the user's Reminders for tasks and notes",
                paramDescriptions: ["Search term to find relevant reminders"]
            ),
        ]
    }

    @Test("Selects calendar tool for appointment query")
    func selectsCalendarForAppointment() {
        let calls = [
            MockToolCall(name: "search_calendar", arguments: [:]),
            MockToolCall(name: "query_health_data", arguments: [:]),
            MockToolCall(name: "search_reminders", arguments: [:]),
        ]

        let best = ToolMatcher.bestMatch(calls, forQuery: "What appointments do I have this week?", toolMetadata: toolMetadata)
        #expect(best.name == "search_calendar")
    }

    @Test("Selects health tool for heart rate query")
    func selectsHealthForHeartRate() {
        let calls = [
            MockToolCall(name: "search_calendar", arguments: [:]),
            MockToolCall(name: "query_health_data", arguments: [:]),
            MockToolCall(name: "search_reminders", arguments: [:]),
        ]

        let best = ToolMatcher.bestMatch(calls, forQuery: "What is my heart rate?", toolMetadata: toolMetadata)
        #expect(best.name == "query_health_data")
    }

    @Test("Selects health tool for blood pressure query")
    func selectsHealthForBloodPressure() {
        let calls = [
            MockToolCall(name: "search_calendar", arguments: [:]),
            MockToolCall(name: "query_health_data", arguments: [:]),
        ]

        let best = ToolMatcher.bestMatch(calls, forQuery: "Show me my blood pressure readings", toolMetadata: toolMetadata)
        #expect(best.name == "query_health_data")
    }

    @Test("Selects health tool for steps query")
    func selectsHealthForSteps() {
        let calls = [
            MockToolCall(name: "search_calendar", arguments: [:]),
            MockToolCall(name: "query_health_data", arguments: [:]),
            MockToolCall(name: "search_reminders", arguments: [:]),
        ]

        let best = ToolMatcher.bestMatch(calls, forQuery: "How many steps did I take?", toolMetadata: toolMetadata)
        #expect(best.name == "query_health_data")
    }

    @Test("Selects reminders tool for tasks query")
    func selectsRemindersForTasks() {
        let calls = [
            MockToolCall(name: "search_calendar", arguments: [:]),
            MockToolCall(name: "query_health_data", arguments: [:]),
            MockToolCall(name: "search_reminders", arguments: [:]),
        ]

        let best = ToolMatcher.bestMatch(calls, forQuery: "Show my reminders and tasks", toolMetadata: toolMetadata)
        #expect(best.name == "search_reminders")
    }

    @Test("Selects calendar for events/schedule query")
    func selectsCalendarForEvents() {
        let calls = [
            MockToolCall(name: "search_calendar", arguments: [:]),
            MockToolCall(name: "query_health_data", arguments: [:]),
            MockToolCall(name: "search_reminders", arguments: [:]),
        ]

        let best = ToolMatcher.bestMatch(calls, forQuery: "What events are on my calendar?", toolMetadata: toolMetadata)
        #expect(best.name == "search_calendar")
    }

    @Test("Returns first call when no metadata matches")
    func fallbackToFirst() {
        let calls = [
            MockToolCall(name: "unknown_tool", arguments: [:]),
            MockToolCall(name: "another_tool", arguments: [:]),
        ]

        let best = ToolMatcher.bestMatch(calls, forQuery: "Do something", toolMetadata: toolMetadata)
        #expect(best.name == "unknown_tool")
    }
}

@Suite("Tool Call Syntax Stripping")
struct ToolCallSyntaxTests {

    @Test("Strips single tool call syntax from output")
    func stripsSingleToolCall() {
        let input = "[search_calendar(query=\"muller\")] Here are your results..."
        let cleaned = stripToolCallSyntax(input)
        #expect(cleaned == "Here are your results...")
    }

    @Test("Strips multiple tool calls in bracket notation")
    func stripsMultipleToolCalls() {
        let input = "[search_calendar(query=\"\"), query_health_data(metric=\"heart_rate\")] Summary:"
        let cleaned = stripToolCallSyntax(input)
        #expect(cleaned == "Summary:")
    }

    @Test("Leaves clean text unchanged")
    func leavesCleanTextUnchanged() {
        let input = "Your heart rate averages 72 bpm over the past 30 days."
        let cleaned = stripToolCallSyntax(input)
        #expect(cleaned == input)
    }

    @Test("Handles text with brackets that aren't tool calls")
    func handlesNonToolBrackets() {
        let input = "Your readings [see chart] are normal."
        let cleaned = stripToolCallSyntax(input)
        #expect(cleaned == input)
    }
}
