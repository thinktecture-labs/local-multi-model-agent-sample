// ChatMessageTests.swift
// Tests for ChatMessage, ToolCallInfo, and AgentConfiguration types.

import Testing
import Foundation
import LocalLife

// MARK: - Duplicated types for testing

private enum TestToolCallStatus: Equatable {
    case pending
    case executing
    case completed
    case failed(String)

    var isFinished: Bool {
        switch self {
        case .completed, .failed: return true
        default: return false
        }
    }
}

private struct TestToolCallInfo: Identifiable, Equatable {
    let id = UUID()
    let toolName: String
    let status: TestToolCallStatus
    let durationMs: Double?
    let resultSummary: String?
    let arguments: [String: String]?
    let resultData: String?

    static func == (lhs: TestToolCallInfo, rhs: TestToolCallInfo) -> Bool {
        lhs.id == rhs.id
            && lhs.toolName == rhs.toolName
            && lhs.status == rhs.status
            && lhs.durationMs == rhs.durationMs
            && lhs.resultSummary == rhs.resultSummary
            && lhs.arguments == rhs.arguments
            && lhs.resultData == rhs.resultData
    }
}

// MARK: - Tests

@Suite("ToolCallStatus")
struct ToolCallStatusTests {

    @Test("pending is not finished")
    func pendingNotFinished() {
        #expect(!TestToolCallStatus.pending.isFinished)
    }

    @Test("executing is not finished")
    func executingNotFinished() {
        #expect(!TestToolCallStatus.executing.isFinished)
    }

    @Test("completed is finished")
    func completedIsFinished() {
        #expect(TestToolCallStatus.completed.isFinished)
    }

    @Test("failed is finished")
    func failedIsFinished() {
        #expect(TestToolCallStatus.failed("error").isFinished)
    }

    @Test("completed equals completed")
    func completedEqualsCompleted() {
        #expect(TestToolCallStatus.completed == TestToolCallStatus.completed)
    }

    @Test("failed with same message equals")
    func failedSameMessageEquals() {
        #expect(TestToolCallStatus.failed("timeout") == TestToolCallStatus.failed("timeout"))
    }

    @Test("failed with different messages not equal")
    func failedDifferentMessagesNotEqual() {
        #expect(TestToolCallStatus.failed("timeout") != TestToolCallStatus.failed("network error"))
    }

    @Test("pending does not equal executing")
    func pendingNotEqualExecuting() {
        #expect(TestToolCallStatus.pending != TestToolCallStatus.executing)
    }
}

@Suite("ToolCallInfo Equatable")
struct ToolCallInfoTests {

    @Test("Same instance equals itself")
    func sameInstanceEquals() {
        let info = TestToolCallInfo(
            toolName: "search_calendar",
            status: .completed,
            durationMs: 42.0,
            resultSummary: "1 event found",
            arguments: ["query": "muller"],
            resultData: "{}"
        )
        #expect(info == info)
    }

    @Test("Different instances with different IDs are not equal")
    func differentIdsNotEqual() {
        let a = TestToolCallInfo(
            toolName: "search_calendar", status: .completed,
            durationMs: 42.0, resultSummary: "1 event",
            arguments: nil, resultData: nil
        )
        let b = TestToolCallInfo(
            toolName: "search_calendar", status: .completed,
            durationMs: 42.0, resultSummary: "1 event",
            arguments: nil, resultData: nil
        )
        // Different UUID, so not equal
        #expect(a != b)
    }

    @Test("ToolCallInfo with nil optionals is valid")
    func nilOptionalsValid() {
        let info = TestToolCallInfo(
            toolName: "search_calendar",
            status: .pending,
            durationMs: nil,
            resultSummary: nil,
            arguments: nil,
            resultData: nil
        )
        #expect(info.toolName == "search_calendar")
        #expect(info.durationMs == nil)
        #expect(info.resultSummary == nil)
    }
}

@Suite("AgentConfiguration Validation")
struct AgentConfigurationTests {

    @Test("System prompt mentions all three tools")
    func systemPromptMentionsAllTools() {
        let prompt = AgentConfiguration.systemPrompt
        #expect(prompt.contains("search_calendar"))
        #expect(prompt.contains("query_health_data"))
        #expect(prompt.contains("search_reminders"))
    }

    @Test("System prompt instructs tool calling")
    func systemPromptRequiresToolCall() {
        let prompt = AgentConfiguration.systemPrompt
        #expect(prompt.contains("MUST call exactly one tool"))
    }

    @Test("System prompt has keyword-based tool routing")
    func systemPromptHasToolRouting() {
        let prompt = AgentConfiguration.systemPrompt
        #expect(prompt.contains("TOOL ROUTING"))
        #expect(prompt.contains("search_calendar →"))
        #expect(prompt.contains("query_health_data →"))
        #expect(prompt.contains("search_reminders →"))
    }

    @Test("System prompt has negative disambiguation")
    func systemPromptDisambiguates() {
        let prompt = AgentConfiguration.systemPrompt
        #expect(prompt.contains("never search_reminders"))
    }

    @Test("Temperature is 0 for deterministic output")
    func temperatureIsDeterministic() {
        #expect(AgentConfiguration.temperature == 0.0)
    }

    @Test("System prompt instructs concise formatting")
    func systemPromptInstructsConcise() {
        let prompt = AgentConfiguration.systemPrompt
        #expect(prompt.contains("bullet points"))
        #expect(prompt.contains("units"))
    }
}
