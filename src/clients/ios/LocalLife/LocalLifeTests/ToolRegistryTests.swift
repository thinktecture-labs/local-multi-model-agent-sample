// ToolRegistryTests.swift
// Tests for ToolRegistry: registration, lookup, execution, and thread safety.

import Testing
import Foundation

// MARK: - Duplicated minimal types (test target cannot import app module)

private struct TestToolResult {
    let success: Bool
    let data: String
    let displaySummary: String
    let error: String?
}

private struct TestToolParameterSchema {
    enum ParamType: String { case string; case integer }
    let name: String
    let type: ParamType
    let description: String
    let isOptional: Bool
}

private protocol TestTool {
    var name: String { get }
    var description: String { get }
    var parameters: [TestToolParameterSchema] { get }
    func execute(arguments: [String: Any]) async -> TestToolResult
}

/// Thread-safe registry mirroring the production ToolRegistry logic.
private final class TestToolRegistry: @unchecked Sendable {
    private var tools: [String: any TestTool] = [:]
    private let lock = NSLock()

    func register(_ tool: any TestTool) {
        lock.withLock { tools[tool.name] = tool }
    }

    func get(_ name: String) -> (any TestTool)? {
        lock.withLock { tools[name] }
    }

    var allTools: [any TestTool] {
        lock.withLock { Array(tools.values) }
    }

    func execute(name: String, arguments: [String: Any]) async -> TestToolResult {
        let tool = lock.withLock { tools[name] }
        guard let tool else {
            let available = lock.withLock { tools.keys.joined(separator: ", ") }
            return TestToolResult(
                success: false, data: "{}",
                displaySummary: "Unknown tool: \(name)",
                error: "Tool '\(name)' not registered. Available: \(available)"
            )
        }
        return await tool.execute(arguments: arguments)
    }
}

// MARK: - Mock Tools

private final class MockCalendarTool: TestTool {
    let name = "search_calendar"
    let description = "Search calendar for events"
    var parameters: [TestToolParameterSchema] {
        [
            TestToolParameterSchema(name: "query", type: .string, description: "Search term", isOptional: true),
            TestToolParameterSchema(name: "days_ahead", type: .integer, description: "Days ahead", isOptional: true),
        ]
    }
    var executeCallCount = 0
    var lastArguments: [String: Any]?

    func execute(arguments: [String: Any]) async -> TestToolResult {
        executeCallCount += 1
        lastArguments = arguments
        return TestToolResult(
            success: true,
            data: "{\"count\":1,\"events\":[{\"title\":\"Dr. Muller - Cardiology\"}]}",
            displaySummary: "1 event: Dr. Muller - Cardiology",
            error: nil
        )
    }
}

private final class MockHealthTool: TestTool {
    let name = "query_health_data"
    let description = "Query health data from HealthKit"
    var parameters: [TestToolParameterSchema] {
        [
            TestToolParameterSchema(name: "metric", type: .string, description: "Health metric", isOptional: false),
            TestToolParameterSchema(name: "days", type: .integer, description: "Past days", isOptional: true),
        ]
    }

    func execute(arguments: [String: Any]) async -> TestToolResult {
        let metric = arguments["metric"] as? String ?? "unknown"
        return TestToolResult(
            success: true,
            data: "{\"metric\":\"\(metric)\",\"avg\":72}",
            displaySummary: "\(metric): avg 72",
            error: nil
        )
    }
}

private final class MockRemindersTool: TestTool {
    let name = "search_reminders"
    let description = "Search reminders for tasks"
    var parameters: [TestToolParameterSchema] {
        [
            TestToolParameterSchema(name: "query", type: .string, description: "Search term", isOptional: false),
        ]
    }

    func execute(arguments: [String: Any]) async -> TestToolResult {
        return TestToolResult(
            success: true,
            data: "{\"count\":3,\"reminders\":[]}",
            displaySummary: "3 reminders found",
            error: nil
        )
    }
}

// MARK: - Tests

@Suite("ToolRegistry")
struct ToolRegistryTests {

    @Test("Register and retrieve a tool by name")
    func registerAndGet() {
        let registry = TestToolRegistry()
        let calendar = MockCalendarTool()
        registry.register(calendar)

        let retrieved = registry.get("search_calendar")
        #expect(retrieved != nil)
        #expect(retrieved?.name == "search_calendar")
    }

    @Test("Get returns nil for unregistered tool")
    func getUnknownTool() {
        let registry = TestToolRegistry()
        #expect(registry.get("nonexistent") == nil)
    }

    @Test("Execute returns error for unknown tool")
    func executeUnknownTool() async {
        let registry = TestToolRegistry()
        registry.register(MockCalendarTool())

        let result = await registry.execute(name: "bogus_tool", arguments: [:])
        #expect(!result.success)
        #expect(result.displaySummary.contains("Unknown tool"))
        #expect(result.error?.contains("search_calendar") == true)
    }

    @Test("Execute dispatches to correct tool")
    func executeCorrectTool() async {
        let registry = TestToolRegistry()
        let calendar = MockCalendarTool()
        let health = MockHealthTool()
        registry.register(calendar)
        registry.register(health)

        let result = await registry.execute(
            name: "search_calendar",
            arguments: ["query": "muller", "days_ahead": 7]
        )
        #expect(result.success)
        #expect(calendar.executeCallCount == 1)
        #expect(result.data.contains("Dr. Muller"))
    }

    @Test("allTools returns all registered tools")
    func allToolsReturnsAll() {
        let registry = TestToolRegistry()
        registry.register(MockCalendarTool())
        registry.register(MockHealthTool())
        registry.register(MockRemindersTool())

        #expect(registry.allTools.count == 3)
    }

    @Test("Register overwrites tool with same name")
    func registerOverwrites() async {
        let registry = TestToolRegistry()
        let first = MockCalendarTool()
        let second = MockCalendarTool()
        registry.register(first)
        registry.register(second)

        _ = await registry.execute(name: "search_calendar", arguments: [:])
        #expect(first.executeCallCount == 0)
        #expect(second.executeCallCount == 1)
    }

    @Test("Concurrent registration is thread-safe")
    func concurrentRegistration() async {
        let registry = TestToolRegistry()

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    let tool = MockCalendarTool()
                    // Override name would require protocol change, so just register same tool
                    // This tests the lock doesn't deadlock under contention
                    if i % 3 == 0 {
                        registry.register(MockCalendarTool())
                    } else if i % 3 == 1 {
                        registry.register(MockHealthTool())
                    } else {
                        registry.register(MockRemindersTool())
                    }
                }
            }
        }

        // Should not crash; final state has 3 tools (last write wins per name)
        #expect(registry.allTools.count == 3)
    }

    @Test("Arguments are forwarded to tool")
    func argumentsForwarded() async {
        let registry = TestToolRegistry()
        let calendar = MockCalendarTool()
        registry.register(calendar)

        _ = await registry.execute(
            name: "search_calendar",
            arguments: ["query": "dentist", "days_ahead": 14]
        )

        #expect(calendar.lastArguments?["query"] as? String == "dentist")
        #expect(calendar.lastArguments?["days_ahead"] as? Int == 14)
    }
}
