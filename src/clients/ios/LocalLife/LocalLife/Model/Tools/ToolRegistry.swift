import Foundation
import LeapSDK
import os

public final class ToolRegistry: @unchecked Sendable {
    private var tools: [String: Tool] = [:]
    private let lock = NSLock()

    public init() {}

    public func register(_ tool: Tool) {
        lock.withLock { tools[tool.name] = tool }
    }

    func get(_ name: String) -> Tool? {
        lock.withLock { tools[name] }
    }

    var allTools: [Tool] {
        lock.withLock { Array(tools.values) }
    }

    /// Convert all registered tools to LeapFunction array for LEAP SDK registration.
    public func asLeapFunctions() -> [LeapFunction] {
        let snapshot = lock.withLock { Array(tools.values) }
        return snapshot.map { tool in
            let leapParams = tool.parameters.map { param in
                let paramType: LeapFunctionParameterType = param.type == .integer
                    ? .integer(IntegerType())
                    : .string(StringType())
                return LeapFunctionParameter(
                    name: param.name,
                    type: paramType,
                    description: param.description,
                    optional: param.isOptional
                )
            }
            return LeapFunction(
                name: tool.name,
                description: tool.description,
                parameters: leapParams
            )
        }
    }

    /// Execute a tool by name. Never throws — errors surfaced in ToolResult.
    func execute(name: String, arguments: [String: Any]) async -> ToolResult {
        let tool = lock.withLock { tools[name] }
        guard let tool else {
            let available = lock.withLock { tools.keys.joined(separator: ", ") }
            return ToolResult(
                success: false,
                data: "{}",
                displaySummary: "Unknown tool: \(name)",
                error: "Tool '\(name)' not registered. Available: \(available)"
            )
        }
        return await tool.execute(arguments: arguments)
    }
}
