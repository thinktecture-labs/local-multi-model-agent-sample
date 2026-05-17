import Foundation

public struct ToolResult {
    public let success: Bool
    public let data: String // JSON string for feeding back to the model
    public let displaySummary: String // Human-readable summary for UI
    public let error: String?
}

/// Typed schema for a tool parameter, replacing untyped `[String: Any]` dictionaries.
public struct ToolParameterSchema {
    public enum ParamType: String {
        case string
        case integer
    }

    public let name: String
    public let type: ParamType
    public let description: String
    public let isOptional: Bool
}

/// A tool that can be registered with the LEAP SDK and executed by the agent.
public protocol Tool {
    /// Unique name matching the LeapFunction registration.
    var name: String { get }

    /// Human-readable description for the model's function calling.
    var description: String { get }

    /// Typed parameter definitions for this tool.
    var parameters: [ToolParameterSchema] { get }

    /// Execute with parsed arguments from the model's function call.
    func execute(arguments: [String: Any]) async -> ToolResult
}
