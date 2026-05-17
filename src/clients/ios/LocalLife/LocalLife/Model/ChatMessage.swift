import Foundation

enum MessageRole: String, Codable {
    case user
    case assistant
}

enum ToolCallStatus: Equatable {
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

struct ToolCallInfo: Identifiable, Equatable {
    let id = UUID()
    let toolName: String
    let status: ToolCallStatus
    let durationMs: Double?
    let resultSummary: String?
    let arguments: [String: String]?
    let resultData: String?

    static func == (lhs: ToolCallInfo, rhs: ToolCallInfo) -> Bool {
        lhs.id == rhs.id
            && lhs.toolName == rhs.toolName
            && lhs.status == rhs.status
            && lhs.durationMs == rhs.durationMs
            && lhs.resultSummary == rhs.resultSummary
            && lhs.arguments == rhs.arguments
            && lhs.resultData == rhs.resultData
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: MessageRole
    var content: String
    let timestamp: Date
    var toolCalls: [ToolCallInfo]?
    var isStreaming: Bool
}
