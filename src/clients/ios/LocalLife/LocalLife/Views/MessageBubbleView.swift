import SwiftUI

struct MessageBubbleView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                formattedContent
                    .padding(12)
                    .background(bubbleColor)
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(toolCalls) { call in
                            ToolBadgeView(call: call)
                        }
                    }
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
        .id(message.id)
    }

    /// Build formatted Text with bold rendering and proper line breaks.
    /// Uses SwiftUI Text concatenation instead of AttributedString(markdown:) which
    /// drops paragraph breaks from \n\n inserted by our regex fixes.
    private var formattedContent: Text {
        guard message.role == .assistant else {
            return Text(message.content)
        }

        var text = message.content

        // Insert line break before bold labels: "bpm**Recent" → "bpm\n**Recent"
        text = text.replacingOccurrences(
            of: "([^\n])\\*\\*([A-Z])",
            with: "$1\n**$2",
            options: .regularExpression
        )

        // Fix missing spaces after punctuation: "word.Next" → "word. Next"
        text = text.replacingOccurrences(
            of: "([.!?:])([A-Z])",
            with: "$1 $2",
            options: .regularExpression
        )

        // Fix missing space when lowercase joins uppercase: "medicationLet" → "medication Let"
        text = text.replacingOccurrences(
            of: "([a-z])([A-Z][a-z])",
            with: "$1 $2",
            options: .regularExpression
        )

        // Build Text by splitting on ** markers: even parts are normal, odd parts are bold
        let parts = text.components(separatedBy: "**")
        var result = Text("")
        for (index, part) in parts.enumerated() {
            if part.isEmpty { continue }
            if index % 2 == 1 {
                result = result + Text(part).bold()
            } else {
                result = result + Text(part)
            }
        }
        return result
    }

    private var bubbleColor: Color {
        message.role == .user ? Theme.accentBlue : Color(.systemGray6)
    }
}

// MARK: - Tool Badge with long-press detail

struct ToolBadgeView: View {
    let call: ToolCallInfo
    @State private var showDetail = false

    private var displayName: String { Theme.toolDisplayName(for: call.toolName) }
    private var color: Color { Theme.toolColor(for: call.toolName) }

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: Theme.toolIcon(for: call.toolName))
                .font(.caption2)
            Text(displayName)
                .font(.caption2)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.12))
        .foregroundStyle(color)
        .clipShape(Capsule())
        .onLongPressGesture {
            showDetail = true
        }
        .sheet(isPresented: $showDetail) {
            ToolCallDetailView(toolCall: call, displayName: displayName)
        }
    }
}

#Preview("User Message") {
    MessageBubbleView(message: ChatMessage(
        role: .user, content: "Prepare me for my meeting with Dr. Muller tomorrow",
        timestamp: Date(), toolCalls: nil, isStreaming: false
    ))
    .padding()
}

#Preview("Assistant + Tool Badges") {
    MessageBubbleView(message: ChatMessage(
        role: .assistant,
        content: "Tomorrow at 2:00 PM you have an appointment with Dr. Muller at Klinikum Stuttgart.",
        timestamp: Date(),
        toolCalls: [
            ToolCallInfo(toolName: "search_calendar", status: .completed, durationMs: 45, resultSummary: nil, arguments: nil, resultData: nil),
            ToolCallInfo(toolName: "query_health_data", status: .completed, durationMs: 32, resultSummary: nil, arguments: nil, resultData: nil),
            ToolCallInfo(toolName: "search_reminders", status: .completed, durationMs: 18, resultSummary: nil, arguments: nil, resultData: nil),
        ],
        isStreaming: false
    ))
    .padding()
}
