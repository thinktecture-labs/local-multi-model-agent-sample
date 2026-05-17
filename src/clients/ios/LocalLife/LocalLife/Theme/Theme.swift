import SwiftUI

enum Theme {
    // Thinktecture CI primary
    static let accentBlue = Color(red: 1/255, green: 112/255, blue: 185/255) // #0170B9

    // Tool badge colors
    static let healthKit = Color.red
    static let calendar = Color.blue
    static let reminders = Color.orange

    // Semantic
    static let success = Color.green
    static let error = Color.red

    /// Centralized display metadata for tool calls.
    static func toolDisplayName(for toolName: String) -> String {
        switch toolName {
        case "query_health_data": "HealthKit"
        case "search_calendar": "Calendar"
        case "search_reminders": "Reminders"
        default: toolName
        }
    }

    static func toolIcon(for toolName: String) -> String {
        switch toolName {
        case "query_health_data": "heart.fill"
        case "search_calendar": "calendar"
        case "search_reminders": "checklist"
        default: "wrench.fill"
        }
    }

    static func toolColor(for toolName: String) -> Color {
        switch toolName {
        case "query_health_data": healthKit
        case "search_calendar": calendar
        case "search_reminders": reminders
        default: accentBlue
        }
    }
}
