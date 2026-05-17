import Foundation
import EventKit

public final class RemindersTool: Tool {
    public let name = "search_reminders"
    public let description = "Search the user's Reminders for tasks and notes. Returns matching reminders with title, notes, completion status, and due date."

    private let eventStore: EKEventStore

    public init(eventStore: EKEventStore = EKEventStore()) {
        self.eventStore = eventStore
    }

    public var parameters: [ToolParameterSchema] {
        [
            ToolParameterSchema(name: "query", type: .string, description: "Search term to find relevant reminders by title or notes", isOptional: false),
            ToolParameterSchema(name: "list_name", type: .string, description: "Optional: restrict search to a specific Reminders list", isOptional: true),
        ]
    }

    /// Request full access to Reminders. Call during app setup.
    func requestAccess() async throws -> Bool {
        try await eventStore.requestFullAccessToReminders()
    }

    public func execute(arguments: [String: Any]) async -> ToolResult {
        let query = (arguments["query"] as? String)?.lowercased() ?? ""
        let listName = arguments["list_name"] as? String

        // Find the target calendar (Reminders list)
        var calendars: [EKCalendar]? = nil
        if let listName {
            let matching = eventStore.calendars(for: .reminder).filter {
                $0.title.lowercased().contains(listName.lowercased())
            }
            if !matching.isEmpty {
                calendars = matching
            }
        }

        let predicate = eventStore.predicateForReminders(in: calendars)

        do {
            let reminders = try await fetchReminders(matching: predicate)

            // Filter by query — match title, notes, OR list name so that
            // "health" finds all reminders in the "Health Questions" list.
            let matched: [EKReminder]
            if query.isEmpty {
                matched = reminders
            } else {
                let byQuery = reminders.filter { reminder in
                    (reminder.title?.lowercased().contains(query) ?? false)
                        || (reminder.notes?.lowercased().contains(query) ?? false)
                        || (reminder.calendar?.title.lowercased().contains(query) ?? false)
                }
                matched = byQuery.isEmpty ? reminders : byQuery
            }

            // Prefer incomplete reminders; show completed only if none are pending
            let incomplete = matched.filter { !$0.isCompleted }
            let filtered = incomplete.isEmpty ? matched : incomplete

            let reminderDicts: [[String: Any]] = filtered.map { reminder in
                var dict: [String: Any] = [
                    "title": reminder.title ?? "Untitled",
                    "is_completed": reminder.isCompleted,
                    "list": reminder.calendar?.title ?? "Unknown",
                ]
                if let notes = reminder.notes, !notes.isEmpty {
                    dict["notes"] = notes
                }
                if let due = reminder.dueDateComponents {
                    if let date = Calendar.current.date(from: due) {
                        dict["due_date"] = ISO8601DateFormatter().string(from: date)
                    }
                }
                return dict
            }

            let result: [String: Any] = [
                "reminders": reminderDicts,
                "count": reminderDicts.count,
                "query": query,
                "list_filter": listName ?? "all",
            ]

            let jsonData = try JSONSerialization.data(withJSONObject: result)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"

            let summary = filtered.isEmpty
                ? "No reminders found for '\(query)'"
                : "\(filtered.count) reminder(s): \(filtered.prefix(3).compactMap { $0.title }.joined(separator: ", "))"

            return ToolResult(success: true, data: jsonString, displaySummary: summary, error: nil)
        } catch {
            return ToolResult(
                success: false,
                data: "{}",
                displaySummary: "Reminders error: \(error.localizedDescription)",
                error: error.localizedDescription
            )
        }
    }

    /// Async wrapper for EKEventStore.fetchReminders (callback-based API).
    private func fetchReminders(matching predicate: NSPredicate) async throws -> [EKReminder] {
        try await withCheckedThrowingContinuation { continuation in
            eventStore.fetchReminders(matching: predicate) { reminders in
                continuation.resume(returning: reminders ?? [])
            }
        }
    }
}
