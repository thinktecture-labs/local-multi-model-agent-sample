import Foundation
import EventKit

public final class CalendarTool: Tool {
    public let name = "search_calendar"
    public let description = "Search the user's calendar for upcoming appointments and events. Returns title, date, time, location, and notes for matching appointments."

    private let eventStore: EKEventStore

    public init(eventStore: EKEventStore = EKEventStore()) {
        self.eventStore = eventStore
    }

    /// Configurable: restrict to a specific calendar by name (e.g. "SDD Demo").
    /// When nil, searches all calendars.
    public var calendarFilter: String? = nil

    public var parameters: [ToolParameterSchema] {
        [
            ToolParameterSchema(name: "query", type: .string, description: "Optional search term to filter events by title, location, or notes", isOptional: true),
            ToolParameterSchema(name: "days_ahead", type: .integer, description: "Number of days ahead to search (default 7)", isOptional: true),
        ]
    }

    /// Request full access to Calendar. Call during app setup.
    func requestAccess() async throws -> Bool {
        try await eventStore.requestFullAccessToEvents()
    }

    public func execute(arguments: [String: Any]) async -> ToolResult {
        let query = arguments["query"] as? String
        let daysAhead = arguments["days_ahead"] as? Int ?? 7

        let startDate = Date()
        guard let endDate = Calendar.current.date(byAdding: .day, value: daysAhead, to: startDate) else {
            return ToolResult(success: false, data: "{}", displaySummary: "Invalid date range", error: "Could not compute end date")
        }

        // Apply calendar filter if configured
        var calendars: [EKCalendar]? = nil
        if let filter = calendarFilter {
            let matching = eventStore.calendars(for: .event).filter {
                $0.title.lowercased() == filter.lowercased()
            }
            if matching.isEmpty {
                return ToolResult(
                    success: false,
                    data: "{}",
                    displaySummary: "Calendar '\(filter)' not found",
                    error: "No calendar named '\(filter)'. Available: \(eventStore.calendars(for: .event).map { $0.title }.joined(separator: ", "))"
                )
            }
            calendars = matching
        }

        let predicate = eventStore.predicateForEvents(
            withStart: startDate, end: endDate, calendars: calendars
        )
        // EventKit's events(matching:) is synchronous — run off the caller's context.
        let store = eventStore
        var events = await Task.detached {
            store.events(matching: predicate)
        }.value

        // Filter by query if provided (strip punctuation so "dr pepper" matches "Dr. Pepper")
        if let query = query?.lowercased(), !query.isEmpty {
            let normalize = { (s: String) in s.lowercased().replacingOccurrences(of: ".", with: "") }
            let normalizedQuery = normalize(query)
            events = events.filter { event in
                (event.title.map { normalize($0).contains(normalizedQuery) } ?? false)
                    || (event.notes.map { normalize($0).contains(normalizedQuery) } ?? false)
                    || (event.location.map { normalize($0).contains(normalizedQuery) } ?? false)
            }
        }

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        let timeFormatter = DateFormatter()
        timeFormatter.dateFormat = "HH:mm"

        let eventDicts: [[String: Any]] = events.map { event in
            var dict: [String: Any] = [
                "title": event.title ?? "Untitled",
                "date": dateFormatter.string(from: event.startDate),
                "start_time": timeFormatter.string(from: event.startDate),
                "end_time": timeFormatter.string(from: event.endDate),
                "is_all_day": event.isAllDay,
            ]
            if let location = event.location, !location.isEmpty {
                dict["location"] = location
            }
            if let notes = event.notes, !notes.isEmpty {
                dict["notes"] = notes
            }
            return dict
        }

        let result: [String: Any] = [
            "events": eventDicts,
            "count": eventDicts.count,
            "search_query": query ?? "",
            "days_ahead": daysAhead,
            "calendar_filter": calendarFilter ?? "all",
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: result)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"

            let summary = events.isEmpty
                ? "No events found"
                : "\(events.count) event(s): \(events.map { $0.title ?? "?" }.joined(separator: ", "))"

            return ToolResult(success: true, data: jsonString, displaySummary: summary, error: nil)
        } catch {
            return ToolResult(success: false, data: "{}", displaySummary: "JSON error", error: error.localizedDescription)
        }
    }
}
