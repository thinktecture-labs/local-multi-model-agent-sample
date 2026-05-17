import SwiftUI

struct ToolCallIndicatorView: View {
    let toolCall: ToolCallInfo
    @State private var showDetail = false

    var body: some View {
        HStack(spacing: 8) {
            statusIcon
            VStack(alignment: .leading, spacing: 2) {
                Text(displayName)
                    .font(.caption.bold())
                if let summary = toolCall.resultSummary {
                    Text(summary)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            Spacer()
            if let duration = toolCall.durationMs {
                Text(String(format: "%.0fms", duration))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(Color(.systemGray6).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .onLongPressGesture {
            showDetail = true
        }
        .sheet(isPresented: $showDetail) {
            ToolCallDetailView(toolCall: toolCall, displayName: displayName)
        }
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch toolCall.status {
        case .pending:
            Image(systemName: "clock")
                .foregroundStyle(.secondary)
        case .executing:
            ProgressView()
                .controlSize(.small)
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
        }
    }

    private var displayName: String {
        Theme.toolDisplayName(for: toolCall.toolName)
    }
}

// MARK: - Detail Sheet

struct ToolCallDetailView: View {
    let toolCall: ToolCallInfo
    let displayName: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    HStack(spacing: 12) {
                        toolIcon
                            .font(.title2)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(displayName)
                                .font(.headline)
                            Text(toolCall.toolName)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        statusBadge
                    }
                    .padding()
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                    // Arguments
                    if let args = toolCall.arguments, !args.isEmpty {
                        detailSection(title: "Arguments", systemImage: "arrow.right.circle") {
                            ForEach(args.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                                HStack(alignment: .top) {
                                    Text(key)
                                        .font(.caption.monospaced().bold())
                                        .foregroundStyle(Theme.accentBlue)
                                    Spacer()
                                    Text(value)
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.primary)
                                        .multilineTextAlignment(.trailing)
                                }
                            }
                        }
                    }

                    // Result
                    if let resultData = toolCall.resultData {
                        detailSection(title: "Response", systemImage: "arrow.left.circle") {
                            Text(prettyJSON(resultData))
                                .font(.caption2.monospaced())
                                .foregroundStyle(.primary)
                                .textSelection(.enabled)
                        }
                    }

                    // Timing
                    if let ms = toolCall.durationMs {
                        detailSection(title: "Performance", systemImage: "gauge.medium") {
                            HStack {
                                Text("Execution time")
                                    .font(.caption)
                                Spacer()
                                Text(String(format: "%.1f ms", ms))
                                    .font(.caption.monospacedDigit().bold())
                                    .foregroundStyle(ms < 100 ? Theme.success : .orange)
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Tool Call Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    @ViewBuilder
    private func detailSection(title: String, systemImage: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 6) {
                content()
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.systemGray6))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var toolIcon: some View {
        Image(systemName: Theme.toolIcon(for: toolCall.toolName))
            .foregroundStyle(Theme.toolColor(for: toolCall.toolName))
    }

    private var statusBadge: some View {
        HStack(spacing: 4) {
            switch toolCall.status {
            case .completed:
                Image(systemName: "checkmark.circle.fill")
                Text("Success")
            case .failed(let msg):
                Image(systemName: "xmark.circle.fill")
                Text(msg.prefix(20))
            case .executing:
                ProgressView().controlSize(.mini)
                Text("Running")
            case .pending:
                Image(systemName: "clock")
                Text("Pending")
            }
        }
        .font(.caption2.bold())
        .foregroundStyle(statusColor)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(statusColor.opacity(0.1))
        .clipShape(Capsule())
    }

    private var statusColor: Color {
        switch toolCall.status {
        case .completed: .green
        case .failed: .red
        case .executing: .blue
        case .pending: .secondary
        }
    }

    private func prettyJSON(_ raw: String) -> String {
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]),
              let str = String(data: pretty, encoding: .utf8)
        else { return raw }
        return str
    }
}

#Preview("Completed") {
    ToolCallIndicatorView(toolCall: ToolCallInfo(
        toolName: "search_calendar", status: .completed,
        durationMs: 45, resultSummary: "1 event: Dr. Muller - Cardiology, tomorrow 14:00",
        arguments: ["query": "muller", "days_ahead": "7"],
        resultData: "{\"count\":1,\"events\":[{\"title\":\"Dr. Muller - Cardiology\",\"start_time\":\"14:00\",\"location\":\"Klinikum Stuttgart\"}]}"
    ))
    .padding()
}

#Preview("Executing") {
    ToolCallIndicatorView(toolCall: ToolCallInfo(
        toolName: "query_health_data", status: .executing,
        durationMs: nil, resultSummary: nil,
        arguments: nil, resultData: nil
    ))
    .padding()
}

#Preview("Failed") {
    ToolCallIndicatorView(toolCall: ToolCallInfo(
        toolName: "search_reminders", status: .failed("Permission denied"),
        durationMs: 12, resultSummary: "Permission denied",
        arguments: ["query": "health"], resultData: nil
    ))
    .padding()
}
