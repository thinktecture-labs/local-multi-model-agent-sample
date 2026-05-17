import SwiftUI

struct ChatView: View {
    @Environment(ChatViewModel.self) private var viewModel
    @State private var showSuggestions = true

    var body: some View {
        @Bindable var vm = viewModel
        ZStack {
            VStack(spacing: 0) {
                headerBar

                ScrollViewReader { proxy in
                    ScrollView {
                        if viewModel.messages.isEmpty && !viewModel.isLoadingModel {
                            emptyState
                                .frame(maxWidth: .infinity)
                                .padding(.top, 80)
                        }

                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.messages) { message in
                                MessageBubbleView(message: message)
                            }

                            if !viewModel.activeToolCalls.isEmpty {
                                VStack(spacing: 6) {
                                    ForEach(viewModel.activeToolCalls) { call in
                                        ToolCallIndicatorView(toolCall: call)
                                    }
                                }
                            }

                            if viewModel.isGenerating {
                                TypingIndicatorView()
                            }

                            // Invisible anchor for auto-scroll
                            Color.clear.frame(height: 1).id("bottom")
                        }
                        .padding()
                    }
                    .defaultScrollAnchor(.bottom)
                    .onChange(of: viewModel.messages.count) {
                        scrollToBottom(proxy)
                    }
                    .onChange(of: viewModel.messages.last?.content) {
                        scrollToBottom(proxy)
                    }
                    .onChange(of: viewModel.activeToolCalls.count) {
                        scrollToBottom(proxy)
                    }
                    .onChange(of: viewModel.isGenerating) {
                        scrollToBottom(proxy)
                    }
                }

                if viewModel.modelReady {
                    suggestionChips
                }

                inputBar
            }

            if viewModel.showModelDownload {
                ModelDownloadView(
                    progress: viewModel.downloadProgress,
                    isLoading: viewModel.isLoadingModel
                ) {
                    Task { await viewModel.downloadModel() }
                }
            } else if viewModel.isLoadingModel && !viewModel.modelReady {
                modelLoadingOverlay
            }
        }
    }

    private var modelLoadingOverlay: some View {
        ZStack {
            Color.black.opacity(0.4).ignoresSafeArea()

            VStack(spacing: 20) {
                AppIconView(size: 80, showBackground: false)
                    .frame(width: 80, height: 80)

                Text("Loading Model...")
                    .font(.headline)

                ProgressView()
                    .controlSize(.large)
                    .tint(Theme.accentBlue)

                Text("Preparing LFM 2.5 for on-device inference")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(32)
            .background(.ultraThickMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .padding(40)
        }
        .transition(.opacity)
    }

    private var emptyState: some View {
        VStack(spacing: 20) {
            AppIconView(size: 72, showBackground: false)
                .frame(width: 72, height: 72)
                .opacity(0.7)

            Text("LocalLife")
                .font(.title2.bold())
                .foregroundStyle(.primary)

            Text("Your on-device AI assistant for\ncalendar, health & reminders.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            HStack(spacing: 16) {
                toolBadge(icon: "calendar", label: "Calendar", color: Theme.calendar)
                toolBadge(icon: "heart.fill", label: "Health", color: Theme.healthKit)
                toolBadge(icon: "checklist", label: "Reminders", color: Theme.reminders)
            }
            .padding(.top, 8)

            Text("Everything runs locally — no data leaves your device.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .padding(.top, 4)
        }
    }

    private func toolBadge(icon: String, label: String, color: Color) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(color)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(width: 70)
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.2)) {
            proxy.scrollTo("bottom", anchor: .bottom)
        }
    }

    private var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("LocalLife")
                    .font(.headline)
                Text("On-device AI Agent")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if !viewModel.messages.isEmpty && !viewModel.isGenerating {
                Button {
                    viewModel.clearChat()
                } label: {
                    Image(systemName: "eraser")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            if viewModel.modelReady {
                HStack(spacing: 6) {
                    Circle()
                        .fill(.green)
                        .frame(width: 6, height: 6)
                    Text("LFM 2.5")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            AirplaneModeIndicatorView(isOffline: viewModel.isOffline)
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
        .onTapGesture(count: 3) {
            viewModel.toggleDemoMode()
        }
    }

    private let suggestions = [
        "What appointments do I have this week?",
        "Show my heart rate trends for the last 30 days",
        "Search my reminders for health questions",
        "Prepare me for my meeting with Dr. Pepper tomorrow",
    ]

    private var suggestionChips: some View {
        VStack(spacing: 0) {
            Divider()

            // Collapsible header
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    showSuggestions.toggle()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "lightbulb.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                    Text("Quick prompts")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: showSuggestions ? "chevron.down" : "chevron.up")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }

            if showSuggestions {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(suggestions, id: \.self) { suggestion in
                            Button {
                                viewModel.inputText = suggestion
                                Task { await viewModel.sendMessage() }
                            } label: {
                                Text(suggestion)
                                    .font(.caption)
                                    .lineLimit(2)
                                    .multilineTextAlignment(.leading)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(Theme.accentBlue.opacity(0.08))
                                    .foregroundStyle(Theme.accentBlue)
                                    .clipShape(Capsule())
                            }
                            .disabled(viewModel.isGenerating)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .background(.ultraThinMaterial)
    }

    private var inputBar: some View {
        @Bindable var vm = viewModel
        return HStack(spacing: 10) {
            // Mic button
            Button {
                viewModel.toggleVoiceInput()
            } label: {
                Image(systemName: viewModel.isListening ? "mic.fill" : "mic")
                    .font(.title3)
                    .foregroundStyle(viewModel.isListening ? .red : Theme.accentBlue)
                    .symbolEffect(.pulse, isActive: viewModel.isListening)
            }
            .disabled(viewModel.isGenerating || !viewModel.modelReady)

            // Text field — shows live transcript while listening
            TextField(
                viewModel.isListening ? "Listening..." : "Ask me anything...",
                text: viewModel.isListening
                    ? .constant(viewModel.speechRecognizer.transcript)
                    : $vm.inputText,
                axis: .vertical
            )
            .lineLimit(1...4)
            .textFieldStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(
                viewModel.isListening
                    ? Color.red.opacity(0.08)
                    : Color(.systemGray6)
            )
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(viewModel.isListening ? Color.red.opacity(0.3) : Color.clear, lineWidth: 1.5)
            )

            // Send button
            Button {
                Task { await viewModel.sendMessage() }
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundStyle(Theme.accentBlue)
            }
            .disabled(
                viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty
                    || viewModel.isGenerating
                    || !viewModel.modelReady
                    || viewModel.isListening
            )
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }
}

#Preview("Chat - Empty") {
    ChatView()
        .environment(ChatViewModel())
}

#Preview("Chat - With Messages") {
    {
        let vm = ChatViewModel()
        vm.modelReady = true
        vm.messages = [
            ChatMessage(role: .user, content: "Prepare me for my meeting with Dr. Muller tomorrow", timestamp: Date(), toolCalls: nil, isStreaming: false),
            ChatMessage(role: .assistant, content: "Tomorrow at 2:00 PM you have an appointment with Dr. Muller at Klinikum Stuttgart (Cardiology).\n\nYour resting heart rate averaged 72 bpm over the past month. Blood pressure stable at 138/85 mmHg.\n\nYou have 5 reminders in your \"Health Questions\" list.", timestamp: Date(), toolCalls: [
                ToolCallInfo(toolName: "search_calendar", status: .completed, durationMs: 45, resultSummary: "1 event found", arguments: nil, resultData: nil),
                ToolCallInfo(toolName: "query_health_data", status: .completed, durationMs: 32, resultSummary: "heart_rate: avg 72 bpm", arguments: nil, resultData: nil),
                ToolCallInfo(toolName: "search_reminders", status: .completed, durationMs: 18, resultSummary: "5 reminders", arguments: nil, resultData: nil),
            ], isStreaming: false),
        ]
        return ChatView()
            .environment(vm)
    }()
}
