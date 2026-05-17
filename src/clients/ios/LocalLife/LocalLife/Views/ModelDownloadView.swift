import SwiftUI

struct ModelDownloadView: View {
    var progress: Double? = nil
    var isLoading: Bool = false
    var onDownload: (() -> Void)? = nil

    var body: some View {
        ZStack {
            Color.black.opacity(0.6).ignoresSafeArea()

            VStack(spacing: 24) {
                Image(systemName: "brain")
                    .font(.system(size: 48))
                    .foregroundStyle(Theme.accentBlue)

                Text("LFM 2.5 1.2B Instruct")
                    .font(.title3.bold())

                Text("~700 MB download required for first use.\nAfter download, everything runs offline.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                if isLoading {
                    ProgressView()
                        .controlSize(.large)
                        .padding(.top, 8)
                } else if let progress {
                    VStack(spacing: 8) {
                        ProgressView(value: progress)
                            .tint(Theme.accentBlue)
                        Text("\(Int(progress * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 40)
                } else {
                    Button("Download Model") {
                        onDownload?()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accentBlue)
                }
            }
            .padding(32)
            .background(.ultraThickMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .padding(40)
        }
    }
}

#Preview("Download Button") {
    ModelDownloadView()
}

#Preview("Downloading 65%") {
    ModelDownloadView(progress: 0.65)
}

#Preview("Loading") {
    ModelDownloadView(isLoading: true)
}
