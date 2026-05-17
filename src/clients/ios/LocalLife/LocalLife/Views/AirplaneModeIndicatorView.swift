import SwiftUI

struct AirplaneModeIndicatorView: View {
    let isOffline: Bool

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: isOffline ? "airplane" : "wifi")
                .font(.caption)
            Text(isOffline ? "Offline" : "Online")
                .font(.caption2)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isOffline ? Color.green.opacity(0.15) : Color.secondary.opacity(0.1))
        .foregroundStyle(isOffline ? .green : .secondary)
        .clipShape(Capsule())
    }
}

#Preview("Online") {
    AirplaneModeIndicatorView(isOffline: false)
        .padding()
}

#Preview("Offline") {
    AirplaneModeIndicatorView(isOffline: true)
        .padding()
}
