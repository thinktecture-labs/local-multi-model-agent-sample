import SwiftUI

/// Programmatic app icon: brain + heart motif on Thinktecture blue.
/// Used to generate the app icon PNG and as the launch screen logo.
struct AppIconView: View {
    var size: CGFloat = 1024
    var showBackground: Bool = true

    private var scale: CGFloat { size / 1024 }

    var body: some View {
        ZStack {
            if showBackground {
                RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0/255, green: 80/255, blue: 150/255),
                                Theme.accentBlue,
                                Color(red: 30/255, green: 140/255, blue: 210/255)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }

            brainHeartSymbol
                .frame(width: size * 0.65, height: size * 0.65)
        }
        .frame(width: size, height: size)
    }

    /// The brain + heart symbol, reusable for icon and splash screen.
    var brainHeartSymbol: some View {
        Canvas { context, canvasSize in
            let w = canvasSize.width
            let h = canvasSize.height
            let cx = w / 2
            let cy = h * 0.42 // shift brain up to make room for heart

            // --- Brain ---
            let brainW = w * 0.82
            let brainH = h * 0.58

            var brain = Path()

            // Left hemisphere — top lobe
            brain.addEllipse(in: CGRect(
                x: cx - brainW * 0.50, y: cy - brainH * 0.50,
                width: brainW * 0.48, height: brainH * 0.52
            ))
            // Left hemisphere — bottom lobe
            brain.addEllipse(in: CGRect(
                x: cx - brainW * 0.52, y: cy - brainH * 0.10,
                width: brainW * 0.46, height: brainH * 0.50
            ))

            // Right hemisphere — top lobe
            brain.addEllipse(in: CGRect(
                x: cx + brainW * 0.02, y: cy - brainH * 0.50,
                width: brainW * 0.48, height: brainH * 0.52
            ))
            // Right hemisphere — bottom lobe
            brain.addEllipse(in: CGRect(
                x: cx + brainW * 0.06, y: cy - brainH * 0.10,
                width: brainW * 0.46, height: brainH * 0.50
            ))

            // Center bridge
            brain.addEllipse(in: CGRect(
                x: cx - brainW * 0.10, y: cy - brainH * 0.38,
                width: brainW * 0.20, height: brainH * 0.72
            ))

            context.fill(brain, with: .color(.white))

            // --- Center fissure line ---
            var fissure = Path()
            fissure.move(to: CGPoint(x: cx, y: cy - brainH * 0.42))
            fissure.addLine(to: CGPoint(x: cx, y: cy + brainH * 0.28))
            context.stroke(
                fissure,
                with: .color(Theme.accentBlue.opacity(0.25)),
                lineWidth: w * 0.015
            )

            // --- Heart at bottom ---
            let heartCy = cy + brainH * 0.42
            let heartSize = w * 0.22
            let heartPath = makeHeart(cx: cx, cy: heartCy, size: heartSize)
            context.fill(heartPath, with: .color(Color(red: 1.0, green: 0.30, blue: 0.35)))

            // --- Pulse line on heart ---
            var pulse = Path()
            let py = heartCy + heartSize * 0.05
            let pw = heartSize * 0.7
            pulse.move(to: CGPoint(x: cx - pw, y: py))
            pulse.addLine(to: CGPoint(x: cx - pw * 0.45, y: py))
            pulse.addLine(to: CGPoint(x: cx - pw * 0.25, y: py - heartSize * 0.30))
            pulse.addLine(to: CGPoint(x: cx + pw * 0.05, y: py + heartSize * 0.22))
            pulse.addLine(to: CGPoint(x: cx + pw * 0.25, y: py - heartSize * 0.10))
            pulse.addLine(to: CGPoint(x: cx + pw * 0.45, y: py))
            pulse.addLine(to: CGPoint(x: cx + pw, y: py))
            context.stroke(pulse, with: .color(.white), style: StrokeStyle(lineWidth: w * 0.022, lineCap: .round, lineJoin: .round))
        }
    }

    private func makeHeart(cx: CGFloat, cy: CGFloat, size: CGFloat) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: cx, y: cy + size * 0.85))
        p.addCurve(
            to: CGPoint(x: cx - size, y: cy - size * 0.15),
            control1: CGPoint(x: cx - size * 0.45, y: cy + size * 0.65),
            control2: CGPoint(x: cx - size * 1.15, y: cy + size * 0.15)
        )
        p.addCurve(
            to: CGPoint(x: cx, y: cy - size * 0.15),
            control1: CGPoint(x: cx - size * 0.85, y: cy - size * 0.6),
            control2: CGPoint(x: cx - size * 0.1, y: cy - size * 0.35)
        )
        p.addCurve(
            to: CGPoint(x: cx + size, y: cy - size * 0.15),
            control1: CGPoint(x: cx + size * 0.1, y: cy - size * 0.35),
            control2: CGPoint(x: cx + size * 0.85, y: cy - size * 0.6)
        )
        p.addCurve(
            to: CGPoint(x: cx, y: cy + size * 0.85),
            control1: CGPoint(x: cx + size * 1.15, y: cy + size * 0.15),
            control2: CGPoint(x: cx + size * 0.45, y: cy + size * 0.65)
        )
        p.closeSubpath()
        return p
    }
}

#Preview("App Icon 1024") {
    AppIconView(size: 300)
}

#Preview("App Icon - Symbol Only") {
    AppIconView(size: 300, showBackground: false)
        .background(.white)
}
