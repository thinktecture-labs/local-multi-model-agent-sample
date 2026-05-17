import Network
import Foundation

final class ConnectivityMonitor: @unchecked Sendable {
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "com.thinktecture.locallife.connectivity")

    deinit {
        monitor.cancel()
    }

    func start(onChange: @escaping @Sendable (Bool) -> Void) {
        monitor.pathUpdateHandler = { path in
            DispatchQueue.main.async {
                onChange(path.status == .satisfied)
            }
        }
        monitor.start(queue: queue)
    }
}
