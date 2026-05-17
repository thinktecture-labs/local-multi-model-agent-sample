import Foundation
import HealthKit

public final class HealthKitTool: Tool {
    public let name = "query_health_data"
    public let description = "Query the user's health data from HealthKit. Can retrieve heart rate, blood pressure, steps, active energy, sleep, and weight for a specified number of past days."

    private let healthStore = HKHealthStore()

    /// When true, returns curated demo data instead of real HealthKit queries.
    /// Toggle via triple-tap on header in the UI.
    public var demoMode: Bool = true

    public init() {}

    private let metricMapping: [String: HKQuantityTypeIdentifier] = [
        "heart_rate": .heartRate,
        "blood_pressure": .bloodPressureSystolic,
        "steps": .stepCount,
        "active_energy": .activeEnergyBurned,
        "weight": .bodyMass,
    ]

    public var parameters: [ToolParameterSchema] {
        [
            ToolParameterSchema(name: "metric", type: .string, description: "The health metric to query (heart_rate, blood_pressure, steps, active_energy, sleep, weight)", isOptional: false),
            ToolParameterSchema(name: "days", type: .integer, description: "Number of past days to query (default 30)", isOptional: true),
        ]
    }

    /// Request read access to HealthKit types.
    func requestAuthorization() async throws {
        let typesToRead: Set<HKObjectType> = [
            HKQuantityType(.heartRate),
            HKQuantityType(.bloodPressureSystolic),
            HKQuantityType(.bloodPressureDiastolic),
            HKQuantityType(.stepCount),
            HKQuantityType(.activeEnergyBurned),
            HKQuantityType(.bodyMass),
            HKCategoryType(.sleepAnalysis),
        ]
        try await healthStore.requestAuthorization(toShare: [], read: typesToRead)
    }

    public func execute(arguments: [String: Any]) async -> ToolResult {
        let metric = arguments["metric"] as? String ?? "heart_rate"
        let days = arguments["days"] as? Int ?? 30

        if demoMode {
            return demoData(metric: metric, days: days)
        }

        return await realHealthKitQuery(metric: metric, days: days)
    }

    // MARK: - Demo Mode (curated fake data for stage)

    private func demoData(metric: String, days: Int) -> ToolResult {
        let data: [String: Any]
        let summary: String

        switch metric {
        case "heart_rate":
            data = [
                "metric": "heart_rate",
                "unit": "bpm",
                "days": days,
                "count": days * 4,
                "avg": 72,
                "min": 58,
                "max": 94,
                "trend": "slightly_increasing",
                "trend_detail": "Average increased from 65 bpm to 72 bpm over past 3 months",
                "notable": "Elevated readings Feb 15-18 (avg 82 bpm)",
                "recent_7day_avg": 74,
                "samples": generateDemoHeartRate(days: days),
            ]
            summary = "heart_rate: avg 72 bpm (trend: slightly increasing), \(days * 4) readings over \(days) days"

        case "blood_pressure":
            data = [
                "metric": "blood_pressure",
                "unit": "mmHg",
                "days": days,
                "count": days,
                "systolic_avg": 138,
                "diastolic_avg": 85,
                "systolic_range": "130-148",
                "diastolic_range": "78-92",
                "trend": "stable",
                "classification": "Stage 1 hypertension",
                "samples": [
                    ["date": daysAgo(1), "systolic": 136, "diastolic": 84],
                    ["date": daysAgo(3), "systolic": 140, "diastolic": 86],
                    ["date": daysAgo(7), "systolic": 138, "diastolic": 85],
                    ["date": daysAgo(14), "systolic": 142, "diastolic": 88],
                    ["date": daysAgo(21), "systolic": 135, "diastolic": 82],
                ],
            ]
            summary = "blood_pressure: avg 138/85 mmHg (stable, Stage 1 hypertension)"

        case "steps":
            data = [
                "metric": "steps",
                "unit": "steps/day",
                "days": days,
                "count": days,
                "avg": 6500,
                "min": 2100,
                "max": 12400,
                "trend": "slightly_declining",
                "trend_detail": "Down from 7200/day to 6500/day over past month",
                "total": 6500 * days,
                "days_above_8000": 8,
                "samples": [
                    ["date": daysAgo(1), "value": 5800],
                    ["date": daysAgo(2), "value": 7200],
                    ["date": daysAgo(3), "value": 6100],
                    ["date": daysAgo(7), "value": 8400],
                    ["date": daysAgo(14), "value": 6900],
                ],
            ]
            summary = "steps: avg 6,500/day (declining, target 8,000)"

        case "active_energy":
            data = [
                "metric": "active_energy",
                "unit": "kcal/day",
                "days": days,
                "count": days,
                "avg": 380,
                "min": 120,
                "max": 650,
                "trend": "stable",
            ]
            summary = "active_energy: avg 380 kcal/day"

        case "weight":
            data = [
                "metric": "weight",
                "unit": "kg",
                "days": days,
                "count": 8,
                "avg": 82.0,
                "min": 81.2,
                "max": 82.8,
                "trend": "stable",
                "latest": 82.1,
            ]
            summary = "weight: 82.1 kg (stable)"

        case "sleep":
            data = [
                "metric": "sleep",
                "unit": "hours/night",
                "days": days,
                "count": days,
                "avg": 6.8,
                "min": 4.5,
                "max": 8.2,
                "trend": "stable",
            ]
            summary = "sleep: avg 6.8 hours/night"

        default:
            return ToolResult(success: false, data: "{}", displaySummary: "Unknown metric: \(metric)", error: "Unsupported metric")
        }

        let jsonData = (try? JSONSerialization.data(withJSONObject: data)) ?? Data()
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
        return ToolResult(success: true, data: jsonString, displaySummary: summary, error: nil)
    }

    private func generateDemoHeartRate(days: Int) -> [[String: Any]] {
        // Generate a few representative samples
        let samples: [(Int, Int)] = [
            (1, 71), (2, 74), (3, 69), (5, 73), (7, 76),
            (10, 70), (14, 68), (15, 82), (16, 84), (17, 80), (18, 78),
            (21, 72), (25, 70), (28, 68), (30, 65),
        ]
        return samples.filter { $0.0 <= days }.map { (daysBack, hr) in
            ["date": daysAgo(daysBack), "value": hr]
        }
    }

    private func daysAgo(_ n: Int) -> String {
        let date = Calendar.current.date(byAdding: .day, value: -n, to: Date()) ?? Date()
        return ISO8601DateFormatter().string(from: date)
    }

    // MARK: - Real HealthKit Queries

    private func realHealthKitQuery(metric: String, days: Int) async -> ToolResult {
        let endDate = Date()
        guard let startDate = Calendar.current.date(byAdding: .day, value: -days, to: endDate) else {
            return ToolResult(success: false, data: "{}", displaySummary: "Invalid date range", error: "Could not compute start date")
        }
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)

        if metric == "sleep" {
            return await querySleep(predicate: predicate, days: days)
        }

        guard let identifier = metricMapping[metric] else {
            return ToolResult(success: false, data: "{}", displaySummary: "Unknown metric: \(metric)", error: "Unsupported metric")
        }

        return await queryQuantitySamples(identifier: identifier, metric: metric, predicate: predicate, days: days)
    }

    private func queryQuantitySamples(
        identifier: HKQuantityTypeIdentifier,
        metric: String,
        predicate: NSPredicate,
        days: Int
    ) async -> ToolResult {
        let quantityType = HKQuantityType(identifier)
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)

        do {
            let samples: [HKQuantitySample] = try await withCheckedThrowingContinuation { continuation in
                let query = HKSampleQuery(
                    sampleType: quantityType,
                    predicate: predicate,
                    limit: HKObjectQueryNoLimit,
                    sortDescriptors: [sortDescriptor]
                ) { _, results, error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume(returning: (results as? [HKQuantitySample]) ?? [])
                    }
                }
                healthStore.execute(query)
            }

            let values = samples.map { extractValue(sample: $0, metric: metric) }

            let result: [String: Any] = [
                "metric": metric,
                "days": days,
                "count": values.count,
                "avg": values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count),
                "min": values.min() ?? 0,
                "max": values.max() ?? 0,
                "trend": computeTrend(values: values),
                "samples": Array(samples.suffix(10)).map { sample in
                    [
                        "date": ISO8601DateFormatter().string(from: sample.startDate),
                        "value": extractValue(sample: sample, metric: metric),
                    ] as [String: Any]
                },
            ]

            let jsonData = try JSONSerialization.data(withJSONObject: result)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
            let avg = values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
            let summary = "\(metric): avg \(String(format: "%.1f", avg)), \(values.count) readings over \(days) days"

            return ToolResult(success: true, data: jsonString, displaySummary: summary, error: nil)
        } catch {
            return ToolResult(success: false, data: "{}", displaySummary: "HealthKit error", error: error.localizedDescription)
        }
    }

    private func querySleep(predicate: NSPredicate, days: Int) async -> ToolResult {
        // Simplified sleep query
        let result: [String: Any] = [
            "metric": "sleep",
            "days": days,
            "note": "Sleep analysis requires HKCategoryType query — implementation pending",
        ]
        let jsonData = (try? JSONSerialization.data(withJSONObject: result)) ?? Data()
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
        return ToolResult(success: true, data: jsonString, displaySummary: "sleep: data pending", error: nil)
    }

    private func extractValue(sample: HKQuantitySample, metric: String) -> Double {
        switch metric {
        case "heart_rate":
            return sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
        case "blood_pressure":
            return sample.quantity.doubleValue(for: .millimeterOfMercury())
        case "steps":
            return sample.quantity.doubleValue(for: .count())
        case "active_energy":
            return sample.quantity.doubleValue(for: .kilocalorie())
        case "weight":
            return sample.quantity.doubleValue(for: .gramUnit(with: .kilo))
        default:
            return 0
        }
    }

    private func computeTrend(values: [Double]) -> String {
        guard values.count >= 4 else { return "insufficient_data" }
        let midpoint = values.count / 2
        let firstHalf = Array(values.prefix(midpoint))
        let secondHalf = Array(values.suffix(midpoint))
        let firstAvg = firstHalf.reduce(0, +) / Double(firstHalf.count)
        let secondAvg = secondHalf.reduce(0, +) / Double(secondHalf.count)
        guard firstAvg != 0 else { return "stable" }
        let delta = ((secondAvg - firstAvg) / firstAvg) * 100
        if abs(delta) < 3 { return "stable" }
        return delta > 0 ? "increasing" : "decreasing"
    }
}
