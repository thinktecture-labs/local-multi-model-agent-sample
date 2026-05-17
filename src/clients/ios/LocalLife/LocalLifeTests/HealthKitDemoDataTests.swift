// HealthKitDemoDataTests.swift
// Tests for HealthKitTool demo mode data — validates that curated demo data
// returns correct structure, units, and values for all supported metrics.

import Testing
import Foundation

// MARK: - Standalone demo data logic (mirrors HealthKitTool.demoData)

/// Reproduces the demo data generation logic from HealthKitTool for testing
/// without HealthKit framework dependency.
private enum DemoHealthData {

    static func query(metric: String, days: Int) -> (success: Bool, data: [String: Any], summary: String) {
        switch metric {
        case "heart_rate":
            let data: [String: Any] = [
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
            ]
            return (true, data, "heart_rate: avg 72 bpm (trend: slightly increasing), \(days * 4) readings over \(days) days")

        case "blood_pressure":
            let data: [String: Any] = [
                "metric": "blood_pressure",
                "unit": "mmHg",
                "days": days,
                "count": days,
                "systolic_avg": 138,
                "diastolic_avg": 85,
                "trend": "stable",
                "classification": "Stage 1 hypertension",
            ]
            return (true, data, "blood_pressure: avg 138/85 mmHg (stable, Stage 1 hypertension)")

        case "steps":
            let data: [String: Any] = [
                "metric": "steps",
                "unit": "steps/day",
                "days": days,
                "avg": 6500,
                "min": 2100,
                "max": 12400,
                "trend": "slightly_declining",
            ]
            return (true, data, "steps: avg 6,500/day (declining, target 8,000)")

        case "active_energy":
            let data: [String: Any] = [
                "metric": "active_energy",
                "unit": "kcal/day",
                "days": days,
                "avg": 380,
                "trend": "stable",
            ]
            return (true, data, "active_energy: avg 380 kcal/day")

        case "weight":
            let data: [String: Any] = [
                "metric": "weight",
                "unit": "kg",
                "days": days,
                "avg": 82.0,
                "latest": 82.1,
                "trend": "stable",
            ]
            return (true, data, "weight: 82.1 kg (stable)")

        case "sleep":
            let data: [String: Any] = [
                "metric": "sleep",
                "unit": "hours/night",
                "days": days,
                "avg": 6.8,
                "trend": "stable",
            ]
            return (true, data, "sleep: avg 6.8 hours/night")

        default:
            return (false, [:], "Unknown metric: \(metric)")
        }
    }
}

// MARK: - Tests

@Suite("HealthKit Demo Data")
struct HealthKitDemoDataTests {

    // MARK: Heart Rate

    @Test("Heart rate demo data has correct structure and values")
    func heartRateData() {
        let (success, data, summary) = DemoHealthData.query(metric: "heart_rate", days: 30)

        #expect(success)
        #expect(data["metric"] as? String == "heart_rate")
        #expect(data["unit"] as? String == "bpm")
        #expect(data["avg"] as? Int == 72)
        #expect(data["min"] as? Int == 58)
        #expect(data["max"] as? Int == 94)
        #expect(data["trend"] as? String == "slightly_increasing")
        #expect(data["count"] as? Int == 120) // 30 * 4
        #expect(summary.contains("72 bpm"))
    }

    @Test("Heart rate count scales with days parameter")
    func heartRateCountScales() {
        let (_, data7, _) = DemoHealthData.query(metric: "heart_rate", days: 7)
        let (_, data90, _) = DemoHealthData.query(metric: "heart_rate", days: 90)

        #expect(data7["count"] as? Int == 28)   // 7 * 4
        #expect(data90["count"] as? Int == 360) // 90 * 4
    }

    // MARK: Blood Pressure

    @Test("Blood pressure demo data includes systolic and diastolic")
    func bloodPressureData() {
        let (success, data, summary) = DemoHealthData.query(metric: "blood_pressure", days: 30)

        #expect(success)
        #expect(data["unit"] as? String == "mmHg")
        #expect(data["systolic_avg"] as? Int == 138)
        #expect(data["diastolic_avg"] as? Int == 85)
        #expect(data["classification"] as? String == "Stage 1 hypertension")
        #expect(summary.contains("138/85"))
    }

    // MARK: Steps

    @Test("Steps demo data has correct averages")
    func stepsData() {
        let (success, data, summary) = DemoHealthData.query(metric: "steps", days: 30)

        #expect(success)
        #expect(data["unit"] as? String == "steps/day")
        #expect(data["avg"] as? Int == 6500)
        #expect(data["trend"] as? String == "slightly_declining")
        #expect(summary.contains("6,500"))
    }

    // MARK: Active Energy

    @Test("Active energy demo data returns kcal")
    func activeEnergyData() {
        let (success, data, _) = DemoHealthData.query(metric: "active_energy", days: 30)

        #expect(success)
        #expect(data["unit"] as? String == "kcal/day")
        #expect(data["avg"] as? Int == 380)
    }

    // MARK: Weight

    @Test("Weight demo data returns kg")
    func weightData() {
        let (success, data, summary) = DemoHealthData.query(metric: "weight", days: 30)

        #expect(success)
        #expect(data["unit"] as? String == "kg")
        #expect(data["latest"] as? Double == 82.1)
        #expect(summary.contains("82.1 kg"))
    }

    // MARK: Sleep

    @Test("Sleep demo data returns hours")
    func sleepData() {
        let (success, data, summary) = DemoHealthData.query(metric: "sleep", days: 30)

        #expect(success)
        #expect(data["unit"] as? String == "hours/night")
        #expect(data["avg"] as? Double == 6.8)
        #expect(summary.contains("6.8"))
    }

    // MARK: Unknown Metric

    @Test("Unknown metric returns failure")
    func unknownMetric() {
        let (success, _, summary) = DemoHealthData.query(metric: "body_temperature", days: 30)

        #expect(!success)
        #expect(summary.contains("Unknown metric"))
    }

    // MARK: JSON Serialization

    @Test("Demo data can be serialized to valid JSON")
    func jsonSerialization() throws {
        let metrics = ["heart_rate", "blood_pressure", "steps", "active_energy", "weight", "sleep"]

        for metric in metrics {
            let (success, data, _) = DemoHealthData.query(metric: metric, days: 30)
            #expect(success, "Metric \(metric) should succeed")

            let jsonData = try JSONSerialization.data(withJSONObject: data)
            let jsonString = String(data: jsonData, encoding: .utf8)
            #expect(jsonString != nil, "Metric \(metric) should produce valid JSON")
            #expect(!jsonString!.isEmpty, "Metric \(metric) JSON should not be empty")
        }
    }

    // MARK: Summary Format (for SLM consumption)

    @Test("Summaries contain units for SLM context")
    func summariesContainUnits() {
        let expectations: [(String, String)] = [
            ("heart_rate", "bpm"),
            ("blood_pressure", "mmHg"),
            ("steps", "/day"),
            ("weight", "kg"),
            ("sleep", "hours"),
            ("active_energy", "kcal"),
        ]

        for (metric, expectedUnit) in expectations {
            let (_, _, summary) = DemoHealthData.query(metric: metric, days: 30)
            #expect(summary.contains(expectedUnit), "Summary for \(metric) should contain '\(expectedUnit)': got '\(summary)'")
        }
    }
}
