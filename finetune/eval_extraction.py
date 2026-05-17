"""
Extraction quality eval — measure structured data extraction accuracy.

Tests whether the 4B model correctly extracts financial metrics from
document text into structured JSON. Each test case provides document text
and expected field values. Accuracy is measured per-field: exact match
for strings/integers, ±10% tolerance for floats.

Usage:
    python -m finetune.eval_extraction
    python -m finetune.eval_extraction --save results/extraction_eval.json
    python -m finetune.eval_extraction --verbose
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.engine.inference.client import SmallLanguageModelClient
from src.engine.knowledge.data_extractor import DataExtractor, parse_extraction_json


# ---------------------------------------------------------------------------
# Test set — expected extractions from known document passages
# ---------------------------------------------------------------------------

TEST_SET = [
    {
        "name": "snowflake_ceo_letter",
        "source": "snowflake-fy2025-annual-report.pdf",
        "text": (
            "Dear fellow Stockholders, Snowflake's fiscal year ended January 31, 2025 "
            "marked another strong period for Snowflake. This led to Snowflake delivering "
            "$3.5 billion of product revenue, representing 30% growth year-over-year."
        ),
        "expected": {
            "company": "Snowflake",
            "fiscal_year": 2025,
            "product_revenue": 3500000000,
            "revenue_growth_pct": 30.0,
        },
    },
    {
        "name": "snowflake_financial_highlights",
        "source": "snowflake-fy2025-annual-report.pdf",
        "text": (
            "Fiscal Year 2025 Business Highlights\n"
            "PRODUCT REVENUE $3,462,422 thousands\n"
            "Representing a year-over-year increase of 30%.\n"
            "NET REVENUE RETENTION RATE 126%\n"
            "Our net revenue retention rate reached 126%.\n"
            "$1M CUSTOMERS 580\n"
            "580 customers with trailing 12-month product revenue greater than $1 million.\n"
            "TOTAL REVENUE $3,626,396 thousands\n"
            "FREE CASH FLOW $884,100 thousands"
        ),
        "expected": {
            "company": "Snowflake",
            "fiscal_year": 2025,
            "revenue": 3626396000,
            "product_revenue": 3462422000,
            "revenue_growth_pct": 30.0,
            "nrr": 126,
            "customers_1m_plus": 580,
            "free_cash_flow": 884100000,
        },
    },
    {
        "name": "nextera_quarterly",
        "source": "nextera_quarterly_report.pdf",
        "text": (
            "Nextera Platform — Quarterly Report FY2024\n"
            "Total 2024 revenue reached EUR311,500, representing strong growth.\n"
            "Q4 2024 was the highest-revenue quarter at EUR103,200 with 11 new customers.\n"
            "Churn declined to 0.7%. ARR growth 21.6%.\n"
            "Total customers: 10. Enterprise tier: 4 customers."
        ),
        "expected": {
            "company": "Nextera",
            "fiscal_year": 2024,
            "revenue": 311500,
            "total_customers": 10,
        },
    },
    {
        "name": "synthetic_startup",
        "source": "synthetic.pdf",
        "text": (
            "TechVenture Inc. Annual Report FY2024\n"
            "Total revenue: $42.5 million, up 85% year-over-year.\n"
            "Net revenue retention: 135%.\n"
            "252 total customers, 12 with ARR above $1M.\n"
            "Product revenue: $38.2 million.\n"
            "Gross margin: 78%.\n"
            "Free cash flow: negative $5.2 million (investing in growth)."
        ),
        "expected": {
            "company": "TechVenture",
            "fiscal_year": 2024,
            "revenue": 42500000,
            "revenue_growth_pct": 85.0,
            "nrr": 135,
            "customers_1m_plus": 12,
            "total_customers": 252,
            "product_revenue": 38200000,
            "gross_margin_pct": 78.0,
            "free_cash_flow": -5200000,
        },
    },
    {
        "name": "minimal_info",
        "source": "minimal.pdf",
        "text": (
            "GlobalTech Corp reported fiscal year 2023 results. "
            "Revenue was approximately $8 billion."
        ),
        "expected": {
            "company": "GlobalTech",
            "fiscal_year": 2023,
            "revenue": 8000000000,
        },
    },
]

# Fields that should be null when not in the expected dict
ALL_NUMERIC_FIELDS = [
    "revenue", "revenue_growth_pct", "nrr", "customers_1m_plus",
    "total_customers", "product_revenue", "gross_margin_pct", "free_cash_flow",
]


def _check_field(expected_val, actual_val, field_name: str) -> bool:
    """Check if an extracted value matches the expected value."""
    if expected_val is None:
        return True  # don't penalize for extracting extra fields

    if actual_val is None:
        return False

    # String: case-insensitive contains
    if isinstance(expected_val, str):
        return expected_val.lower() in str(actual_val).lower()

    # Integer: exact match
    if isinstance(expected_val, int) and isinstance(actual_val, int):
        return expected_val == actual_val

    # Float/number: ±10% tolerance
    if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
        if expected_val == 0:
            return actual_val == 0
        return abs(actual_val - expected_val) / abs(expected_val) <= 0.10

    return str(expected_val) == str(actual_val)


async def run_eval(client: SmallLanguageModelClient, verbose: bool = False) -> dict:
    """Run the extraction eval and return results."""
    extractor = DataExtractor(client=client, db_path=":memory:")

    results = []
    total_fields = 0
    correct_fields = 0
    total_time_ms = 0

    for case in TEST_SET:
        t0 = time.perf_counter()
        result = await extractor.extract(case["text"], case["source"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_time_ms += elapsed_ms

        extracted = result.extracted or {}
        expected = case["expected"]

        field_results = {}
        for field, exp_val in expected.items():
            act_val = extracted.get(field)
            correct = _check_field(exp_val, act_val, field)
            field_results[field] = {
                "expected": exp_val,
                "actual": act_val,
                "correct": correct,
            }
            total_fields += 1
            if correct:
                correct_fields += 1

        case_correct = all(fr["correct"] for fr in field_results.values())

        if verbose:
            tag = "PASS" if case_correct else "FAIL"
            print(f"  [{tag}] {case['name']} ({elapsed_ms:.0f}ms)")
            for field, fr in field_results.items():
                mark = "OK" if fr["correct"] else "MISS"
                print(f"    [{mark}] {field}: expected={fr['expected']} got={fr['actual']}")

        results.append({
            "name": case["name"],
            "source": case["source"],
            "success": result.success,
            "case_correct": case_correct,
            "fields": field_results,
            "latency_ms": round(elapsed_ms, 1),
            "raw_output": result.raw_output,
        })

    accuracy = correct_fields / total_fields * 100 if total_fields > 0 else 0
    case_accuracy = sum(1 for r in results if r["case_correct"]) / len(results) * 100

    return {
        "field_accuracy": round(accuracy, 1),
        "case_accuracy": round(case_accuracy, 1),
        "total_fields": total_fields,
        "correct_fields": correct_fields,
        "total_cases": len(results),
        "correct_cases": sum(1 for r in results if r["case_correct"]),
        "mean_latency_ms": round(total_time_ms / len(results), 0),
        "results": results,
    }


async def main():
    parser = argparse.ArgumentParser(description="Extraction quality eval")
    parser.add_argument("--save", type=str, default="", help="Save results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-field results")
    args = parser.parse_args()

    client = SmallLanguageModelClient.create_with_auto_detection()

    print("Running extraction eval...")
    results = await run_eval(client, verbose=args.verbose)

    print(f"\n{'='*60}")
    print(f"  Extraction Eval Results")
    print(f"{'='*60}")
    print(f"  Field accuracy:  {results['field_accuracy']}% ({results['correct_fields']}/{results['total_fields']})")
    print(f"  Case accuracy:   {results['case_accuracy']}% ({results['correct_cases']}/{results['total_cases']})")
    print(f"  Mean latency:    {results['mean_latency_ms']}ms")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
