#!/usr/bin/env python3
"""
Generate high-quality multi-turn tool-calling training data.

Key design decisions:
1. Turn 2 queries MUST match inference-time concretized phrasing
   (both vague "Calculate the growth" AND explicit "Calculate 103200 * 1.15")
2. Heavy calculator signal words: "calculate", "compute", "multiply", "divide",
   "what is X + Y", "add", "subtract", "percentage", "how much"
3. Volume: 300 examples to match 1266 single-turn ratio (~20%)
4. All sql_query -> calculator (matching the 2-tool architecture)
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.engine.inference.config import SCENARIO_CONFIG

random.seed(42)

# ---------------------------------------------------------------------------
# Templates for SQL Turn 1 queries
# ---------------------------------------------------------------------------
SQL_TEMPLATES = [
    # Revenue queries
    ("What was revenue in {quarter} {year}?",
     "SELECT revenue FROM sales WHERE year = {year} AND quarter = '{quarter}'",
     lambda q, y: {"revenue": random.choice([24700, 31200, 42800, 55100, 68300, 84900, 103200])}),
    ("What was total revenue in {year}?",
     "SELECT SUM(revenue) as total FROM sales WHERE year = {year}",
     lambda q, y: {"total": random.choice([98700, 311500, 410200])}),
    ("Show the highest quarterly revenue in {year}",
     "SELECT MAX(revenue) as max_rev FROM sales WHERE year = {year}",
     lambda q, y: {"max_rev": random.choice([42800, 103200])}),
    ("What was the lowest quarterly revenue in {year}?",
     "SELECT MIN(revenue) as min_rev FROM sales WHERE year = {year}",
     lambda q, y: {"min_rev": random.choice([24700, 55100])}),
    ("Find average revenue per quarter in {year}",
     "SELECT AVG(revenue) as avg_rev FROM sales WHERE year = {year}",
     lambda q, y: {"avg_rev": random.choice([24675, 77875])}),

    # Customer queries
    ("How many customers are on the {tier} tier?",
     "SELECT COUNT(*) as count FROM customers WHERE tier = '{tier_lower}'",
     lambda q, y: {"count": random.randint(1, 6)}),
    ("Show total customers",
     "SELECT COUNT(*) as total FROM customers",
     lambda q, y: {"total": 10}),
    ("How many new customers joined in {quarter} {year}?",
     "SELECT new_customers FROM sales WHERE year = {year} AND quarter = '{quarter}'",
     lambda q, y: {"new_customers": random.choice([4, 7, 8, 9, 11])}),
    ("Find total new customers in {year}",
     "SELECT SUM(new_customers) as total FROM sales WHERE year = {year}",
     lambda q, y: {"total": random.choice([15, 35])}),
    ("Show the top customer by MRR",
     "SELECT name, mrr FROM customers ORDER BY mrr DESC LIMIT 1",
     lambda q, y: {"name": "BrightHealth GmbH", "mrr": 7000}),
    ("Find the lowest MRR customer",
     "SELECT name, mrr FROM customers ORDER BY mrr ASC LIMIT 1",
     lambda q, y: {"name": "EduTech Berlin", "mrr": 299}),
    ("What is the total MRR across all customers?",
     "SELECT SUM(mrr) as total_mrr FROM customers",
     lambda q, y: {"total_mrr": 24995}),
    ("What is the average MRR across all customers?",
     "SELECT AVG(mrr) as avg_mrr FROM customers",
     lambda q, y: {"avg_mrr": 2499.5}),

    # Product queries
    ("Find the cheapest product price",
     "SELECT name, price_monthly FROM products ORDER BY price_monthly ASC LIMIT 1",
     lambda q, y: {"name": "Nextera Starter", "price_monthly": 299}),
    ("What is the most expensive product monthly price?",
     "SELECT name, price_monthly FROM products ORDER BY price_monthly DESC LIMIT 1",
     lambda q, y: {"name": "Nextera Enterprise", "price_monthly": 3500}),
    ("What is the {product} plan monthly price?",
     "SELECT price_monthly FROM products WHERE name LIKE '%{product}%'",
     lambda q, y: {"price_monthly": {"Starter": 299, "Professional": 999, "Enterprise": 3500, "Fine-Tuning": 500}.get(q, 999)}),
    ("How many products do we offer?",
     "SELECT COUNT(*) as count FROM products",
     lambda q, y: {"count": 4}),

    # Churn/growth queries
    ("What is the churn rate in {quarter} {year}?",
     "SELECT churn_rate FROM sales WHERE year = {year} AND quarter = '{quarter}'",
     lambda q, y: {"churn_rate": random.choice([0.7, 0.8, 0.9, 1.0, 1.2, 1.5])}),
    ("What was ARR growth in {quarter} {year}?",
     "SELECT arr_growth_pct FROM sales WHERE year = {year} AND quarter = '{quarter}'",
     lambda q, y: {"arr_growth_pct": random.choice([18.2, 21.6, 24.3, 28.7])}),
    ("How many industries are in our customer base?",
     "SELECT COUNT(DISTINCT industry) as count FROM customers",
     lambda q, y: {"count": 10}),
]

# ---------------------------------------------------------------------------
# Templates for Calculator Turn 2 queries (matching inference-time phrasing)
# Three styles: explicit-numeric, concretized-vague, and mixed
# ---------------------------------------------------------------------------
CALC_TEMPLATES = [
    # --- Percentage calculations ---
    # Explicit: "Calculate 15% of 103200"
    ("Calculate {pct}% of {val}", "{val} * {pct} / 100"),
    ("What is {pct}% of {val}?", "{val} * {pct} / 100"),
    ("Compute {pct} percent of {val}", "{val} * {pct} / 100"),
    # Growth: "Calculate {pct}% growth on {val}"
    ("Calculate {pct}% growth on {val}", "{val} * (1 + {pct} / 100)"),
    ("What would {val} be with {pct}% growth?", "{val} * (1 + {pct} / 100)"),
    ("Apply {pct}% increase to {val}", "{val} * (1 + {pct} / 100)"),
    # Discount
    ("Calculate {pct}% discount on {val}", "{val} * (1 - {pct} / 100)"),
    ("What is {val} with a {pct}% discount?", "{val} * (1 - {pct} / 100)"),

    # --- Multiplication ---
    ("Calculate {val} times {mult}", "{val} * {mult}"),
    ("Multiply {val} by {mult}", "{val} * {mult}"),
    ("What is {val} multiplied by {mult}?", "{val} * {mult}"),
    ("Compute {val} times {mult}", "{val} * {mult}"),

    # --- Annual from monthly ---
    ("Calculate the annual cost at {val} per month", "{val} * 12"),
    ("What is the annual value of {val} monthly?", "{val} * 12"),
    ("Convert {val} monthly to annual", "{val} * 12"),

    # --- Division ---
    ("Calculate {val} divided by {div}", "{val} / {div}"),
    ("Divide {val} by {div}", "{val} / {div}"),
    ("What is {val} divided by {div}?", "{val} / {div}"),
    ("Calculate average from {val} total across {div} items", "{val} / {div}"),

    # --- Subtraction ---
    ("Calculate the difference between {val} and {val2}", "{val} - {val2}"),
    ("Subtract {val2} from {val}", "{val} - {val2}"),
    ("What is {val} minus {val2}?", "{val} - {val2}"),

    # --- Addition ---
    ("Add {val} and {val2}", "{val} + {val2}"),
    ("What is {val} plus {val2}?", "{val} + {val2}"),
    ("Calculate {val} plus {val2}", "{val} + {val2}"),

    # --- Concretized-vague (inference style) ---
    ("Calculate the growth on that revenue", "{val} * 1.15"),
    ("Calculate the annual cost for that price", "{val} * 12"),
    ("Calculate the total for those customers", "{val} * {mult}"),
    ("Compute the commission on that amount", "{val} * 0.05"),
    ("Calculate the annual recurring revenue", "{val} * 12"),
    ("What would doubling that amount give?", "{val} * 2"),
    ("Calculate the percentage increase", "(({val} - {val2}) / {val2}) * 100"),
    ("Compute the monthly run rate from quarterly revenue", "{val} / 3"),
    ("Calculate the per-customer average", "{val} / {div}"),
    ("What would that cost annually?", "{val} * 12"),
]

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
YEARS = [2023, 2024]
TIERS = ["Starter", "Professional", "Enterprise"]
PRODUCTS = ["Starter", "Professional", "Enterprise", "Fine-Tuning"]
PCTS = [5, 10, 15, 20, 25, 30]
MULTS = [2, 3, 4, 5, 10, 12, 25, 50, 100]
DIVS = [2, 3, 4, 5, 7, 8, 10, 12]


def _make_sql_turn(template_idx=None):
    """Generate a randomized SQL Turn 1."""
    if template_idx is None:
        template_idx = random.randint(0, len(SQL_TEMPLATES) - 1)
    query_tpl, sql_tpl, result_fn = SQL_TEMPLATES[template_idx]

    quarter = random.choice(QUARTERS)
    year = random.choice(YEARS)
    tier = random.choice(TIERS)
    product = random.choice(PRODUCTS)

    query = query_tpl.format(
        quarter=quarter, year=year, tier=tier, tier_lower=tier.lower(), product=product,
    )
    sql = sql_tpl.format(
        quarter=quarter, year=year, tier=tier, tier_lower=tier.lower(), product=product,
    )
    result = result_fn(product, year)

    return query, sql, result


def _extract_numeric(result):
    """Pull the first numeric value from a result dict."""
    for v in result.values():
        if isinstance(v, (int, float)):
            return v
    return 1000


def _make_calc_turn(sql_result, template_idx=None):
    """Generate a randomized Calculator Turn 2 based on SQL result."""
    val = _extract_numeric(sql_result)
    if template_idx is None:
        template_idx = random.randint(0, len(CALC_TEMPLATES) - 1)
    query_tpl, expr_tpl = CALC_TEMPLATES[template_idx]

    pct = random.choice(PCTS)
    mult = random.choice(MULTS)
    div = random.choice(DIVS)
    val2 = random.choice([299, 500, 999, 3500, 24700, val // 2 if val > 1 else 1])

    # Format — some templates may not use all vars, that's fine
    try:
        query = query_tpl.format(val=val, pct=pct, mult=mult, div=div, val2=val2)
    except (KeyError, IndexError):
        query = f"Calculate {val} * {mult}"

    try:
        expr = expr_tpl.format(val=val, pct=pct, mult=mult, div=div, val2=val2)
    except (KeyError, IndexError):
        expr = f"{val} * {mult}"

    return query, expr


def generate_examples(n=300):
    """Generate n multi-turn training examples."""
    examples = []
    for _ in range(n):
        sql_query, sql_stmt, sql_result = _make_sql_turn()
        calc_query, calc_expr = _make_calc_turn(sql_result)

        example = {
            "multi_turn": True,
            "turns": [
                {
                    "query": sql_query,
                    "tool_call": {
                        "name": "sql_query",
                        "arguments": {"query": sql_stmt},
                    },
                    "tool_result": sql_result,
                },
                {
                    "query": calc_query,
                    "tool_call": {
                        "name": "calculator",
                        "arguments": {"expression": calc_expr},
                    },
                },
            ],
        }
        examples.append(example)

    return examples


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    examples = generate_examples(n)

    outpath = f"{SCENARIO_CONFIG.training_data_dir}/tool_routing_multi_turn{SCENARIO_CONFIG.training_data_suffix}.jsonl"
    with open(outpath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} multi-turn examples → {outpath}")

    # Stats
    calc_queries = [ex["turns"][1]["query"] for ex in examples]
    vague = sum(1 for q in calc_queries if not any(c.isdigit() for c in q))
    explicit = len(calc_queries) - vague
    print(f"  Vague/concretized style: {vague} ({vague*100//len(calc_queries)}%)")
    print(f"  Explicit numeric style:  {explicit} ({explicit*100//len(calc_queries)}%)")


if __name__ == "__main__":
    main()
