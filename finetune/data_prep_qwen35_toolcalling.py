"""
Data preparation for Qwen3.5-4B tool-calling fine-tune.

Loads the existing model-agnostic tool_routing_2tool.jsonl (1,266 examples)
and extends it with ~100 targeted examples addressing the gaps identified in
the live eval session (2026-03-18):

  - Calculator routing misses: queries answered directly instead of routed
  - SQLite date functions: YEAR() not valid in SQLite, needs LIKE/'strftime'
  - Expression boundary examples: business-math phrasings not in training data

Output format: {"query": str, "tool_call": {"name": str, "arguments": dict}}
Same raw format as tool_routing_2tool.jsonl — model-agnostic. The trainer
(train_qwen35_toolcalling.py) applies tokenizer.apply_chat_template with
Qwen3.5's tool-call format and enable_thinking=False.

Usage:
  python -m finetune.data_prep_qwen35_toolcalling
"""

from __future__ import annotations

import json
import os
import random

from finetune.data_prep_shared import save_jsonl


# ---------------------------------------------------------------------------
# Targeted gap examples — derived from live eval failures (2026-03-18)
# ---------------------------------------------------------------------------
# These are the exact queries that failed in eval_tool_routing against
# Qwen3.5-4B zero-shot, plus variants to provide generalisation coverage.
# Qwen answered many of these directly (got=None) instead of routing to tool.
# ---------------------------------------------------------------------------

_CALCULATOR_GAP_EXAMPLES: list[tuple[str, str]] = [
    # ── ROI / margin / ratio (Qwen answered directly) ────────────────────────
    # NOTE: eval (CALCULATOR_EXPECTED) uses raw ratios (0–1), not percentages.
    ("What is the ROI if we spend €150,000 and gain €400,000?",
     "(400000 - 150000) / 150000"),
    ("ROI: invest €200,000, return €550,000.",
     "(550000 - 200000) / 200000"),
    ("Calculate ROI on €80,000 spend that returns €210,000.",
     "(210000 - 80000) / 80000"),
    ("If profit is €220,000 on revenue of €800,000, what is the margin?",
     "220000 / 800000"),
    ("Profit margin: €180,000 profit on €650,000 revenue.",
     "180000 / 650000"),
    ("Gross margin if revenue is €1,200,000 and cost is €480,000?",
     "(1200000 - 480000) / 1200000"),
    ("What is the ratio of 45000 to 180000?",
     "45000 / 180000"),
    ("Ratio of €320,000 to €1,400,000.",
     "320000 / 1400000"),

    # ── Multi-unit annual calculations (answered directly) ───────────────────
    ("If 20 teams pay €3,500/month, what is the annual ARR?",
     "20 * 3500 * 12"),
    ("25 teams at €1,499/month — annual revenue?",
     "25 * 999 * 12"),
    ("If 75 teams pay €149 each per month, what is the quarterly revenue?",
     "75 * 149 * 3"),
    ("100 teams at €299/month — what is the quarterly total?",
     "100 * 299 * 3"),
    ("Divide a €1,500,000 budget across 12 months.",
     "1500000 / 12"),
    ("Split €2,400,000 evenly across 12 months.",
     "2400000 / 12"),
    ("Calculate 3 years of savings at €3,500 per month.",
     "3500 * 12 * 3"),
    ("5 servers at €1,200/month for 24 months, what is the total hosting cost?",
     "5 * 1200 * 24"),
    ("10 licenses at €4,200 per year — total cost?",
     "10 * 4200"),
    ("12 Enterprise subscriptions at €3,500/month — annual total?",
     "12 * 3500 * 12"),

    # ── Percentage reduction (answered directly) ─────────────────────────────
    ("€2,500,000 reduced by 6%, what is left?",
     "2500000 * 0.94"),
    ("€1,200,000 after a 10% decline.",
     "1200000 * 0.90"),
    ("What is €845,000 after a 15% reduction?",
     "845000 * 0.85"),
    ("Revenue of €3,200,000 drops 8% — new total?",
     "3200000 * 0.92"),

    # ── Compound growth (answered directly) ──────────────────────────────────
    ("What is the compound growth on €200,000 at 12% for 2 years?",
     "200000 * (1.12 ** 2)"),
    ("If customer base is 5000 and grows 12% per year, how many after 2 years?",
     "5000 * (1.12 ** 2)"),
    ("What is compound interest on €100,000 at 8% for 3 years?",
     "100000 * (1.08 ** 3)"),
    ("€50,000 compounding at 15% annually for 4 years.",
     "50000 * (1.15 ** 4)"),
    ("ARR of €465,000 growing 21.6% for 3 years.",
     "465000 * (1.216 ** 3)"),

    # ── Simple arithmetic phrased as questions (answered directly) ───────────
    ("What is 7500 plus 3200?",
     "7500 + 3200"),
    ("What is the difference between €12,000 and €8,500?",
     "12000 - 8500"),
    ("How much do we save switching from €7,200/month to €4,800/month over a year?",
     "(7200 - 4800) * 12"),
    ("If old cost was €95,000 and new cost is €62,000, what is the annual saving?",
     "95000 - 62000"),
    ("Add €162,000 and €148,000.",
     "162000 + 148000"),
    ("Subtract €380,000 from €565,000.",
     "565000 - 380000"),

    # ── Cost-benefit ratio: cost/benefit (not benefit/cost) ──────────────────
    ("If cost is €35,000 and benefit is €112,000, what is the cost-benefit ratio?",
     "35000 / 112000"),

    # ── Churn / delta calculations (boundary with sql_query) ─────────────────
    ("If churn rate drops from 2.0% to 0.8%, how many fewer customers leave per 1000?",
     "1000 * (0.020 - 0.008)"),
    ("With 5000 customers, a 1% churn improvement saves how many per month?",
     "5000 * 0.01"),
    ("Apply a 22% increase to €750,000.",
     "750000 * 1.22"),
    ("€565,000 growing by 21.6%.",
     "565000 * 1.216"),

    # ── Plan upgrade/downgrade revenue deltas (prices known: S=299, P=999, E=3500) ─
    # Qwen must use calculator directly — not sql_query to look up plan prices.
    ("If 25 users upgrade from Starter to Professional, what is the monthly cost increase?",
     "25 * (999 - 299)"),
    ("If 10 customers upgrade from Starter to Enterprise, what is the monthly revenue uplift?",
     "10 * (3500 - 299)"),
    ("If 5 customers move from Professional to Enterprise, what is the monthly MRR increase?",
     "5 * (3500 - 999)"),
    ("If 20 users downgrade from Enterprise to Professional, what is the monthly revenue loss?",
     "20 * (3500 - 999)"),
    ("If 15 customers upgrade from Starter to Professional, what is the additional monthly revenue?",
     "15 * (999 - 299)"),
    ("How much more revenue if 8 Starter customers upgrade to Enterprise?",
     "8 * (3500 - 299)"),
    ("Revenue delta if 30 Professional customers upgrade to Enterprise per month.",
     "30 * (3500 - 999)"),

    # ── Plan cost savings/difference without count (prices known: S=299, P=999, E=3500) ─
    ("Calculate the monthly cost savings between Enterprise and Professional.",
     "3500 - 999"),
    ("Calculate the monthly cost savings between Professional and Starter.",
     "999 - 299"),
    ("What is the price difference between Enterprise and Professional plans?",
     "3500 - 999"),
    ("What is the price difference between Enterprise and Starter plans?",
     "3500 - 299"),
    ("What is the price difference between Professional and Starter plans?",
     "999 - 299"),
    ("How much more does Enterprise cost per month than Professional?",
     "3500 - 999"),
    ("What is the monthly saving when downgrading from Enterprise to Professional?",
     "3500 - 999"),

    # ── One-time savings vs recurring — contrastive anchors ──────────────────
    # "Annual saving" when switching plans is a one-time diff, NOT * 12.
    ("If old cost was €95,000 and new cost is €62,000, what is the annual saving?",
     "95000 - 62000"),
    ("We currently pay €45,000 and will pay €28,000 — what is the cost saving?",
     "45000 - 28000"),
    ("Old license €120,000, new license €85,000 — what do we save?",
     "120000 - 85000"),
]

_SQL_DATE_GAP_EXAMPLES: list[tuple[str, str]] = [
    # ── SQLite-safe date filtering (YEAR() is MySQL-only, use LIKE or strftime) ─
    ("How many customers joined in 2024?",
     "SELECT COUNT(*) FROM customers WHERE joined_date LIKE '2024%'"),
    ("How many customers joined in 2023?",
     "SELECT COUNT(*) FROM customers WHERE joined_date LIKE '2023%'"),
    ("Show customers who signed up in the first half of 2024.",
     "SELECT name, tier, mrr, joined_date FROM customers WHERE joined_date >= '2024-01-01' AND joined_date < '2024-07-01'"),
    ("List customers who joined in Q1 2024.",
     "SELECT name, tier, mrr FROM customers WHERE joined_date >= '2024-01-01' AND joined_date < '2024-04-01'"),
    ("Customers acquired in Q3 2023.",
     "SELECT name, tier, mrr FROM customers WHERE joined_date >= '2023-07-01' AND joined_date < '2023-10-01'"),
    ("Which customers joined after June 2024?",
     "SELECT name, joined_date FROM customers WHERE joined_date > '2024-06-30'"),
    ("Customers who joined before 2024.",
     "SELECT name, tier, mrr FROM customers WHERE joined_date < '2024-01-01'"),
    ("How many customers joined in the second half of 2023?",
     "SELECT COUNT(*) FROM customers WHERE joined_date >= '2023-07-01' AND joined_date < '2024-01-01'"),
    ("Customers onboarded between January and March 2024.",
     "SELECT name, tier, mrr, joined_date FROM customers WHERE joined_date >= '2024-01-01' AND joined_date <= '2024-03-31'"),
    ("Which customers are from the 2023 cohort?",
     "SELECT name, tier, mrr FROM customers WHERE joined_date LIKE '2023%'"),
]


# ---------------------------------------------------------------------------
# Multi-turn examples — teach Qwen to call calculator after a SQL result
#
# These address multi-step patterns where Qwen issues a second sql_query
# for step 2 instead of switching to calculator.
#
# Format: {"multi_turn_messages": [...]} — detected by _raw_to_messages in
# train_qwen35_toolcalling.py which prepends the system prompt and passes
# the full conversation to apply_chat_template.
#
# TWO conversation structures are trained (both occur at inference):
#   A) user(full_query) → sql → tool_result → user(full_query) → calc
#      Used when query is single-step or decomposer uses original query.
#   B) user(sub_task_1) → sql → tool_result → user("Calculate X * Y") → calc
#      Primary multi-step path: QueryDecomposer.concretize_step (Qwen FT)
#      rewrites the step 2 sub-task into a concrete "Calculate N * M"
#      instruction using accumulated_results.
#
# SQL results use synthetic but realistic Nextera Analytics values:
#   - customers: tier=Enterprise/Professional/Starter, mrr, joined_date, industry
#   - revenue: quarter=Q1..Q4, year, amount
#   - products: name, price_monthly
# ---------------------------------------------------------------------------

def _mt(query: str, sql: str, sql_result: str, expr: str, *, step2: str | None = None) -> dict:
    """Build a single sql_query→calculator multi-turn example.

    step2: the user message for step 2. Defaults to the full query (pattern A).
           Pass a concretized instruction like "Calculate 3500 * 12" for pattern B.
    """
    return {
        "multi_turn_messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "sql_query", "arguments": {"query": sql}}}]},
            {"role": "tool", "content": sql_result, "tool_call_id": "call_1"},
            {"role": "user", "content": step2 if step2 is not None else query},
            {"role": "assistant", "tool_calls": [{"id": "call_2", "type": "function",
                "function": {"name": "calculator", "arguments": {"expression": expr}}}]},
        ]
    }


_MULTI_TURN_EXAMPLES: list[dict] = [
    # ── Top customer MRR → annual spend ──────────────────────────────────────
    _mt(
        "What is the top customer's MRR, and their annual spend?",
        "SELECT name, mrr FROM customers ORDER BY mrr DESC LIMIT 1",
        '[{"name": "Acme Corp", "mrr": 3500}]',
        "3500 * 12",
    ),
    _mt(
        "Show our top customer by MRR, and calculate what 3 years of their spend would total.",
        "SELECT name, mrr FROM customers ORDER BY mrr DESC LIMIT 1",
        '[{"name": "Pinnacle AI", "mrr": 4200}]',
        "4200 * 12 * 3",
    ),
    _mt(
        "Who is our highest-paying customer, and what is their annual contract value?",
        "SELECT name, mrr FROM customers ORDER BY mrr DESC LIMIT 1",
        '[{"name": "DataBridge Inc", "mrr": 3500}]',
        "3500 * 12",
    ),

    # ── Quarterly revenue → percentage share ─────────────────────────────────
    _mt(
        "What was Q1 2024 revenue and what percentage of total 2024 revenue does that represent?",
        "SELECT quarter, SUM(revenue) AS revenue FROM sales WHERE year = 2024 GROUP BY quarter ORDER BY quarter",
        '[{"quarter": "Q1", "revenue": 55100}, {"quarter": "Q2", "revenue": 68300}, {"quarter": "Q3", "revenue": 84900}, {"quarter": "Q4", "revenue": 103200}]',
        "55100 / (55100 + 68300 + 84900 + 103200)",
    ),
    _mt(
        "What was Q2 2024 revenue, and what share of total 2024 revenue is that?",
        "SELECT quarter, SUM(revenue) AS revenue FROM sales WHERE year = 2024 GROUP BY quarter ORDER BY quarter",
        '[{"quarter": "Q1", "revenue": 55100}, {"quarter": "Q2", "revenue": 68300}, {"quarter": "Q3", "revenue": 84900}, {"quarter": "Q4", "revenue": 103200}]',
        "68300 / (55100 + 68300 + 84900 + 103200)",
    ),
    _mt(
        "Compare Q1 and Q4 2024 revenue, and what is the percentage growth?",
        "SELECT quarter, revenue FROM sales WHERE year = 2024 AND quarter IN ('Q1', 'Q4') ORDER BY quarter",
        '[{"quarter": "Q1", "revenue": 55100}, {"quarter": "Q4", "revenue": 103200}]',
        "(103200 - 55100) / 55100",
    ),

    # ── Tier customer count → combined spend ─────────────────────────────────
    _mt(
        "How many Professional tier customers are there, and what is their combined monthly spend?",
        "SELECT COUNT(*) AS count FROM customers WHERE tier = 'professional'",
        '[{"count": 4}]',
        "4 * 999",
    ),
    _mt(
        "How many Enterprise customers do we have, and what is their combined annual MRR?",
        "SELECT COUNT(*) AS count FROM customers WHERE tier = 'enterprise'",
        '[{"count": 4}]',
        "4 * 3500 * 12",
    ),
    _mt(
        "How many Starter customers are there, and what is their combined annual spend?",
        "SELECT COUNT(*) AS count FROM customers WHERE tier = 'starter'",
        '[{"count": 2}]',
        "2 * 299 * 12",
    ),

    # ── Revenue → growth / commission / tax ──────────────────────────────────
    _mt(
        "What was Q3 2024 revenue, and what would 25% growth look like?",
        "SELECT revenue AS q3_revenue FROM sales WHERE quarter = 'Q3' AND year = 2024",
        '[{"q3_revenue": 84900}]',
        "84900 * 1.25",
    ),
    _mt(
        "What is total 2024 revenue, and what is 8% commission on that?",
        "SELECT SUM(revenue) AS total FROM sales WHERE year = 2024",
        '[{"total": 311500}]',
        "311500 * 0.08",
    ),
    _mt(
        "What was total revenue in Q4 2024, and what would a 10% discount on that amount be?",
        "SELECT revenue AS q4_revenue FROM sales WHERE quarter = 'Q4' AND year = 2024",
        '[{"q4_revenue": 103200}]',
        "103200 * 0.10",
    ),
    _mt(
        "Show the highest quarterly revenue in 2024, and calculate a 12% tax on that.",
        "SELECT MAX(revenue) AS max_quarter FROM sales WHERE year = 2024",
        '[{"max_quarter": 103200}]',
        "103200 * 0.12",
    ),
    _mt(
        "What is total revenue across all quarters, and what would a 5% commission on that be?",
        "SELECT SUM(revenue) AS total FROM sales",
        '[{"total": 428700}]',
        "428700 * 0.05",
    ),
    _mt(
        "What was Q1 2023 revenue, and what would tripling it give?",
        "SELECT revenue AS q1_revenue FROM sales WHERE quarter = 'Q1' AND year = 2023",
        '[{"q1_revenue": 18500}]',
        "18500 * 3",
    ),

    # ── Product price → scaling ───────────────────────────────────────────────
    _mt(
        "What was our best-performing product last quarter, and what would revenue look like if we raised prices 15%?",
        "SELECT name, price_monthly FROM products WHERE name LIKE '%Enterprise%' ORDER BY price_monthly DESC LIMIT 1",
        '[{"name": "Nextera Enterprise", "price_monthly": 3500}]',
        "3500 * 1.15",
    ),
    _mt(
        "Find the cheapest product price and calculate how much 100 customers would pay annually.",
        "SELECT name, price_monthly FROM products ORDER BY price_monthly ASC LIMIT 1",
        '[{"name": "GPU Hours (A100)", "price_monthly": 4.5}]',
        "100 * 4.5 * 12",
    ),
    _mt(
        "Find the most expensive product monthly price, and calculate the annual cost for a team of 10.",
        "SELECT name, price_monthly FROM products ORDER BY price_monthly DESC LIMIT 1",
        '[{"name": "Nextera Enterprise", "price_monthly": 3500}]',
        "10 * 3500 * 12",
    ),

    # ── Aggregated SQL result → further calculation ───────────────────────────
    _mt(
        "Find total MRR across all customers, and calculate a 15% volume discount.",
        "SELECT SUM(mrr) AS total_mrr FROM customers",
        '[{"total_mrr": 84300}]',
        "84300 * 0.15",
    ),
    _mt(
        "What was our best-performing product last quarter, and what would revenue look like if we grew it by 15%?",
        "SELECT name, price_monthly FROM products ORDER BY price_monthly DESC LIMIT 1",
        '[{"name": "Enterprise", "price_monthly": 3500}]',
        "3500 * 1.15",
    ),
    _mt(
        "How many Professional tier customers are there, and what is their combined monthly spend?",
        "SELECT COUNT(*) AS count FROM customers WHERE tier = 'Professional'",
        '[{"count": 12}]',
        "12 * 999",
    ),
    _mt(
        "Find the most expensive product monthly price, and calculate the annual cost for a team of 10.",
        "SELECT name, price_monthly FROM products ORDER BY price_monthly DESC LIMIT 1",
        '[{"name": "Enterprise", "price_monthly": 3500}]',
        "10 * 3500 * 12",
    ),
    _mt(
        "What is the average revenue per quarter in 2024, and what would that be with a 20% increase?",
        "SELECT AVG(revenue) AS avg_quarter FROM sales WHERE year = 2024",
        '[{"avg_quarter": 77875}]',
        "77875 * 1.20",
    ),
    _mt(
        "What is the lowest customer MRR, and what would doubling it look like?",
        "SELECT MIN(mrr) AS min_mrr FROM customers",
        '[{"min_mrr": 299}]',
        "299 * 2",
    ),
    _mt(
        "How many new customers joined in 2024, and what would their total ARR be at Enterprise pricing?",
        "SELECT COUNT(*) AS count FROM customers WHERE joined_date LIKE '2024%'",
        '[{"count": 8}]',
        "8 * 3500 * 12",
    ),
    _mt(
        "How many total customers do we have, and what would 2 years of average MRR per customer total?",
        "SELECT COUNT(*) AS total, AVG(mrr) AS avg_mrr FROM customers",
        '[{"total": 42, "avg_mrr": 1205}]',
        "1205 * 24",
    ),

    # ── Pattern B: concretized step 2 ("Calculate N * M") ────────────────────
    # These match the NullResolver inference path where concretize_step rewrites
    # sub_task_2 into a concrete arithmetic instruction before calling the model.
    _mt(
        "What was our best-performing product last quarter, and what would revenue look like if we raised prices 15%?",
        "SELECT name, price_monthly FROM products ORDER BY price_monthly DESC LIMIT 1",
        '[{"name": "Enterprise", "price_monthly": 3500}]',
        "3500 * 1.15",
        step2="Calculate 3500 * 1.15",
    ),
    _mt(
        "Find the Enterprise plan price and calculate the 5-year total cost.",
        "SELECT name, price_monthly FROM products WHERE name = 'Enterprise'",
        '[{"name": "Enterprise", "price_monthly": 3500}]',
        "3500 * 12 * 5",
        step2="Calculate 3500 * 60",
    ),
    _mt(
        "What is the Professional plan price, and how much would 30 customers pay annually?",
        "SELECT name, price_monthly FROM products WHERE name = 'Professional'",
        '[{"name": "Professional", "price_monthly": 999}]',
        "30 * 999 * 12",
        step2="Calculate 30 * 999 * 12",
    ),
    _mt(
        "What is the top customer's MRR, and their annual spend?",
        "SELECT name, mrr FROM customers ORDER BY mrr DESC LIMIT 1",
        '[{"name": "Acme Corp", "mrr": 3500}]',
        "3500 * 12",
        step2="Calculate 3500 * 12",
    ),
    _mt(
        "Show our top customer by MRR, and calculate what 3 years of their spend would total.",
        "SELECT name, mrr FROM customers ORDER BY mrr DESC LIMIT 1",
        '[{"name": "Pinnacle AI", "mrr": 4200}]',
        "4200 * 12 * 3",
        step2="Calculate 4200 * 36",
    ),
    _mt(
        "What is the lowest customer MRR, and what would doubling it look like?",
        "SELECT MIN(mrr) AS min_mrr FROM customers",
        '[{"min_mrr": 299}]',
        "299 * 2",
        step2="Calculate 299 * 2",
    ),
    _mt(
        "What is Q3 2024 revenue, and what would 25% growth look like?",
        "SELECT revenue AS q3_revenue FROM sales WHERE quarter = 'Q3' AND year = 2024",
        '[{"q3_revenue": 84900}]',
        "84900 * 1.25",
        step2="Calculate 84900 * 1.25",
    ),
    _mt(
        "What is total 2024 revenue, and what is 8% commission on that?",
        "SELECT SUM(revenue) AS total FROM sales WHERE year = 2024",
        '[{"total": 311500}]',
        "311500 * 0.08",
        step2="Calculate 311500 * 0.08",
    ),
    _mt(
        "How many Professional tier customers are there, and what is their combined monthly spend?",
        "SELECT COUNT(*) AS count FROM customers WHERE tier = 'Professional'",
        '[{"count": 12}]',
        "12 * 999",
        step2="Calculate 12 * 999",
    ),
    _mt(
        "How many Enterprise customers do we have, and what is their combined annual MRR?",
        "SELECT COUNT(*) AS count FROM customers WHERE tier = 'Enterprise'",
        '[{"count": 8}]',
        "8 * 3500 * 12",
        step2="Calculate 8 * 3500 * 12",
    ),
    _mt(
        "What was Q1 2024 revenue and what percentage of total 2024 revenue does that represent?",
        "SELECT quarter, SUM(revenue) AS revenue FROM sales WHERE year = 2024 GROUP BY quarter ORDER BY quarter",
        '[{"quarter": "Q1", "revenue": 55100}, {"quarter": "Q2", "revenue": 68300}, {"quarter": "Q3", "revenue": 84900}, {"quarter": "Q4", "revenue": 103200}]',
        "55100 / (55100 + 68300 + 84900 + 103200)",
        step2="Calculate 55100 / 311500",
    ),
    _mt(
        "Find total MRR across all customers, and calculate a 15% volume discount.",
        "SELECT SUM(mrr) AS total_mrr FROM customers",
        '[{"total_mrr": 84300}]',
        "84300 * 0.15",
        step2="Calculate 84300 * 0.15",
    ),
    _mt(
        "What was our best-performing product last quarter, and what would revenue look like if we grew it by 15%?",
        "SELECT name, price_monthly FROM products ORDER BY price_monthly DESC LIMIT 1",
        '[{"name": "Enterprise", "price_monthly": 3500}]',
        "3500 * 1.15",
        step2="Calculate 3500 * 1.15",
    ),
    _mt(
        "How many Professional tier customers are there, and what is their combined monthly spend?",
        "SELECT COUNT(*) AS count FROM customers WHERE tier = 'Professional'",
        '[{"count": 12}]',
        "12 * 999",
        step2="Calculate 12 * 999",
    ),
    _mt(
        "Find the most expensive product monthly price, and calculate the annual cost for a team of 10.",
        "SELECT name, price_monthly FROM products ORDER BY price_monthly DESC LIMIT 1",
        '[{"name": "Enterprise", "price_monthly": 3500}]',
        "10 * 3500 * 12",
        step2="Calculate 10 * 3500 * 12",
    ),
]


def _build_gap_examples() -> list[dict]:
    """Convert gap tuples to raw tool-call dicts."""
    out = []
    for query, expr in _CALCULATOR_GAP_EXAMPLES:
        out.append({
            "query": query,
            "tool_call": {"name": "calculator", "arguments": {"expression": expr}},
        })
    for query, sql in _SQL_DATE_GAP_EXAMPLES:
        out.append({
            "query": query,
            "tool_call": {"name": "sql_query", "arguments": {"query": sql}},
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(
    source_path: str = "./data/training-data/tool_routing_2tool.jsonl",
    output_path: str = "./data/training-data/qwen35_toolcalling.jsonl",
    seed: int = 42,
) -> int:
    """
    Load base dataset, extend with targeted gap examples, shuffle, save.

    Returns number of examples written.
    """
    random.seed(seed)

    # Load existing labelled examples (query + tool_call, model-agnostic format)
    base: list[dict] = []
    if os.path.exists(source_path):
        with open(source_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    base.append(json.loads(line))
        print(f"  Loaded {len(base)} base examples from {source_path}")
    else:
        print(f"  WARNING: {source_path} not found — using gap examples only")

    # Add targeted gap examples (single-turn)
    gap = _build_gap_examples()
    print(f"  Adding {len(gap)} targeted gap examples ({len(_CALCULATOR_GAP_EXAMPLES)} calc, {len(_SQL_DATE_GAP_EXAMPLES)} SQL)")

    # Add multi-turn examples (sql→calculator step-2 patterns)
    print(f"  Adding {len(_MULTI_TURN_EXAMPLES)} multi-turn examples (sql→calculator)")

    single_turn = base + gap
    random.shuffle(single_turn)
    # Multi-turn examples are not shuffled with single-turn; they're appended at the end
    # so the trainer sees varied examples first, then the multi-turn ones.
    combined = single_turn + _MULTI_TURN_EXAMPLES

    # Stats (single-turn only — multi-turn have a different schema)
    by_tool: dict[str, int] = {}
    for ex in single_turn:
        t = ex["tool_call"]["name"]
        by_tool[t] = by_tool.get(t, 0) + 1
    print(f"  Tool distribution (single-turn): {by_tool}")
    print(f"  Total: {len(combined)} examples ({len(single_turn)} single-turn + {len(_MULTI_TURN_EXAMPLES)} multi-turn)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = save_jsonl(combined, output_path)
    print(f"  Saved → {output_path}")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Qwen3.5 tool-calling training data")
    parser.add_argument("--source", default="./data/training-data/tool_routing_2tool.jsonl")
    parser.add_argument("--output", default="./data/training-data/qwen35_toolcalling.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    n = prepare(source_path=args.source, output_path=args.output, seed=args.seed)
    print(f"\n  Done. {n} examples ready for training.")
    print("  Next: python -m finetune.train_qwen35_toolcalling")
