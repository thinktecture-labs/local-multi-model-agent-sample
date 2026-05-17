#!/usr/bin/env python3
"""
Generate a synthetic Nextera quarterly report PDF for OCR demo.

Produces a 2-page PDF with tables and prose using the exact same data
as data/loader.py — making OCR-extracted answers cross-verifiable
against SQL queries for the keynote demo.

Output: data/demo-documents/nextera_quarterly_report.pdf

Requirements:
    pip install fpdf2

Usage:
    python scripts/generate_ocr_demo_doc.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fpdf import FPDF
except ImportError:
    print("fpdf2 not installed. Run: pip install fpdf2")
    sys.exit(1)

from src.engine.inference.config import SCENARIO_CONFIG


# ---------------------------------------------------------------------------
# Data (mirrors data/loader.py SQL_SEED exactly)
# ---------------------------------------------------------------------------

SALES = [
    {"quarter": "Q1 2024", "revenue": 55_100, "new_customers": 7, "churn_rate": 1.0, "arr_growth": 28.7},
    {"quarter": "Q2 2024", "revenue": 68_300, "new_customers": 8, "churn_rate": 0.9, "arr_growth": 23.9},
    {"quarter": "Q3 2024", "revenue": 84_900, "new_customers": 9, "churn_rate": 0.8, "arr_growth": 24.3},
    {"quarter": "Q4 2024", "revenue": 103_200, "new_customers": 11, "churn_rate": 0.7, "arr_growth": 21.6},
]

CUSTOMERS = [
    {"name": "BrightHealth GmbH", "tier": "Enterprise", "mrr": 7_000, "industry": "Healthcare", "joined": "2023-03-01"},
    {"name": "FinVault SA", "tier": "Enterprise", "mrr": 5_000, "industry": "Finance", "joined": "2024-01-08"},
    {"name": "Horizon AI", "tier": "Enterprise", "mrr": 4_200, "industry": "Technology", "joined": "2024-03-22"},
    {"name": "Acme Corp", "tier": "Enterprise", "mrr": 3_500, "industry": "Manufacturing", "joined": "2023-01-15"},
    {"name": "GreenOps BV", "tier": "Professional", "mrr": 1_499, "industry": "Energy", "joined": "2024-02-14"},
]

PRODUCTS = [
    {"name": "Nextera Starter", "monthly": 299, "annual": 2_990},
    {"name": "Nextera Professional", "monthly": 999, "annual": 9_990},
    {"name": "Nextera Enterprise", "monthly": 3_500, "annual": 35_000},
]

TOTAL_REVENUE_2024 = sum(s["revenue"] for s in SALES)
TOTAL_CUSTOMERS = 10  # from loader.py
INDUSTRIES = ["Manufacturing", "Healthcare", "Software", "Analytics", "Education",
              "Finance", "Energy", "Technology", "Insurance", "Logistics"]


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

class NexteraReport(FPDF):
    """Custom PDF with Nextera branding."""

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, "SAMPLE - Synthetic Nextera Demo Data", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def section_subtitle(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
        """Render a table with header row and data rows."""
        if col_widths is None:
            w = int((self.w - 20) / len(headers))
            col_widths = [w] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, val, border=1, align="C")
            self.ln()
        self.ln(4)


def generate_pdf(output_path: str) -> None:
    pdf = NexteraReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Page 1: Quarterly Business Review ──────────────────────────────

    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 20, 80)
    pdf.cell(0, 12, "Nextera Platform", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Q4 2024 Business Review", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Revenue table
    pdf.section_subtitle("Quarterly Revenue Performance")
    pdf.add_table(
        headers=["Quarter", "Revenue", "New Customers", "Churn Rate", "ARR Growth"],
        rows=[
            [
                s["quarter"],
                f"EUR{s['revenue']:,.0f}",
                str(s["new_customers"]),
                f"{s['churn_rate']}%",
                f"{s['arr_growth']}%",
            ]
            for s in SALES
        ],
        col_widths=[30, 35, 35, 30, 30],
    )

    pdf.body_text(
        f"Total 2024 revenue reached EUR{TOTAL_REVENUE_2024:,.0f}, representing strong "
        f"quarter-over-quarter growth throughout the year. Q4 2024 was the highest-revenue "
        f"quarter at EUR103,200 with 11 new customers acquired and churn declining to 0.7%."
    )

    # Customer table (top 5 by MRR)
    pdf.section_subtitle("Top Customers by Monthly Recurring Revenue")
    pdf.add_table(
        headers=["Customer", "Tier", "MRR", "Industry", "Joined"],
        rows=[
            [
                c["name"],
                c["tier"],
                f"EUR{c['mrr']:,.0f}",
                c["industry"],
                c["joined"],
            ]
            for c in CUSTOMERS
        ],
        col_widths=[40, 30, 25, 30, 28],
    )

    pdf.body_text(
        f"BrightHealth GmbH remains the highest-MRR customer at EUR7,000/month on the "
        f"Enterprise tier. The top 5 customers represent the majority of recurring revenue."
    )

    # ── Page 2: Product & Pricing Summary ──────────────────────────────

    pdf.add_page()

    pdf.section_title("Product & Pricing Summary")

    pdf.add_table(
        headers=["Product", "Monthly Price", "Annual Price"],
        rows=[
            [p["name"], f"EUR{p['monthly']:,.0f}", f"EUR{p['annual']:,.0f}"]
            for p in PRODUCTS
        ],
        col_widths=[55, 40, 40],
    )

    pdf.body_text(
        f"Nextera offers three pricing tiers: Starter (EUR299/month), Professional "
        f"(EUR999/month), and Enterprise (EUR3,500/month). All tiers include the core "
        f"platform with increasing limits on users, API calls, and support SLA."
    )

    pdf.section_subtitle("Key Metrics")
    pdf.body_text(
        f"Total customers: {TOTAL_CUSTOMERS}\n"
        f"Industries represented: {', '.join(INDUSTRIES)}\n"
        f"Total 2024 revenue: EUR{TOTAL_REVENUE_2024:,.0f}\n"
        f"Best quarter: Q4 2024 (EUR103,200)\n"
        f"Lowest churn: Q4 2024 (0.7%)\n"
        f"Highest ARR growth: Q1 2024 (28.7%)"
    )

    pdf.section_subtitle("Industry Distribution")
    pdf.body_text(
        "The customer base spans 10 industries with strong representation in "
        "Manufacturing, Healthcare, Software, and Finance. Enterprise-tier customers "
        "are concentrated in Healthcare (BrightHealth GmbH) and Finance (FinVault SA), "
        "reflecting the platform's strength in regulated industries that require "
        "on-premises AI deployment and data sovereignty compliance."
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pdf.output(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = os.path.join(SCENARIO_CONFIG.demo_documents_dir, "nextera_quarterly_report.pdf")
    print(f"Generating {output}...")
    generate_pdf(output)
    size_kb = os.path.getsize(output) / 1024
    print(f"Done! {size_kb:.1f} KB, 2 pages")
    print(f"  Revenue table: 4 quarters of 2024 data")
    print(f"  Customer table: top 5 by MRR")
    print(f"  Pricing table: 3 product tiers")
    print(f"  Cross-validate: Q3 2024 revenue = EUR84,900 (matches SQL)")
