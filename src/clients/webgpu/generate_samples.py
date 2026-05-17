#!/usr/bin/env python3
"""
Generate sample PDF documents for the WebGPU Document Triage demo.

Creates synthetic healthcare PDFs with fictional Nextera Medical Center
branding and synthetic PII for demonstrating entity extraction on-device.

All names, identifiers, addresses, phone numbers, dates, and clinical
findings are entirely fabricated. No real patient data.

Usage:
    pip install fpdf2
    python generate_samples.py

Output:
    samples/patient_intake_referral.pdf
    samples/lab_report_metabolic.pdf
"""

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos


SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)


def _ascii(text: str) -> str:
    """Replace Unicode chars with ASCII equivalents for core PDF fonts."""
    return (
        text
        .replace("\u2014", "--")   # em-dash
        .replace("\u2013", "-")    # en-dash
        .replace("\u2018", "'")    # left single quote
        .replace("\u2019", "'")    # right single quote
        .replace("\u201c", '"')    # left double quote
        .replace("\u201d", '"')    # right double quote
        .replace("\u2026", "...")  # ellipsis
        .replace("\u00b0", " deg") # degree
        .replace("\u2265", ">=")   # >=
    )

# ── Colors ──────────────────────────────────────────────────────────
NEXTERA_BLUE = (1, 112, 185)     # #0170B9
DARK_TEXT     = (26, 26, 46)     # #1a1a2e
GRAY_TEXT     = (100, 100, 120)
LIGHT_BG     = (244, 244, 248)   # #f4f4f8
WHITE         = (255, 255, 255)
RED_FLAG      = (204, 61, 53)


class NexteraPDF(FPDF):
    """Base PDF class with Nextera Medical Center branding."""

    def header(self):
        # Blue header bar
        self.set_fill_color(*NEXTERA_BLUE)
        self.rect(0, 0, 210, 18, "F")

        # Logo text
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*WHITE)
        self.set_xy(10, 3)
        self.cell(0, 12, "NEXTERA MEDICAL CENTER", align="L")

        # Right-side label
        self.set_font("Helvetica", "", 9)
        self.set_xy(10, 3)
        self.cell(190, 12, "SAMPLE - SYNTHETIC DATA", align="R")

        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 10, f"Synthetic demo document -- fictional Nextera Medical Center  |  Page {self.page_no()}", align="C")

    def section_header(self, title: str):
        """Print a section header with gray background."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*DARK_TEXT)
        self.set_fill_color(*LIGHT_BG)
        self.cell(0, 8, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(2)

    def label_value(self, label: str, value: str, bold_value: bool = False):
        """Print a label: value pair."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY_TEXT)
        self.cell(45, 5, label + ":")
        self.set_text_color(*DARK_TEXT)
        self.set_font("Helvetica", "B" if bold_value else "", 9)
        self.cell(0, 5, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def body_text(self, text: str):
        """Print body text."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK_TEXT)
        self.multi_cell(0, 5, text)
        self.ln(1)


def generate_patient_intake():
    """Generate patient_intake_referral.pdf."""
    pdf = NexteraPDF()
    pdf.add_page()

    # Document title
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*NEXTERA_BLUE)
    pdf.cell(0, 8, "Cardiology Department -- Referral Intake Form", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Patient Information
    pdf.section_header("PATIENT INFORMATION")
    pdf.label_value("Name", "Maria Gonzalez", bold_value=True)
    pdf.label_value("Date of Birth", "04/22/1965")
    pdf.label_value("Insurance", "BlueCross BlueShield, ID: BC-2847193")
    pdf.label_value("Address", "742 Evergreen Terrace, Stuttgart, 70174")
    pdf.label_value("Phone", "+49-711-555-0842")
    pdf.ln(3)

    # Referring Physician
    pdf.section_header("REFERRING PHYSICIAN")
    pdf.label_value("Physician", "Dr. Thomas Weber, MD")
    pdf.label_value("Department", "Internal Medicine, Nextera Primary Care")
    pdf.label_value("Date of Referral", "2026-02-25")
    pdf.ln(3)

    # Chief Complaint
    pdf.section_header("CHIEF COMPLAINT")
    pdf.body_text(
        "Persistent chest pain and shortness of breath for 3 weeks. "
        "Patient reports episodes of substernal chest pain, worse with exertion, "
        "radiating to the left arm. Episodes last 5-10 minutes, relieved by rest. "
        "Denies syncope, palpitations, or peripheral edema."
    )
    pdf.ln(2)

    # Medications
    pdf.section_header("CURRENT MEDICATIONS")
    meds = [
        ("Metoprolol 50mg", "daily", "Beta-blocker"),
        ("Lisinopril 10mg", "daily", "ACE inhibitor"),
        ("Aspirin 81mg", "daily", "Antiplatelet"),
    ]
    pdf.set_font("Helvetica", "", 9)
    for med, freq, cls in meds:
        pdf.set_text_color(*DARK_TEXT)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(40, 5, f"  {med}")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(25, 5, freq)
        pdf.cell(0, 5, f"({cls})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Allergies
    pdf.section_header("ALLERGIES")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*RED_FLAG)
    pdf.cell(0, 5, "  Sulfa drugs (rash)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "  Penicillin (anaphylaxis -- SEVERE)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Vitals
    pdf.section_header("VITALS AT REFERRAL")
    vitals = [
        ("Blood Pressure", "142/88 mmHg"),
        ("Resting Heart Rate", "78 bpm"),
        ("SpO2", "97% on room air"),
        ("Temperature", "36.8 C"),
        ("BMI", "28.4"),
    ]
    for label, value in vitals:
        pdf.label_value(label, value)
    pdf.ln(2)

    # Diagnostics
    pdf.section_header("DIAGNOSTICS")
    pdf.body_text(
        "ECG at referring office showed nonspecific ST-T wave changes in leads V4-V6. "
        "Chest X-ray: mild cardiomegaly, no infiltrates."
    )
    pdf.ln(1)

    # Family History
    pdf.section_header("FAMILY HISTORY")
    pdf.body_text("Father: Myocardial infarction at age 58 (deceased)")
    pdf.body_text("Mother: Hypertension, Type 2 Diabetes")
    pdf.body_text("Sister: No cardiac history")
    pdf.ln(1)

    # Assessment
    pdf.section_header("ASSESSMENT & REQUESTED EVALUATION")
    pdf.body_text(
        "61-year-old female with exertional chest pain, hypertension, and family history "
        "of premature coronary artery disease. ECG changes warrant further evaluation. "
        "Requesting stress echocardiography and cardiology consultation."
    )
    pdf.ln(6)

    # Signature
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 5, "Electronically signed: Dr. Thomas Weber, MD", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, "Nextera Medical Center -- Internal Medicine", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*RED_FLAG)
    pdf.cell(0, 5, "SYNTHETIC SAMPLE -- All names, IDs, and clinical findings are fabricated for demo purposes.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out = SAMPLES_DIR / "patient_intake_referral.pdf"
    pdf.output(str(out))
    print(f"  Created {out} ({out.stat().st_size / 1024:.0f} KB)")


def generate_lab_report():
    """Generate lab_report_metabolic.pdf."""
    pdf = NexteraPDF()
    pdf.add_page()

    # Document title
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*NEXTERA_BLUE)
    pdf.cell(0, 8, "Clinical Laboratory Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Patient Information
    pdf.section_header("PATIENT INFORMATION")
    pdf.label_value("Name", "Robert Chen", bold_value=True)
    pdf.label_value("MRN", "NEX-20240892")
    pdf.label_value("Date of Birth", "11/03/1978")
    pdf.label_value("Sex", "Male")
    pdf.label_value("Collection Date", "2026-02-28 08:15")
    pdf.label_value("Report Date", "2026-02-28 14:30")
    pdf.label_value("Fasting", "Yes (12 hours)")
    pdf.label_value("Ordering Physician", "Dr. Sarah Kim, Family Medicine")
    pdf.ln(3)

    # CMP Table
    pdf.section_header("COMPLETE METABOLIC PANEL")
    _lab_table_header(pdf)
    cmp_rows = [
        ("Glucose", "112 mg/dL", "70-100 mg/dL", "HIGH"),
        ("BUN", "18 mg/dL", "7-20 mg/dL", ""),
        ("Creatinine", "1.1 mg/dL", "0.7-1.3 mg/dL", ""),
        ("Sodium", "141 mEq/L", "136-145 mEq/L", ""),
        ("Potassium", "4.2 mEq/L", "3.5-5.0 mEq/L", ""),
        ("Chloride", "102 mEq/L", "98-106 mEq/L", ""),
        ("CO2", "24 mEq/L", "23-29 mEq/L", ""),
        ("Calcium", "9.4 mg/dL", "8.5-10.5 mg/dL", ""),
        ("Total Protein", "7.1 g/dL", "6.0-8.3 g/dL", ""),
        ("Albumin", "4.2 g/dL", "3.5-5.5 g/dL", ""),
        ("Total Bilirubin", "0.8 mg/dL", "0.1-1.2 mg/dL", ""),
        ("Alk Phosphatase", "78 U/L", "44-147 U/L", ""),
        ("AST (SGOT)", "28 U/L", "10-40 U/L", ""),
        ("ALT (SGPT)", "35 U/L", "7-56 U/L", ""),
        ("eGFR", "88 mL/min", ">60 mL/min", ""),
    ]
    for row in cmp_rows:
        _lab_table_row(pdf, *row)
    pdf.ln(3)

    # HbA1c
    pdf.section_header("HEMOGLOBIN A1C")
    _lab_table_header(pdf)
    _lab_table_row(pdf, "HbA1c", "6.8%", "<5.7%", "HIGH")
    _lab_table_row(pdf, "Est. Avg Glucose", "148 mg/dL", "", "")
    pdf.ln(3)

    # Lipid Panel
    pdf.section_header("LIPID PANEL")
    _lab_table_header(pdf)
    lipid_rows = [
        ("Total Cholesterol", "234 mg/dL", "<200 mg/dL", "HIGH"),
        ("LDL Cholesterol", "156 mg/dL", "<100 mg/dL", "HIGH"),
        ("HDL Cholesterol", "42 mg/dL", ">40 mg/dL", ""),
        ("Triglycerides", "180 mg/dL", "<150 mg/dL", "HIGH"),
        ("VLDL Cholesterol", "36 mg/dL", "5-40 mg/dL", ""),
        ("Total/HDL Ratio", "5.6", "<5.0", "HIGH"),
    ]
    for row in lipid_rows:
        _lab_table_row(pdf, *row)
    pdf.ln(3)

    # Interpretation
    pdf.section_header("INTERPRETATION")
    pdf.body_text(
        "Elevated HbA1c (6.8%) consistent with diabetes mellitus (ADA threshold >= 6.5%). "
        "Fasting glucose also elevated at 112 mg/dL. Recommend diabetology referral for "
        "initiation of therapy and dietary counseling."
    )
    pdf.body_text(
        "Dyslipidemia with significantly elevated LDL (156 mg/dL) and triglycerides "
        "(180 mg/dL). Combined with diabetes diagnosis, cardiovascular risk is elevated. "
        "Consider statin therapy per ACC/AHA guidelines."
    )
    pdf.body_text("Renal and hepatic panels within normal limits.")
    pdf.ln(4)

    # Signatures
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 5, "Verified by: Dr. Marcus Reeves, MD, Clinical Pathology", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, "Laboratory Director: Prof. Elena Vasquez, PhD", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, "CLIA #: 42D2067890", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.cell(0, 5, "Nextera Clinical Laboratory -- Stuttgart Campus", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*RED_FLAG)
    pdf.cell(0, 5, "SYNTHETIC SAMPLE -- All names, IDs, and lab values are fabricated for demo purposes.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out = SAMPLES_DIR / "lab_report_metabolic.pdf"
    pdf.output(str(out))
    print(f"  Created {out} ({out.stat().st_size / 1024:.0f} KB)")


def _lab_table_header(pdf: FPDF):
    """Print lab result table header."""
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*WHITE)
    pdf.set_fill_color(*NEXTERA_BLUE)
    pdf.cell(50, 6, "  Test", fill=True)
    pdf.cell(35, 6, "Result", fill=True, align="C")
    pdf.cell(40, 6, "Reference Range", fill=True, align="C")
    pdf.cell(20, 6, "Flag", fill=True, align="C")
    pdf.ln()


def _lab_table_row(pdf: FPDF, test: str, result: str, ref: str, flag: str):
    """Print a single lab result row."""
    is_high = flag in ("HIGH", "LOW")
    bg = (255, 245, 245) if is_high else WHITE

    pdf.set_fill_color(*bg)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*DARK_TEXT)
    pdf.cell(50, 5, f"  {test}", fill=True)

    if is_high:
        pdf.set_font("Helvetica", "B", 8)
    pdf.cell(35, 5, result, fill=True, align="C")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(40, 5, ref, fill=True, align="C")

    if flag:
        pdf.set_text_color(*RED_FLAG)
        pdf.set_font("Helvetica", "B", 8)
    else:
        pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(20, 5, flag, fill=True, align="C")
    pdf.ln()


if __name__ == "__main__":
    print("Generating sample PDFs for WebGPU Document Triage demo...\n")
    generate_patient_intake()
    generate_lab_report()
    print("\nDone.")
