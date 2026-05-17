# Demo Queries: Meridian Health ADR

Use these after dropping the Meridian Health PDF into the Observatory.

## Kill the WiFi Demo

1. Toggle airplane mode ON
2. Ask: "What database did Meridian choose?" -- works, all local
3. Try compare mode -- disabled (grayed out)
4. Toggle airplane mode OFF

## Hybrid Routing Demo (HITL)

1. Toggle to HYBRID mode
2. Ask: "Compare Meridian's data residency approach with AWS Healthcare on compliance coverage"
3. Local model answers weakly, confidence ~30% (entity grounding penalty: Meridian/AWS not in Nextera KB)
4. Trace shows "BELOW THRESHOLD -- user approval needed"
5. Click "Escalate to GPT-5.4?" button
6. GPT-5.4 responds with detailed comparison, cost shows ~$0.0014
7. Privacy badge updates -- first external bytes sent (prompt tokens x 4 bytes)

## Keynote Highlights

### Data Sovereignty (the money question)
**Q:** What are Meridian's data residency requirements?
**A:** All patient data must remain within EU data centers at all times. No patient identifiable information (PII) may be processed, stored, or cached outside the European Economic Area. Primary storage is in Frankfurt (DE), backup/DR in Amsterdam (NL). This applies to databases, analytics pipelines, AI/ML training and inference, and log aggregation.

### Local AI (mirrors our thesis)
**Q:** Why did Meridian choose local AI over cloud APIs?
**A:** Four reasons: (1) Patient data must never leave the EU boundary, ruling out US-hosted AI APIs. (2) Deterministic inference required for clinical decision support (reproducible results). (3) Model versioning and rollback required for regulatory compliance. (4) Cost predictability -- no per-token billing surprises at 340-hospital scale. Implementation uses llama.cpp with GGUF models on Azure GPU VMs within the EU region.

### Technical Decisions
**Q:** Which database did Meridian choose and why?
**A:** PostgreSQL 16 (Azure Database for PostgreSQL Flexible Server). Rationale: open-source with no vendor lock-in, native JSONB support for varying hospital data formats, row-level security (RLS) for multi-tenant isolation between hospitals, logical replication for real-time analytics feeds, and pg_cron for scheduled data retention enforcement.

## Full Query Set

### Data Residency
**Q:** Where must patient data be stored?
**A:** Primary storage in Azure Germany Central (Frankfurt, DE). Backup and disaster recovery in Amsterdam, NL. Critical PII is restricted to Azure Germany Central only. Operational and analytics data can use any EU region.

**Q:** Why did they reject AWS?
**A:** AWS was strong technically but its EU Data Boundary commitment was less mature at evaluation time compared to Azure. GCP was rejected for limited EU regions for Healthcare workloads. On-premises was rejected due to the 340-hospital scale and maintenance burden.

### API Design
**Q:** What is the API versioning strategy?
**A:** URL-based versioning (/api/v1/, /api/v2/). Breaking changes require a new major version. Minimum 12-month deprecation window. v1 and v2 must run concurrently during migration periods. Version sunset dates published 6 months in advance.

**Q:** How does authentication work?
**A:** OAuth 2.0 + OpenID Connect (Azure AD B2C) for hospital staff. mTLS for system-to-system integrations. RBAC with 6 predefined roles: Admin, Physician, Nurse, Billing, Auditor, ReadOnly. Session tokens expire after 30 minutes of inactivity. All API calls logged to immutable audit trail.

### Security & Compliance
**Q:** What encryption standards does Meridian use?
**A:** At rest: AES-256 with customer-managed keys (Azure Key Vault, HSM-backed). In transit: TLS 1.3 minimum. Field-level: Patient SSN and diagnosis codes encrypted at the application layer before database storage. Key rotation: automatic every 90 days with zero-downtime rotation.

**Q:** Which compliance certifications does Meridian have?
**A:** HIPAA (Compliant, Coalfire), GDPR (Compliant, TUV Rheinland), ISO 27001 (Certified, BSI Group), SOC 2 Type II (In Progress, Deloitte), EU AI Act (Assessment Phase).

**Q:** What are the access control policies?
**A:** Principle of least privilege via Azure Policy. No standing admin access -- all privileged operations require just-in-time (JIT) approval. Network segmentation with patient data in isolated VNet with NSG rules. All database access via private endpoints only (no public IP). Quarterly access reviews with automatic deprovisioning.

### AI & ML
**Q:** What AI use cases are planned?
**A:** (1) Clinical decision support -- suggest differential diagnoses. (2) Appointment optimization -- predict no-shows. (3) Document classification -- categorize incoming faxes, referrals, lab reports. (4) Anomaly detection -- flag unusual billing patterns for fraud investigation.

**Q:** How are models governed under the EU AI Act?
**A:** All models registered in an internal model registry with lineage tracking. Training data provenance documented for every version. Bias audits required before clinical deployment. All clinical models classified as "high risk". Human-in-the-loop required for all clinical decision support outputs.

### Performance
**Q:** What is the latency SLA for patient lookups?
**A:** p99 < 50ms (API gateway to response). Other SLAs: appointment booking < 200ms, clinical search < 500ms, AI inference < 2 seconds, report generation < 30 seconds.

**Q:** What is the target uptime SLA?
**A:** 99.95% uptime (monthly). RPO: 5 minutes for critical PII. RTO: 30 minutes for full recovery. Multi-region active-passive failover from Frankfurt to Amsterdam. Monthly chaos engineering tests.

---

## OCR Demo — Snowflake Annual Report

> Upload `snowflake-fy2025-annual-report.pdf` first (or run `bash scripts/setup_ocr.sh`)

### The Keynote Beat

> "Our agent already answers questions about Nextera's internal data. Now watch
> what happens when we upload a real Snowflake annual report — the agent can
> immediately answer questions about how Nextera compares to industry benchmarks."

### Snowflake Demo Queries

**Q:** How many Snowflake customers spend more than $1M ARR?
**A:** 580 customers have trailing 12-month product revenue exceeding $1M.

**Q:** What is Snowflake's net revenue retention rate?
**A:** Net Revenue Retention (NRR) is 126%.

**Q:** What was Snowflake's total revenue in FY2025?
**A:** Total revenue was $3.63 billion, up 29% year-over-year.

**Q:** How does Snowflake's NRR compare to Nextera's?
**A:** Snowflake's NRR is 126%. (Note: requires both Snowflake PDF and Nextera KB indexed)

**Q:** What percentage of Snowflake's revenue comes from product vs services?
**A:** Product revenue was $3.46B (95% of total); professional services $174M (5%).

---

## OCR Demo — Nextera Quarterly Report

> Upload `nextera_quarterly_report.pdf` first (or run `python scripts/generate_ocr_demo_doc.py`)

### Nextera Table Extraction

**Q:** What was total revenue in Q4 2024?
**A:** Revenue in Q4 2024 was €103,200. (Source: Nextera Platform Q4 2024 Business Review)

**Q:** Which customer has the highest MRR?
**A:** BrightHealth GmbH has the highest MRR at €7,000/month on the Enterprise tier.

### Cross-Validation (OCR vs SQL — same numbers!)

1. Upload the quarterly report PDF → ask "What was Q3 2024 revenue?"
   → OCR-extracted answer: "€84,900" (from the PDF table)
2. Then ask "What were the total sales in Q3 2024?" (same question, SQL path)
   → SQL answer: "€84,900" (from the database)
3. Both answers match — proving OCR extraction accuracy

---

## Structured Extraction — Cross-Source Queries

> Upload `snowflake-fy2025-first50.pdf`, click "Extract structured data", then query.

### The Keynote Beat

> "We uploaded Snowflake's annual report. The agent extracted the key metrics —
> revenue, NRR, customer count — into structured data. Now watch: the agent joins
> our internal sales data with the extracted competitor data. One SQL query, two
> data sources, zero cloud."

### Extraction Demo Flow

1. Upload Snowflake PDF → 209 chunks indexed in ~1.6s
2. Click **"Extract structured data"** → 9 fields extracted in ~1.5s, stored in `competitors` table
3. Verify: raw JSON visible via "Raw LLM output" toggle
4. Now ask cross-source questions (switch to normal agent mode — clear the document chat badge)

### Cross-Source Demo Queries

**Q:** How does our revenue growth compare to Snowflake's?
**Expected:** Nextera 2024 growth ~21-29% vs Snowflake 30% YoY. Agent should query both `sales` and `competitors` tables.

**Q:** Which company has more customers?
**Expected:** Nextera has 10 customers, Snowflake has 745 total (580 >$1M). Agent should JOIN both tables.

**Q:** Compare our churn rate to Snowflake's net revenue retention.
**Expected:** Nextera churn 0.7% (Q4 2024) vs Snowflake NRR 126%. Different metrics but the agent surfaces both.

**Q:** What is Snowflake's free cash flow?
**Expected:** $884.1M — from the `competitors` table (extracted, not from the PDF chunks).

**Q:** List all competitors in the database.
**Expected:** SELECT * FROM competitors → shows Snowflake row with all extracted fields.

### API Endpoints

```bash
# Extract data from an uploaded document
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"document_id": "snowflake-fy2025-first50"}'

# List all extracted competitors
curl http://localhost:8000/competitors
```
