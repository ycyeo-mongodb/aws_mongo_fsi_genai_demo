# Sample documents — Onboarding Auto-Map demo

Drop any of these into the **Onboarding Auto-Map** uploader on the **Graph Network** tab.
Bedrock Claude (multimodal) will OCR the document, identify the document type, extract entities,
and the backend will match each entity against the live `network_graph` collection.

## National ID cards (Cambodian NRIC)

- `01_NRIC_sok_pisey_HIGH_RISK.pdf` — 🚨 HIGH RISK — address matches Phantom Lane synthetic-identity ring (3 customers already)
- `02_NRIC_bunly_sopheap_MEDIUM_RISK.pdf` — ⚠ MEDIUM RISK — address matches Sihanoukville mule cluster
- `03_NRIC_chan_dara_CLEAN.pdf` — ✓ CLEAN — no fraud-ring matches; brand-new isolated customer

## Passports

- `04_PASSPORT_visal_chann_CLEAN.pdf` — ✓ CLEAN — standard passport, no graph matches

## Insurance Policies

- `05_POLICY_ghost_beneficiary_HIGH_RISK.pdf` — 🚨 HIGH RISK — beneficiary Vann Vanna (CUST-00007) is already named on 6 policies → ghost beneficiary
- `06_POLICY_normal_family_CLEAN.pdf` — ✓ CLEAN — straightforward family policy, no fraud-ring matches

Use **↺ Reset auto-extracted** in the UI to remove the inserted nodes between runs.

_Regenerate this folder any time with `python generate_sample_documents.py`._
