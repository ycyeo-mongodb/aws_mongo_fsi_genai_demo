#!/usr/bin/env python3
"""
Generate realistic-looking sample documents for the Onboarding Auto-Map demo.

Produces three document categories — National ID cards, a Passport data page,
and Insurance Policy contracts — each designed to trigger a different path
through /api/graph/extract-pdf:

  data/sample_documents/
    01_NRIC_sok_pisey_HIGH_RISK.pdf            → matches Phantom-Lane fraud ring
    02_NRIC_bunly_sopheap_MEDIUM_RISK.pdf      → matches Sihanoukville mule cluster
    03_NRIC_chan_dara_CLEAN.pdf                → no matches, clean onboarding
    04_PASSPORT_visal_chann_CLEAN.pdf          → standard passport flow
    05_POLICY_ghost_beneficiary_HIGH_RISK.pdf  → triggers ghost-beneficiary alert (Vann Vanna)
    06_POLICY_normal_family_CLEAN.pdf          → normal family policy, no alerts

Run:
    python generate_sample_documents.py
"""

from __future__ import annotations
from pathlib import Path
import random
import string

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "data" / "sample_documents"


# ──────────────────────────────────────────────────────────
# Sample data — designed to trigger specific graph patterns
# ──────────────────────────────────────────────────────────

NRIC_SAMPLES = [
    {
        "filename": "01_NRIC_sok_pisey_HIGH_RISK.pdf",
        "name_en": "SOK PISEY",
        "name_km": "សុខ ពិសី",
        "id_number": "200312-7894-561",
        "dob": "12 MAR 1985",
        "sex": "M",
        "nationality": "KHMER",
        "place_of_birth": "PHNOM PENH",
        "address": "42 Riverside Phantom Lane, Phnom Penh",
        "issued": "15 JUN 2022",
        "expires": "15 JUN 2032",
        "outcome": "🚨 HIGH RISK — address matches Phantom Lane synthetic-identity ring (3 customers already)",
    },
    {
        "filename": "02_NRIC_bunly_sopheap_MEDIUM_RISK.pdf",
        "name_en": "BUNLY SOPHEAP",
        "name_km": "ប៊ុនលី សុភាព",
        "id_number": "190604-4456-721",
        "dob": "04 JUN 1992",
        "sex": "M",
        "nationality": "KHMER",
        "place_of_birth": "SIHANOUKVILLE",
        "address": "108B Industrial Park Block C, Sihanoukville",
        "issued": "22 NOV 2023",
        "expires": "22 NOV 2033",
        "outcome": "⚠ MEDIUM RISK — address matches Sihanoukville mule cluster",
    },
    {
        "filename": "03_NRIC_chan_dara_CLEAN.pdf",
        "name_en": "CHAN DARA",
        "name_km": "ច័ន្ទ ដារា",
        "id_number": "180820-2345-678",
        "dob": "20 AUG 1990",
        "sex": "F",
        "nationality": "KHMER",
        "place_of_birth": "PHNOM PENH",
        "address": "88 Diamond Island Tower B, Phnom Penh",
        "issued": "10 JAN 2024",
        "expires": "10 JAN 2034",
        "outcome": "✓ CLEAN — no fraud-ring matches; brand-new isolated customer",
    },
]

PASSPORT_SAMPLES = [
    {
        "filename": "04_PASSPORT_visal_chann_CLEAN.pdf",
        "type_": "P",
        "country": "KHM",
        "passport_no": "N0481725",
        "surname": "CHANN",
        "given_names": "VISAL RATANAK",
        "nationality": "KHMER",
        "dob": "07 OCT 1988",
        "sex": "M",
        "place_of_birth": "BATTAMBANG",
        "issued": "12 APR 2024",
        "expires": "12 APR 2034",
        "authority": "MINISTRY OF FOREIGN AFFAIRS",
        "address": "212 Street 240, Daun Penh, Phnom Penh",
        "outcome": "✓ CLEAN — standard passport, no graph matches",
    },
]

POLICY_SAMPLES = [
    {
        "filename": "05_POLICY_ghost_beneficiary_HIGH_RISK.pdf",
        "policy_id": "POL-AL-2026-04-7745",
        "policy_type": "life",
        "underwriter": "Atlas Life Assurance",
        "issue_date": "2026-04-15",
        "expiry_date": "2036-04-15",
        "coverage_usd": 220000,
        "premium_monthly_usd": 128.00,
        "holder": {"name": "Tep Channary", "address": "12 Sokha Road, Phnom Penh",
                   "id_number": "180920-1145-339", "dob": "1986-09-20"},
        "beneficiaries": [
            {"name": "Vann Vanna", "relationship": "business associate", "share_pct": 100},
        ],
        "outcome": "🚨 HIGH RISK — beneficiary Vann Vanna (CUST-00007) is already named on 6 policies → ghost beneficiary",
    },
    {
        "filename": "06_POLICY_normal_family_CLEAN.pdf",
        "policy_id": "POL-AH-2026-04-9012",
        "policy_type": "health",
        "underwriter": "Atlas Health Mutual",
        "issue_date": "2026-04-20",
        "expiry_date": "2031-04-20",
        "coverage_usd": 50000,
        "premium_monthly_usd": 38.00,
        "holder": {"name": "Pich Sreyleak", "address": "55 Russian Boulevard, Phnom Penh",
                   "id_number": "190512-2298-115", "dob": "1991-05-12"},
        "beneficiaries": [
            {"name": "Pich Boran", "relationship": "spouse", "share_pct": 50},
            {"name": "Pich Theara", "relationship": "child", "share_pct": 50},
        ],
        "outcome": "✓ CLEAN — straightforward family policy, no fraud-ring matches",
    },
]


# ──────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────

KH_BLUE = colors.HexColor("#0d2c5c")
KH_RED  = colors.HexColor("#c0142a")
ATLAS   = colors.HexColor("#1a237e")
ACCENT  = colors.HexColor("#00897b")


def _draw_id_photo(c: canvas.Canvas, x: float, y: float, w: float, h: float, initials: str) -> None:
    """A simple gray placeholder block for the photo, with initials overlay."""
    c.setFillColor(colors.HexColor("#cfd8dc"))
    c.rect(x, y, w, h, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#90a4ae"))
    c.setFont("Helvetica-Bold", 22)
    text_width = c.stringWidth(initials, "Helvetica-Bold", 22)
    c.drawString(x + (w - text_width) / 2, y + h / 2 - 8, initials)
    c.setStrokeColor(colors.HexColor("#546e7a"))
    c.rect(x, y, w, h, fill=0, stroke=1)


def _initials(name: str) -> str:
    parts = name.replace(",", "").split()
    return "".join(p[0] for p in parts[:2]).upper() if parts else "?"


# ──────────────────────────────────────────────────────────
# 1) National ID card  (landscape A4 trimmed look)
# ──────────────────────────────────────────────────────────

def build_nric(out_path: Path, sample: dict) -> None:
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

    # ─── Frame ───
    margin = 1.5 * cm
    card_w = page_w - 2 * margin
    card_h = page_h - 2 * margin

    # Outer border with rounded corners
    c.setFillColor(colors.HexColor("#eef4ff"))
    c.roundRect(margin, margin, card_w, card_h, 10, fill=1, stroke=0)
    c.setStrokeColor(KH_BLUE)
    c.setLineWidth(2)
    c.roundRect(margin, margin, card_w, card_h, 10, fill=0, stroke=1)

    # ─── Header band ───
    c.setFillColor(KH_BLUE)
    c.rect(margin, page_h - margin - 1.6 * cm, card_w, 1.6 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, page_h - margin - 0.6 * cm, "ព្រះរាជាណាចក្រកម្ពុជា")
    c.setFont("Helvetica", 9)
    c.drawCentredString(page_w / 2, page_h - margin - 1.05 * cm, "KINGDOM OF CAMBODIA  ·  ជាតិ សាសនា ព្រះមហាក្សត្រ")
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#ffd700"))
    c.drawCentredString(page_w / 2, page_h - margin - 1.45 * cm, "NATIONAL IDENTITY CARD  ·  អត្តសញ្ញាណប័ណ្ណសញ្ជាតិខ្មែរ")

    # ─── Photo area (left) ───
    photo_x = margin + 0.8 * cm
    photo_y = margin + 0.8 * cm
    photo_w = 4.3 * cm
    photo_h = 5.5 * cm
    _draw_id_photo(c, photo_x, photo_y + 1.2 * cm, photo_w, photo_h, _initials(sample["name_en"]))

    # Below photo: signature placeholder
    c.setStrokeColor(colors.HexColor("#546e7a"))
    c.setLineWidth(0.5)
    c.line(photo_x, photo_y + 0.8 * cm, photo_x + photo_w, photo_y + 0.8 * cm)
    c.setFillColor(colors.HexColor("#546e7a"))
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(photo_x, photo_y + 0.4 * cm, "Signature ហត្ថលេខា")

    # ─── Fields (right) ───
    fx = photo_x + photo_w + 1.0 * cm
    fy = page_h - margin - 2.4 * cm
    line_h = 0.85 * cm

    def field(label_en, label_km, value, y_offset, value_size=11, value_bold=True):
        c.setFillColor(colors.HexColor("#37474f"))
        c.setFont("Helvetica", 7)
        c.drawString(fx, y_offset + 0.32 * cm, f"{label_en}  ·  {label_km}")
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold" if value_bold else "Helvetica", value_size)
        c.drawString(fx, y_offset, value)
        c.setStrokeColor(colors.HexColor("#cfd8dc"))
        c.setLineWidth(0.3)
        c.line(fx, y_offset - 0.1 * cm, fx + 12 * cm, y_offset - 0.1 * cm)

    field("Name (Latin)", "ឈ្មោះ (ឡាតាំង)", sample["name_en"], fy)
    field("Name (Khmer)", "ឈ្មោះ (ខ្មែរ)", sample["name_km"], fy - line_h)
    field("Identity Number", "លេខអត្តសញ្ញាណប័ណ្ណ", sample["id_number"], fy - 2 * line_h)
    field("Date of Birth", "ថ្ងៃខែឆ្នាំកំណើត", f"{sample['dob']}     Sex / ភេទ: {sample['sex']}", fy - 3 * line_h, value_size=10)
    field("Place of Birth", "ទីកន្លែងកំណើត", sample["place_of_birth"], fy - 4 * line_h, value_size=10)
    field("Address", "អាសយដ្ឋាន", sample["address"], fy - 5 * line_h, value_size=10)
    field("Nationality", "សញ្ជាតិ", sample["nationality"], fy - 6 * line_h, value_size=10)

    # Issue / expiry
    iy = fy - 7 * line_h
    c.setFillColor(colors.HexColor("#37474f"))
    c.setFont("Helvetica", 7)
    c.drawString(fx, iy + 0.32 * cm, "Issued / ចេញ        Expires / ផុតកំណត់")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(fx, iy, sample["issued"] + "       " + sample["expires"])

    # ─── Footer hologram strip ───
    c.setFillColor(KH_RED)
    c.rect(margin, margin, card_w, 0.5 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(page_w / 2, margin + 0.18 * cm, "<<<KHM" + sample["id_number"].replace("-", "") + "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    c.setTitle(f"NRIC — {sample['name_en']}")
    c.save()


# ──────────────────────────────────────────────────────────
# 2) Passport data page  (portrait, two-column)
# ──────────────────────────────────────────────────────────

def build_passport(out_path: Path, sample: dict) -> None:
    page_w, page_h = A4
    c = canvas.Canvas(str(out_path), pagesize=A4)

    # Outer border
    c.setStrokeColor(KH_BLUE)
    c.setLineWidth(1.5)
    c.rect(1.5 * cm, 1.5 * cm, page_w - 3 * cm, page_h - 3 * cm, stroke=1, fill=0)

    # Header
    c.setFillColor(KH_BLUE)
    c.rect(1.5 * cm, page_h - 4 * cm, page_w - 3 * cm, 2.5 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(page_w / 2, page_h - 2.2 * cm, "KINGDOM OF CAMBODIA")
    c.setFont("Helvetica", 11)
    c.drawCentredString(page_w / 2, page_h - 2.7 * cm, "ព្រះរាជាណាចក្រកម្ពុជា")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#ffd700"))
    c.drawCentredString(page_w / 2, page_h - 3.4 * cm, "PASSPORT  ·  លិខិតឆ្លងដែន")

    # Photo (left)
    photo_x = 2.5 * cm
    photo_y = page_h - 11 * cm
    _draw_id_photo(c, photo_x, photo_y, 4 * cm, 5.5 * cm, _initials(sample["surname"] + " " + sample["given_names"]))

    # Right column fields
    fx = photo_x + 4 * cm + 1 * cm
    fy = page_h - 5.5 * cm
    line_h = 0.8 * cm

    def line(label, value, y_offset, value_size=11, bold=True):
        c.setFillColor(colors.HexColor("#37474f"))
        c.setFont("Helvetica", 7.5)
        c.drawString(fx, y_offset + 0.28 * cm, label)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", value_size)
        c.drawString(fx, y_offset, value)

    line("Type / ប្រភេទ", sample["type_"], fy)
    line("Country code / កូដប្រទេស", sample["country"], fy - line_h)
    line("Passport No. / លេខលិខិតឆ្លងដែន", sample["passport_no"], fy - 2 * line_h)
    line("Surname / នាមត្រកូល", sample["surname"], fy - 3 * line_h)
    line("Given names / នាមខ្លួន", sample["given_names"], fy - 4 * line_h)
    line("Nationality / សញ្ជាតិ", sample["nationality"], fy - 5 * line_h, value_size=10)
    line("Date of birth / ថ្ងៃខែឆ្នាំកំណើត", sample["dob"] + "     Sex / ភេទ: " + sample["sex"], fy - 6 * line_h, value_size=10, bold=False)

    # Bottom block (full width)
    by = photo_y - 1.8 * cm
    line("Place of birth / ទីកន្លែងកំណើត", sample["place_of_birth"], by, value_size=10)
    line("Address / អាសយដ្ឋាន", sample["address"], by - line_h, value_size=10, bold=False)
    line("Date of issue / ថ្ងៃចេញ", sample["issued"], by - 2 * line_h, value_size=10, bold=False)
    line("Date of expiry / ថ្ងៃផុតកំណត់", sample["expires"], by - 3 * line_h, value_size=10, bold=False)
    line("Authority / អាជ្ញាធរ", sample["authority"], by - 4 * line_h, value_size=10, bold=False)

    # MRZ at bottom (machine-readable zone)
    mrz_y = 2.3 * cm
    surname_norm = sample["surname"].upper().replace(" ", "<")
    given_norm = sample["given_names"].upper().replace(" ", "<")
    line1 = f"P<KHM{surname_norm}<<{given_norm}"
    line1 = (line1 + "<" * 44)[:44]
    digits = "".join(ch for ch in sample["passport_no"] if ch.isalnum())
    line2 = f"{digits}<KHM{sample['dob'].replace(' ','').replace('-','')}{sample['sex']}{sample['expires'].replace(' ','').replace('-','')}<<<<<<<<<<<<<<<<<<<"
    line2 = (line2 + "<" * 44)[:44]
    c.setFont("Courier-Bold", 11)
    c.setFillColor(colors.black)
    c.drawString(2.3 * cm, mrz_y + 0.5 * cm, line1)
    c.drawString(2.3 * cm, mrz_y, line2)

    c.setTitle(f"Passport — {sample['surname']} {sample['given_names']}")
    c.save()


# ──────────────────────────────────────────────────────────
# 3) Insurance Policy contract  (portrait, multi-page possible)
# ──────────────────────────────────────────────────────────

def build_policy(out_path: Path, sample: dict) -> None:
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"Policy — {sample['policy_id']}",
    )
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=20,
                        textColor=ACCENT, spaceAfter=2, alignment=0)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                        textColor=ATLAS, spaceBefore=14, spaceAfter=8, alignment=0)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=11,
                           textColor=colors.grey)
    label_style = ParagraphStyle("lbl", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.HexColor("#546e7a"))

    story = []

    # Header band
    header_tbl = Table([[
        Paragraph(f"<b>{sample['underwriter'].upper()}</b><br/><font size=8 color='#546e7a'>Insurance Policy Contract</font>", body),
        Paragraph(f"<font size=10>Policy Number</font><br/><b><font size=14 color='#1a237e'>{sample['policy_id']}</font></b>", body),
    ]], colWidths=[10 * cm, 7 * cm])
    header_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, ACCENT),
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e0f2f1")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#fafafa")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # Policy summary — type, coverage, premium, dates
    story.append(Paragraph("Policy Summary", h2))
    summary = [
        ["Policy Type", sample["policy_type"].upper()],
        ["Sum Assured", f"USD {sample['coverage_usd']:,.0f}"],
        ["Premium", f"USD {sample['premium_monthly_usd']:,.2f} / month"],
        ["Issue Date", sample["issue_date"]],
        ["Expiry Date", sample["expiry_date"]],
    ]
    t = Table(summary, colWidths=[5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
    ]))
    story.append(t)

    # Policy holder
    story.append(Paragraph("Policy Holder", h2))
    h = sample["holder"]
    holder_data = [
        ["Full Name", h["name"]],
        ["National ID", h.get("id_number", "—")],
        ["Date of Birth", h.get("dob", "—")],
        ["Address on File", h.get("address", "—")],
    ]
    t = Table(holder_data, colWidths=[5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
    ]))
    story.append(t)

    # Beneficiaries
    story.append(Paragraph("Beneficiaries", h2))
    benef_rows = [["Name", "Relationship", "Share %"]]
    for b in sample["beneficiaries"]:
        benef_rows.append([b["name"], b.get("relationship", "—"), f"{b.get('share_pct', 100)}%"])
    t = Table(benef_rows, colWidths=[8 * cm, 6 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ATLAS),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
    ]))
    story.append(t)

    # Terms
    story.append(Paragraph("Terms &amp; Conditions (Excerpt)", h2))
    story.append(Paragraph(
        "1. The policy holder agrees to pay the monthly premium stated above in exchange for the sum "
        "assured payable to the named beneficiaries upon a covered event. <br/>"
        "2. Beneficiaries must present the original policy contract and a certified copy of the holder's "
        "identification at the time of claim. <br/>"
        "3. Atlas is entitled to refuse claims involving misrepresentation or undisclosed material facts. <br/>"
        "4. Cancellation requires 30 days' written notice. <br/>"
        "5. This contract is governed by the laws of the Kingdom of Cambodia.",
        body,
    ))

    # Signature
    story.append(Spacer(1, 1.5 * cm))
    sig = Table([
        [Paragraph("____________________________<br/><font size=8>Policy Holder Signature</font>", body),
         Paragraph("____________________________<br/><font size=8>Atlas Authorised Officer</font>", body)],
    ], colWidths=[8.5 * cm, 8.5 * cm])
    story.append(sig)

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "This policy contract is generated for demonstration purposes only. Atlas Life Assurance and "
        "Atlas Health Mutual are fictitious entities.",
        small,
    ))

    doc.build(story)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating sample documents → {OUT_DIR}\n")

    print("National ID cards:")
    for s in NRIC_SAMPLES:
        out = OUT_DIR / s["filename"]
        build_nric(out, s)
        print(f"  ✓ {s['filename']}")
        print(f"      {s['outcome']}")

    print("\nPassports:")
    for s in PASSPORT_SAMPLES:
        out = OUT_DIR / s["filename"]
        build_passport(out, s)
        print(f"  ✓ {s['filename']}")
        print(f"      {s['outcome']}")

    print("\nInsurance Policies:")
    for s in POLICY_SAMPLES:
        out = OUT_DIR / s["filename"]
        build_policy(out, s)
        print(f"  ✓ {s['filename']}")
        print(f"      {s['outcome']}")

    # README
    readme = OUT_DIR / "README.md"
    readme.write_text("\n".join([
        "# Sample documents — Onboarding Auto-Map demo",
        "",
        "Drop any of these into the **Onboarding Auto-Map** uploader on the **Graph Network** tab.",
        "Bedrock Claude (multimodal) will OCR the document, identify the document type, extract entities,",
        "and the backend will match each entity against the live `network_graph` collection.",
        "",
        "## National ID cards (Cambodian NRIC)",
        "",
        *[f"- `{s['filename']}` — {s['outcome']}" for s in NRIC_SAMPLES],
        "",
        "## Passports",
        "",
        *[f"- `{s['filename']}` — {s['outcome']}" for s in PASSPORT_SAMPLES],
        "",
        "## Insurance Policies",
        "",
        *[f"- `{s['filename']}` — {s['outcome']}" for s in POLICY_SAMPLES],
        "",
        "Use **↺ Reset auto-extracted** in the UI to remove the inserted nodes between runs.",
        "",
        "_Regenerate this folder any time with `python generate_sample_documents.py`._",
    ]) + "\n", encoding="utf-8")
    print(f"\n  ✓ README.md")
    print(f"\nDone. Folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
