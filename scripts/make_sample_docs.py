"""Generate the binary demo documents (PDF annual report, DOCX loyalty guide).

Usage: python -m scripts.make_sample_docs
The generated files are committed, so this only needs to run if the text changes.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "documents"

ANNUAL_REPORT_PAGES = [
    (
        "Maple Ridge Outfitters - Annual Report Fiscal 2025",
        [
            "Message from the CEO",
            "Fiscal 2025 was a year of disciplined growth for Maple Ridge Outfitters. "
            "Net revenue reached CAD 48.2 million, an increase of 12% over fiscal 2024. "
            "Gross margin improved to 46.5%, up from 44.9% the previous year, driven by "
            "lower freight costs and a richer mix of private-label products.",
            "We opened two new stores in fiscal 2025: Kelowna, British Columbia in March "
            "and Halifax, Nova Scotia in October, bringing our network to eight stores.",
            "- Priya Natarajan, Chief Executive Officer",
        ],
    ),
    (
        "Digital and Customer Performance",
        [
            "E-commerce represented 31% of total revenue in fiscal 2025, compared with 27% "
            "in fiscal 2024. Mobile devices accounted for 64% of online sessions.",
            "The loyalty program grew to 182,000 active members. Gold members generated "
            "on average 2.8 times the annual spend of non-members.",
            "Net Promoter Score rose from 52 to 58. The product return rate was 7.9% of "
            "units sold, slightly better than the 8.4% recorded in fiscal 2024.",
        ],
    ),
    (
        "Strategic Priorities for Fiscal 2026",
        [
            "1. Sustainability: reduce Scope 1 and 2 carbon emissions by 18% by the end "
            "of fiscal 2027 against a fiscal 2023 baseline, and source 60% of apparel "
            "from suppliers rated A for sustainability.",
            "2. Inventory productivity: cut stock-outs of top-20 SKUs by one third through "
            "automated replenishment and weekly cycle counts.",
            "3. Omnichannel: raise the e-commerce share of revenue to 35% and offer "
            "two-hour in-store pickup at every location.",
            "4. Footprint: evaluate two additional store locations in Atlantic Canada.",
        ],
    ),
]

LOYALTY_SECTIONS = [
    ("Maple Ridge Rewards - Member Guide", None),
    ("Tiers", [
        "Bronze: every member starts at Bronze, with no minimum spend.",
        "Silver: reached after CAD 500 in qualifying spend within a calendar year.",
        "Gold: reached after CAD 1,500 in qualifying spend within a calendar year.",
    ]),
    ("Earning points", [
        "Members earn 1 point per dollar spent. Silver members earn 1.25 points per dollar "
        "and Gold members earn 2 points per dollar.",
        "Members receive a 200-point bonus during their birthday month.",
        "Every 1,000 points can be redeemed for a CAD 10 reward certificate.",
    ]),
    ("Gold benefits", [
        "Gold members receive free standard shipping on all online orders, an extended "
        "60-day return window, and early access to seasonal sales.",
    ]),
    ("Expiry and terms", [
        "Points expire 18 months after they are earned if the account has no qualifying "
        "purchase in that period. Tier status is reviewed every January based on the "
        "previous calendar year's spend.",
    ]),
]


def make_pdf(path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for title, paragraphs in ANNUAL_REPORT_PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.multi_cell(0, 9, title)
        pdf.ln(3)
        pdf.set_font("Helvetica", size=11)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(2)
    pdf.output(str(path))


def make_docx(path: Path) -> None:
    import docx

    document = docx.Document()
    for heading, paragraphs in LOYALTY_SECTIONS:
        document.add_heading(heading, level=1 if paragraphs is None else 2)
        for paragraph in paragraphs or []:
            document.add_paragraph(paragraph)
    table = document.add_table(rows=1, cols=3)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text, table.rows[0].cells[2].text = (
        "Tier", "Annual spend (CAD)", "Points per dollar")
    for tier, spend, rate in [("Bronze", "0", "1"), ("Silver", "500", "1.25"), ("Gold", "1,500", "2")]:
        cells = table.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = tier, spend, rate
    document.save(str(path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_pdf(OUT / "annual_report_2025.pdf")
    make_docx(OUT / "loyalty_program.docx")
    print(f"Wrote {OUT / 'annual_report_2025.pdf'} and {OUT / 'loyalty_program.docx'}")


if __name__ == "__main__":
    main()
