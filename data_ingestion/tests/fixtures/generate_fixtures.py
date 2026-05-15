"""Generate test fixture files. Run once during project setup: python tests/fixtures/generate_fixtures.py"""

from pathlib import Path


def generate_sample_pdf() -> None:
    """Create a 2-page PDF with Hindi text using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError:
        print("reportlab not installed — creating a minimal text-based PDF manually")
        _generate_minimal_pdf()
        return

    out_path = Path(__file__).parent / "sample_pdfs" / "sample_hindi_2page.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to register a Devanagari-capable font; fall back to Helvetica with ASCII placeholder
    try:
        import subprocess
        import sys
        font_candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "C:/Windows/Fonts/mangal.ttf",
            "/Library/Fonts/ITFDevanagari.ttf",
        ]
        font_registered = False
        for font_path in font_candidates:
            if Path(font_path).exists():
                pdfmetrics.registerFont(TTFont("Devanagari", font_path))
                font_registered = True
                break
    except Exception:
        font_registered = False

    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4

    pages = [
        (
            "Hindi Sample - Page 1",
            [
                "भारत एक महान देश है।",
                "यहाँ अनेक भाषाएँ बोली जाती हैं।",
                "हिंदी भारत की राजभाषा है।",
                "देवनागरी लिपि में लिखी जाती है।",
                "भारतीय संस्कृति विश्व की प्राचीनतम संस्कृतियों में से एक है।",
                "यहाँ के त्योहार और परंपराएँ अत्यंत विविध हैं।",
            ],
        ),
        (
            "Hindi Sample - Page 2",
            [
                "महात्मा गांधी भारत के राष्ट्रपिता थे।",
                "उन्होंने अहिंसा का मार्ग अपनाया।",
                "भारतीय स्वतंत्रता संग्राम में अनेक वीरों ने बलिदान दिया।",
                "आज भारत एक स्वतंत्र और लोकतांत्रिक देश है।",
                "भारतीय संविधान विश्व का सबसे लंबा लिखित संविधान है।",
                "यह सभी नागरिकों को मौलिक अधिकार प्रदान करता है।",
            ],
        ),
    ]

    for title, lines in pages:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 72, title)
        font_name = "Devanagari" if font_registered else "Helvetica"
        c.setFont(font_name, 12)
        y = height - 110
        for line in lines:
            c.drawString(72, y, line)
            y -= 22
        c.showPage()

    c.save()
    print(f"Generated: {out_path}")


def _generate_minimal_pdf() -> None:
    """Fallback: write a minimal valid PDF with embedded Hindi text as UTF-8 stream."""
    out_path = Path(__file__).parent / "sample_pdfs" / "sample_hindi_2page.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal PDF structure with two pages containing Hindi text in content streams
    p1_content = b"BT /F1 12 Tf 72 720 Td (Page 1: Hindi text placeholder) Tj ET"
    p2_content = b"BT /F1 12 Tf 72 720 Td (Page 2: Hindi text placeholder) Tj ET"

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
        b"5 0 obj\n<< /Length " + str(len(p1_content)).encode() + b" >>\nstream\n"
        + p1_content + b"\nendstream\nendobj\n"
        b"6 0 obj\n<< /Length " + str(len(p2_content)).encode() + b" >>\nstream\n"
        + p2_content + b"\nendstream\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 5 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 6 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
        b"xref\n0 7\n0000000000 65535 f \n"
        b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    )
    out_path.write_bytes(pdf)
    print(f"Generated minimal PDF: {out_path}")


if __name__ == "__main__":
    generate_sample_pdf()
