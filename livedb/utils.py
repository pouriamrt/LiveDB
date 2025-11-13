import asyncio
from typing import Tuple, Optional
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


async def save_text_as_pdf_async(
    text: str,
    output_path: str,
    *,
    page_size=letter,
    margins: Tuple[float, float, float, float] = (
        72,
        72,
        72,
        72,
    ),  # left, right, top, bottom (pt)
    font_name: str = "Helvetica",
    font_size: int = 12,
    line_spacing: float = 1.3,  # multiplier on font_size
    ttf_font_path: Optional[str] = None,  # e.g. "NotoSans-Regular.ttf" for full Unicode
):
    """
    Asynchronously save `text` as a paginated, wrapped PDF.
    - Uses asyncio.to_thread() so your event loop stays responsive.
    - If `ttf_font_path` is provided, registers it and uses as `font_name`.
    """

    def _sync_write():
        # Register optional TTF (for Unicode)
        if ttf_font_path:
            pdfmetrics.registerFont(TTFont(font_name, ttf_font_path))

        c = canvas.Canvas(output_path, pagesize=page_size)
        width, height = page_size
        left, right, top, bottom = margins
        avail_w = width - left - right
        y = height - top
        line_h = font_size * line_spacing

        c.setFont(font_name, font_size)

        def wrap_paragraph_by_width(paragraph: str):
            """Yield lines wrapped to `avail_w` using font metrics."""
            words = paragraph.split()
            if not words:
                yield ""  # blank line
                return
            line = words[0]
            for word in words[1:]:
                candidate = f"{line} {word}"
                if pdfmetrics.stringWidth(candidate, font_name, font_size) <= avail_w:
                    line = candidate
                else:
                    yield line
                    line = word
            yield line

        for para in text.split("\n"):
            for line in wrap_paragraph_by_width(para):
                if y < bottom + line_h:  # new page
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = height - top
                c.drawString(left, y, line)
                y -= line_h
            # paragraph spacing
            y -= line_h * 0.3

        c.save()

    # Run blocking PDF generation in a worker thread
    await asyncio.to_thread(_sync_write)
