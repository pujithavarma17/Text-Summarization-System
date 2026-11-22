"""Download formats: a readable TXT report and an Excel workbook (Overview + Summaries sheets)."""

import io
import textwrap
from datetime import datetime

from summarizer.preprocessing import tokenize_sentences

WIDTH = 90  # characters per line in the TXT report


def summary_points(summary):
    """The summary split into sentences (one bullet point each)."""
    return tokenize_sentences(summary) or [summary]


def reduction(result):
    return 100 - round(100 * result["summary_words"] / max(result["original_words"], 1))


def format_summary(summary, style="Paragraph", width=WIDTH):
    """Summary text as wrapped bullet points or a wrapped paragraph."""
    if style == "Bullet Points":
        return "\n".join(
            textwrap.fill(point, width, initial_indent="  • ", subsequent_indent="    ")
            for point in summary_points(summary)
        )
    return textwrap.fill(summary, width, initial_indent="  ", subsequent_indent="  ")


def to_txt(results, style="Paragraph"):
    """All summaries as one neatly formatted text report."""
    done = sum("summary" in r for r in results)
    lines = [
        "TEXT SUMMARIZATION REPORT",
        f"Created: {datetime.now():%d %b %Y, %H:%M}   ·   "
        f"Documents: {done} of {len(results)} summarized   ·   Style: {style}",
        "=" * WIDTH,
    ]

    for number, r in enumerate(results, start=1):
        lines += ["", f"{number}. {r['name']}", "-" * WIDTH]

        if "error" in r:
            lines.append(f"  NOT SUMMARIZED: {r['error']}")
            continue

        lines += [
            f"  Method: {r.get('method', '')}   ·   Words: {r['original_words']:,} -> "
            f"{r['summary_words']:,}   ·   {reduction(r)}% shorter",
            "",
            "  SUMMARY",
            format_summary(r["summary"], style),
            "",
            "  KEY POINTS",
        ]
        for i, point in enumerate(r["key_points"], start=1):
            lines.append(textwrap.fill(point, WIDTH, initial_indent=f"  {i}. ",
                                       subsequent_indent="     "))
        lines += ["", "  IMPORTANT TOPICS", "  " + ", ".join(r["topics"])]

    lines += ["", "=" * WIDTH, "End of report"]
    return "\n".join(lines) + "\n"


def to_excel(results):
    """All summaries as an Excel workbook (bytes), ready to download."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="6D4CF5")  # the app's purple
    wrap = Alignment(wrap_text=True, vertical="top")

    def add_sheet(sheet, headers, widths, rows):
        sheet.append(headers)
        for cell in sheet[1]:
            cell.font, cell.fill = header_font, header_fill
        for row in rows:
            sheet.append(row)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = wrap
        for column, width in zip("ABCDEFGHIJ", widths):
            sheet.column_dimensions[column].width = width
        sheet.freeze_panes = "A2"                 # header stays visible when scrolling
        sheet.auto_filter.ref = sheet.dimensions  # filter / sort buttons on the header

    workbook = Workbook()

    # Sheet 1: one row per file
    overview_rows = []
    for number, r in enumerate(results, start=1):
        if "error" in r:
            overview_rows.append([number, r["name"], "", "", "", "", "", "", f"Not summarized: {r['error']}"])
        else:
            overview_rows.append([
                number, r["name"], r.get("method", ""), r["original_words"], r["summary_words"],
                f"{reduction(r)}%",
                "\n".join(f"{i}. {p}" for i, p in enumerate(r["key_points"], start=1)),
                ", ".join(r["topics"]),
                "Done",
            ])
    add_sheet(
        workbook.active,
        ["#", "File", "Method", "Original words", "Summary words", "Reduced",
         "Key points", "Important topics", "Status"],
        [5, 30, 12, 15, 16, 10, 70, 35, 30],
        overview_rows,
    )
    workbook.active.title = "Overview"

    # Sheet 2: one row per summary point
    summary_rows = [
        [r["name"], i, point]
        for r in results if "summary" in r
        for i, point in enumerate(summary_points(r["summary"]), start=1)
    ]
    add_sheet(workbook.create_sheet("Summaries"),
              ["File", "Point #", "Summary point"], [30, 9, 110], summary_rows)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
