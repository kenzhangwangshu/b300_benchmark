"""Render blog tables as PNG images styled to match the HTML mock-ups.

Visual contract (mirrors the HTML spec):
- Header row: dark navy (#1a1a2e) background, light text (#e0e0e0), bold.
- Body cells: white by default. Optional per-row tint (e.g. green/red).
- Accent text colors: positive #4CAF50, negative #F44336.
- Subtle grey cell borders (#ddd).
- Title + subtitle stacked above the table; muted footnote below.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HEADER_BG    = "#1a1a2e"
HEADER_TEXT  = "#e0e0e0"
HEADER_RULE  = "#4a4a6a"
BODY_BG      = "#ffffff"
BODY_TEXT    = "#1f2937"
ROW_RULE     = "#dddddd"
SUBTITLE     = "#6b7280"
FOOTNOTE     = "#8a8a8a"
ACCENT_GREEN = "#4CAF50"
ACCENT_RED   = "#F44336"
ROW_GREEN_BG = "#1a2a1a"
ROW_RED_BG   = "#2a1a1a"
ROW_GREEN_BG_LIGHT = "#EAF5EA"
ROW_RED_BG_LIGHT   = "#FBEAEA"
MUTED_ROW    = "#888888"


def _setup_fonts():
    mpl.rcParams.update({
        "font.family": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def render_table(
    out_path: str | Path,
    title: str,
    columns,                 # list[dict]: {"label", "align" ("left"|"center"|"right"), "width" (rel)}
    rows,                    # list[dict]: {"cells": list[dict|str], "bg": optional}
    subtitle: str | None = None,
    footnote: str | None = None,
    width_in: float = 11.0,
    row_height_in: float = 0.42,
    header_height_in: float = 0.50,
    title_block_in: float = 0.95,
    footnote_in: float = 0.45,
    side_margin_in: float = 0.25,
):
    """Render one table to ``out_path`` as a PNG.

    A *cell* is either a plain string or a dict with keys:
      text, color, weight ("bold"|"normal"), bg, align (override).
    A *row* may carry ``bg`` (full-row tint) and ``text_color`` (default
    color override for all cells in the row).
    """
    _setup_fonts()

    n_cols = len(columns)
    raw_widths = [c.get("width", 1.0) for c in columns]
    total_w = sum(raw_widths)
    avail = width_in - 2 * side_margin_in
    col_widths = [w / total_w * avail for w in raw_widths]
    col_xs = []
    x = side_margin_in
    for w in col_widths:
        col_xs.append(x)
        x += w

    n_rows = len(rows)
    table_height = header_height_in + n_rows * row_height_in
    height_in = (title_block_in + table_height +
                 (footnote_in if footnote else 0.18) + 0.20)

    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.axis("off")

    # ---- Title block --------------------------------------------------
    y_cursor = height_in - 0.30
    ax.text(side_margin_in, y_cursor, title,
            fontsize=15, fontweight="bold", color="#111", ha="left", va="top")
    y_cursor -= 0.42
    if subtitle:
        ax.text(side_margin_in, y_cursor, subtitle,
                fontsize=10, color=SUBTITLE, ha="left", va="top")

    # ---- Header row ---------------------------------------------------
    y_top = height_in - title_block_in
    y_header_bottom = y_top - header_height_in
    ax.add_patch(mpatches.Rectangle(
        (side_margin_in, y_header_bottom), avail, header_height_in,
        facecolor=HEADER_BG, edgecolor="none"))
    for col, x0, w in zip(columns, col_xs, col_widths):
        align = col.get("align", "left")
        if align == "center":
            tx, ha = x0 + w / 2, "center"
        elif align == "right":
            tx, ha = x0 + w - 0.10, "right"
        else:
            tx, ha = x0 + 0.10, "left"
        ax.text(tx, y_header_bottom + header_height_in / 2,
                col["label"], fontsize=10, fontweight="bold",
                color=HEADER_TEXT, ha=ha, va="center")
    ax.plot([side_margin_in, side_margin_in + avail],
            [y_header_bottom, y_header_bottom],
            color=HEADER_RULE, linewidth=1.6)

    # ---- Body rows ----------------------------------------------------
    y_cur = y_header_bottom
    for row in rows:
        row_bg = row.get("bg", BODY_BG)
        ax.add_patch(mpatches.Rectangle(
            (side_margin_in, y_cur - row_height_in), avail, row_height_in,
            facecolor=row_bg, edgecolor="none"))

        default_color = row.get("text_color", BODY_TEXT)
        for col, x0, w, cell in zip(columns, col_xs, col_widths, row["cells"]):
            if isinstance(cell, str):
                cell = {"text": cell}
            if "bg" in cell:
                ax.add_patch(mpatches.Rectangle(
                    (x0, y_cur - row_height_in), w, row_height_in,
                    facecolor=cell["bg"], edgecolor="none"))
            text   = cell.get("text", "")
            color  = cell.get("color", default_color)
            weight = cell.get("weight", "normal")
            align  = cell.get("align", col.get("align", "left"))
            if align == "center":
                tx, ha = x0 + w / 2, "center"
            elif align == "right":
                tx, ha = x0 + w - 0.10, "right"
            else:
                tx, ha = x0 + 0.10, "left"
            ax.text(tx, y_cur - row_height_in / 2,
                    text, fontsize=10, color=color, fontweight=weight,
                    ha=ha, va="center")

        # row bottom rule
        ax.plot([side_margin_in, side_margin_in + avail],
                [y_cur - row_height_in, y_cur - row_height_in],
                color=ROW_RULE, linewidth=0.6)
        y_cur -= row_height_in

    # ---- Footnote -----------------------------------------------------
    if footnote:
        ax.text(side_margin_in, y_cur - 0.20, footnote,
                fontsize=8.5, color=FOOTNOTE, ha="left", va="top",
                wrap=True)

    fig.savefig(out_path, dpi=150, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out_path}")
