"""Build contact_sheet.png: every project's figure.png in a labelled grid.

Usage (from the repository root):
    .venv/bin/python tools/make_contact_sheet.py [--out contact_sheet.png]

Reads projects/NN-slug/figure.png only; never modifies a project. Each figure
is scaled to fit a fixed 16:9 cell (aspect preserved, letterboxed on white)
and labelled "NN short-name" underneath. A missing figure gets a grey
placeholder cell labelled "missing", so gaps are visible rather than silent.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
COLS = 5
CELL_W, CELL_H = 640, 360  # image area per cell (16:9)
LABEL_H = 46
PAD = 14
TITLE_H = 90
BG = (255, 255, 255)
INK = (28, 28, 30)
MUTED = (110, 110, 118)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    prop = font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(prop), size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "contact_sheet.png"))
    args = ap.parse_args()

    projects = sorted(p for p in (ROOT / "projects").iterdir() if p.is_dir() and p.name[:2].isdigit())
    rows = math.ceil(len(projects) / COLS)
    W = COLS * CELL_W + (COLS + 1) * PAD
    H = TITLE_H + rows * (CELL_H + LABEL_H) + (rows + 1) * PAD
    sheet = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(sheet)

    f_title, f_sub, f_label = font(40, bold=True), font(20), font(22)
    draw.text((PAD, 14), "Outcrop: contact sheet", font=f_title, fill=INK)
    draw.text((PAD, 60), f"{len(projects)} projects, one figure each (projects/NN-slug/figure.png)", font=f_sub, fill=MUTED)

    for i, pdir in enumerate(projects):
        r, c = divmod(i, COLS)
        x0 = PAD + c * (CELL_W + PAD)
        y0 = TITLE_H + PAD + r * (CELL_H + LABEL_H + PAD)
        fig = pdir / "figure.png"
        nn, _, slug = pdir.name.partition("-")
        label = f"{nn}  {slug}"
        if fig.exists():
            im = Image.open(fig)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                flat = Image.new("RGB", im.size, BG)
                flat.paste(im, mask=im.split()[-1])
                im = flat
            else:
                im = im.convert("RGB")
            im.thumbnail((CELL_W, CELL_H), Image.LANCZOS)
            sheet.paste(im, (x0 + (CELL_W - im.width) // 2, y0 + (CELL_H - im.height) // 2))
        else:
            draw.rectangle([x0, y0, x0 + CELL_W, y0 + CELL_H], fill=(225, 225, 228))
            draw.text((x0 + CELL_W // 2, y0 + CELL_H // 2), "missing", font=f_label, fill=MUTED, anchor="mm")
            label += "  (no figure)"
        draw.rectangle([x0, y0, x0 + CELL_W, y0 + CELL_H], outline=(215, 215, 220), width=1)
        draw.text((x0 + CELL_W // 2, y0 + CELL_H + LABEL_H // 2), label, font=f_label, fill=INK, anchor="mm")

    sheet.save(args.out, optimize=True)
    print(f"wrote {args.out} ({W}x{H}, {len(projects)} cells)")


if __name__ == "__main__":
    main()
