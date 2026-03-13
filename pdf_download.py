# pdf_download.py
# Build a PDF that mirrors the HTML grid (Lines% bucket colors, cell text = Lines% only)
# and appends a "Execution Details" card section (one card per executed date per repo)
# including Build, Lines, Functions, and **Tests** (total/passed/failed/errors/skipped).

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from typing import Optional, Tuple, List, Dict, Set
from pathlib import Path
import pandas as pd
import datetime as dt
import math
import os
import json
from datetime import datetime

# ---------------------------
# Files
# ---------------------------
CONFIG_FILE = Path("config_api.json")  # used to include configured labels even if CSV has no rows for them

# ---------------------------
# Layout Config (easy to tune)
# ---------------------------
TITLE_FONT_SIZE    = 14   # main title
HEADER_FONT_SIZE   = 11   # "Repository / Level" + date headers
CELL_FONT_SIZE     = 10   # grid cell % text

LABEL_COL_W        = 300  # width of repo/level column
COL_W_MIN          = 90   # min width per date column
ROW_H              = 26   # height of each grid row
CELL_PAD_X         = 6    # left padding for cell text

HEADER_BAND_H      = 22   # reserved vertical band below header baseline
HEADER_TO_GRID_GAP = 6    # gap between header and first grid row

GRID_LINE_W        = 0.6
GRID_LINE_COLOR    = colors.HexColor("#cccccc")

DETAILS_TITLE      = "Execution Details"
CARD_COLS          = 2
CARD_H             = 104  # baseline/fallback; actual height is computed dynamically per card
CARD_GUTTER        = 14
CARD_PAD           = 10
CARD_BORDER_W      = 1.2
BOX_COLOR_SIZE     = 8    # small color box (Lines/Functions)

SHOW_LEGEND        = True  # draw a small legend below header

# --------- Color helpers (Lines% buckets = same as HTML) ---------
def lines_bucket_color_rgb(value: Optional[float]):
    """Return ReportLab color for coverage bucket based on Lines% (also reused for Functions%)."""
    if value is None or value < 0:
        return colors.HexColor("#d62728")  # NE Red
    if value < 25.0:
        return colors.HexColor("#ff7f0e")  # Orange
    if value < 50.0:
        return colors.HexColor("#ffbb78")  # Light Orange
    if value < 75.0:
        return colors.HexColor("#ffd700")  # Yellow
    return colors.HexColor("#2ca02c")      # Green

# --------- Small utilities ---------
def _fmt_duration_hms(total_seconds: Optional[int]) -> str:
    if not total_seconds or int(total_seconds) <= 0:
        return ""
    s = int(total_seconds)
    h, rem = divmod(s, 3600)
    m, ss = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {ss}s"
    if m > 0:
        return f"{m}m {ss}s"
    return f"{ss}s"

def _draw_color_box(c: canvas.Canvas, x: float, y: float, size: float, fill_color) -> None:
    c.setFillColor(fill_color)
    c.setStrokeColor(colors.black)
    c.rect(x, y, size, size, fill=1, stroke=1)

def _card(c: canvas.Canvas, x: float, y: float, w: float, h: float, stroke_color=colors.HexColor("#999999")):
    """Simple squared card (rectangle with a light gray border). (x,y) is bottom-left."""
    c.setStrokeColor(stroke_color)
    c.setFillColor(colors.white)
    c.setLineWidth(CARD_BORDER_W)
    c.rect(x, y, w, h, fill=1, stroke=1)

# ---- Badge helpers for tests ----
BADGE_PASS_BG = colors.Color(0.86, 0.96, 0.92)   # light green
BADGE_PASS_FG = colors.HexColor("#2e7d32")
BADGE_FAIL_BG = colors.Color(0.99, 0.90, 0.90)   # light red
BADGE_FAIL_FG = colors.HexColor("#c62828")
BADGE_WARN_BG = colors.Color(1.00, 0.96, 0.88)   # light amber
BADGE_WARN_FG = colors.HexColor("#b26a00")
BADGE_SKIP_BG = colors.Color(0.93, 0.94, 0.95)   # light gray
BADGE_SKIP_FG = colors.HexColor("#37474f")

BADGE_FONT     = "Helvetica"
BADGE_FONT_SZ  = 8
BADGE_PAD_X    = 4
BADGE_PAD_Y    = 2
BADGE_RADIUS   = 2

def _badge(c: canvas.Canvas, x: float, y: float, text: str,
           bg, fg, font=BADGE_FONT, size=BADGE_FONT_SZ, pad_x=BADGE_PAD_X, pad_y=BADGE_PAD_Y, radius=BADGE_RADIUS) -> float:
    """
    Draw a rounded color badge at (x,y). Returns the width consumed.
    Note: (x,y) is bottom-left of the badge rect.
    """
    text_w = c.stringWidth(text, font, size)
    w = text_w + pad_x * 2
    h = size + pad_y * 2 - 1  # small tweak so text sits centered
    c.setFillColor(bg)
    c.setStrokeColor(bg)
    try:
        c.roundRect(x, y, w, h, radius, fill=1, stroke=0)
    except Exception:
        c.rect(x, y, w, h, fill=1, stroke=0)
    c.setFillColor(fg)
    c.setFont(font, size)
    c.drawString(x + pad_x, y + pad_y - 1, text)
    return w

# ------------- Legend -------------
def _draw_legend(c: canvas.Canvas, x: float, y: float) -> float:
    """
    Draw a small bucket color legend; returns the width drawn.
    """
    items = [
        ("NE", "#d62728"),
        ("<25%", "#ff7f0e"),
        ("25–50%", "#ffbb78"),
        ("50–75%", "#ffd700"),
        (">=75%", "#2ca02c"),
    ]
    cur_x = x
    c.setFont("Helvetica", 8)
    for label, hexcol in items:
        c.setFillColor(colors.HexColor(hexcol))
        c.rect(cur_x, y, 10, 10, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(cur_x + 14, y, label)
        cur_x += 70
    return cur_x - x

# ------------- Measure tests block height based on data -------------
def calc_tests_block_height(ttot: Optional[int], font_size: int = 9) -> int:
    """
    Return vertical space (px) needed for the 'Tests' section.
    - If tests not available: 'Tests:' + one muted line.
    - If tests available: 'Tests:' + 'Total & badges' line + 'Errors/Skipped' line.
    """
    if ttot is None:
        return 12 + 12 + 4
    else:
        return 12 + 14 + 14 + 4

# ------------- Measure card height before drawing -------------
def calc_card_height(lines: Optional[float],
                     funcs: Optional[float],
                     bsucc: Optional[int],
                     bconc: str,
                     bdur: Optional[int],
                     ttot: Optional[int]) -> int:
    base_title = 14
    base_date  = 12
    base_build = 12
    base_lines = 14
    base_funcs = 14
    gaps = 10

    h = CARD_PAD + base_title + base_date + base_build + base_lines + base_funcs + gaps
    h += calc_tests_block_height(ttot)
    h += CARD_PAD
    return max(int(h), CARD_H)

# ------------- Config label loader -------------
def _load_config_labels(config_path: Path) -> Set[str]:
    labels: Set[str] = set()
    try:
        if not config_path.exists():
            return labels
        data = json.loads(config_path.read_text(encoding="utf-8"))
        for item in data:
            if not item.get("enabled", False):
                continue
            user = (item.get("user") or "").strip()
            repo = (item.get("repo") or "").strip()
            level = (item.get("test_level") or "").strip() or "L1"
            if user and repo:
                labels.add(f"{user}/{repo} [{level}]")
    except Exception:
        pass
    return labels

# --------- Main PDF builder ---------
def build_pdf_from_csv(
    csv_path: Path,
    output_pdf: Path,
    num_days: int,
    anchor_mode: str,
    custom_anchor_date: Optional[str],
    title: str = "Execution Grid — All Repos / Levels",
    include_status_in_grid: bool = False,
    details_only_executed: bool = True
) -> None:
    """Render a PDF grid (same window & rules as HTML), then append a card-based
    'Execution Details' section per repo & executed date, including Tests info."""
    if not csv_path.exists():
        print(f"[PDF] Missing CSV: {csv_path}")
        return

    # Load and normalize
    df = pd.read_csv(csv_path)
    if "label" not in df.columns and "repo" in df.columns:
        df = df.rename(columns={"repo":"label"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["lines_pct"] = pd.to_numeric(df["lines_pct"], errors="coerce")
    df["functions_pct"] = pd.to_numeric(df["functions_pct"], errors="coerce")

    if "build_success" in df.columns:
        df["build_success"] = pd.to_numeric(df["build_success"], errors="coerce")
    else:
        df["build_success"] = pd.NA
    if "build_duration_sec" in df.columns:
        df["build_duration_sec"] = pd.to_numeric(df["build_duration_sec"], errors="coerce")
    else:
        df["build_duration_sec"] = pd.NA
    if "build_status" not in df.columns:
        df["build_status"] = ""
    if "build_conclusion" not in df.columns:
        df["build_conclusion"] = ""

    for col in ["tests_total","tests_passed","tests_failures","tests_errors","tests_skipped"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date","label"])
    df = (df.sort_values(["label","date"])
            .drop_duplicates(subset=["label","date"], keep="last"))

    # --- Always build calendar window from anchor (Option A) ---
    if custom_anchor_date:
        anchor = dt.date.fromisoformat(custom_anchor_date)
    elif anchor_mode == "today":
        anchor = dt.date.today()
    else:
        all_dates = sorted(df["date"].unique())
        anchor = max(all_dates) if all_dates else dt.date.today()

    calendar_days = [anchor - dt.timedelta(days=i) for i in range(num_days)][::-1]

    # Labels from CSV and from config (union so repos always show)
    labels_csv = set(df["label"].unique())
    labels_cfg = _load_config_labels(CONFIG_FILE)
    labels = sorted(labels_csv.union(labels_cfg))

    # Build a lookup (label, date) -> dict of fields we need
    map_last = df.groupby(["label","date"], as_index=False).last()
    recs: Dict[tuple, dict] = {}
    for _, row in map_last.iterrows():
        recs[(row["label"], row["date"])] = {
            "lines": float(row["lines_pct"]) if pd.notna(row["lines_pct"]) else None,
            "funcs": float(row["functions_pct"]) if pd.notna(row["functions_pct"]) else None,
            "bsucc": (int(row["build_success"]) if pd.notna(row["build_success"]) else None),
            "bstat": (row["build_status"] if isinstance(row["build_status"], str) else ""),
            "bconc": (row["build_conclusion"] if isinstance(row["build_conclusion"], str) else ""),
            "bdur":  (int(row["build_duration_sec"]) if pd.notna(row["build_duration_sec"]) else None),
            # tests
            "ttot":  (int(row["tests_total"])    if pd.notna(row["tests_total"])    else None),
            "tpass": (int(row["tests_passed"])   if pd.notna(row["tests_passed"])   else None),
            "tfail": (int(row["tests_failures"]) if pd.notna(row["tests_failures"]) else None),
            "terr":  (int(row["tests_errors"])   if pd.notna(row["tests_errors"])   else None),
            "tskip": (int(row["tests_skipped"])  if pd.notna(row["tests_skipped"])  else None),
        }

    # Ensure all (label, date) slots exist so grid shows NE for missing data
    EMPTY = {
        "lines": None, "funcs": None,
        "bsucc": None, "bstat": "", "bconc": "", "bdur": None,
        "ttot": None, "tpass": None, "tfail": None, "terr": None, "tskip": None,
    }
    for label in labels:
        for d in calendar_days:
            recs.setdefault((label, d), EMPTY.copy())

    # =========================
    # PDF canvas (write to TMP then replace => avoids Windows lock)
    # =========================
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdf = output_pdf.with_suffix(".tmp.pdf")
    try:
        if tmp_pdf.exists():
            tmp_pdf.unlink()
    except Exception:
        pass

    page_w, page_h = landscape(A4)  # ~842 x 595
    c = canvas.Canvas(str(tmp_pdf), pagesize=(page_w, page_h))

    # Margins
    margin_l = 36
    margin_r = 24
    margin_t = 42
    margin_b = 36

    # Table geometry
    label_col_w = LABEL_COL_W
    row_h = ROW_H
    start_x = margin_l

    # =========================
    # GRID (with horizontal pagination by date slice)
    # =========================

    # Compute how many date columns fit per page (using COL_W_MIN per column)
    available_w = page_w - margin_l - margin_r - label_col_w
    max_cols_per_page = max(1, int(available_w // COL_W_MIN))

    # Helper: chunk a list into slices of size n
    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    def draw_grid_header_and_get_next_y(dates_slice):
        """
        Draws title and header row ("Repository / Level" + the current dates slice),
        returns Y where the first grid row should start (below reserved band).
        """
        # Title
        c.setFont("Helvetica-Bold", TITLE_FONT_SIZE)
        c.setFillColor(colors.black)
        c.drawString(margin_l, page_h - margin_t, title)

        # Header labels
        c.setFont("Helvetica-Bold", HEADER_FONT_SIZE)
        header_y = page_h - margin_t - 18
        c.drawString(start_x + 2, header_y, "Repository / Level")
        for j, d in enumerate(dates_slice):
            x = start_x + label_col_w + j * COL_W_MIN
            c.drawString(x + 4, header_y, d.isoformat())

        # Legend (optional)
        if SHOW_LEGEND:
            legend_y = header_y - 14
            _draw_legend(c, start_x + 2, legend_y)
            next_row_y = legend_y - (HEADER_BAND_H // 2) - HEADER_TO_GRID_GAP
        else:
            next_row_y = header_y - HEADER_BAND_H - HEADER_TO_GRID_GAP

        return next_row_y

    def draw_grid_page(dates_slice, first_page=False):
        # Start a new page for each slice except the very first grid page
        if not first_page:
            c.showPage()
        y_local = draw_grid_header_and_get_next_y(dates_slice)
        c.setFont("Helvetica", CELL_FONT_SIZE)

        for label in labels:
            # new page if row would go below bottom
            if y_local < margin_b + row_h:
                c.showPage()
                y_local = draw_grid_header_and_get_next_y(dates_slice)
                c.setFont("Helvetica", CELL_FONT_SIZE)

            # Label column cell
            c.setFillColor(colors.white)
            c.setStrokeColor(GRID_LINE_COLOR)
            c.setLineWidth(GRID_LINE_W)
            c.rect(start_x, y_local, label_col_w - 2, row_h - 2, fill=1, stroke=1)

            # Row label text
            c.setFillColor(colors.black)
            c.drawString(start_x + 2, y_local + (row_h // 2) - 3, label)

            # Date cells for this slice
            for j, d in enumerate(dates_slice):
                x = start_x + label_col_w + j * COL_W_MIN
                rec = recs.get((label, d), {})
                lines = rec.get("lines")

                # background by Lines bucket
                fill_col = lines_bucket_color_rgb(lines)
                c.setFillColor(fill_col)
                c.setStrokeColor(GRID_LINE_COLOR)
                c.setLineWidth(GRID_LINE_W)
                c.rect(x, y_local, COL_W_MIN - 2, row_h - 2, fill=1, stroke=1)

                # text color and value
                light_bg = fill_col in (colors.HexColor("#ffd700"), colors.HexColor("#ffbb78"))
                c.setFillColor(colors.black if light_bg else colors.white)
                pct_text = "NE" if (lines is None or lines < 0) else f"{math.trunc(float(lines)*10)/10:.1f}%"
                c.drawString(x + CELL_PAD_X, y_local + (row_h // 2) - 3, pct_text)

            y_local -= row_h

    # Draw the grid across N pages (each with a slice of dates)
    first_grid_page = True
    for dates_slice in _chunks(calendar_days, max_cols_per_page):
        draw_grid_page(dates_slice, first_page=first_grid_page)
        first_grid_page = False

    # =========================
    # DETAILS CARDS (like popup), only for executed days
    # =========================
    def draw_details_header_at(y_header: float):
        c.setFont("Helvetica-Bold", TITLE_FONT_SIZE)
        c.setFillColor(colors.black)
        c.drawString(margin_l, y_header, DETAILS_TITLE)

    # Always start details on a fresh page (simpler after multi-page grid)
    c.showPage()
    header_y = page_h - margin_t
    draw_details_header_at(header_y)
    top_y = header_y - 16

    # Two-column layout cursors
    card_cols = CARD_COLS
    gutter    = CARD_GUTTER
    card_w    = (page_w - margin_l - margin_r - (card_cols - 1) * gutter) / card_cols
    cur_x     = margin_l
    cur_y     = top_y

    def new_details_page():
        nonlocal cur_x, cur_y
        c.showPage()
        header_y2 = page_h - margin_t
        draw_details_header_at(header_y2)
        cur_x = margin_l
        cur_y = header_y2 - 16

    def place_next_card(card_h: int):
        """Return (x,y) for next card of height card_h; advance cursors; new page if needed."""
        nonlocal cur_x, cur_y
        x = cur_x
        y_card = cur_y - card_h

        # move to next column
        cur_x = cur_x + card_w + gutter
        # wrap to next row if exceeded columns
        if cur_x + card_w > page_w - margin_r + 0.1:
            cur_x = margin_l
            cur_y = y_card - gutter

        # if next row below bottom, new page and recalc position
        if y_card < margin_b + card_h:
            new_details_page()
            x = cur_x
            y_card = cur_y - card_h
            cur_x = cur_x + card_w + gutter
            if cur_x + card_w > page_w - margin_r + 0.1:
                cur_x = margin_l
                cur_y = y_card - gutter
        return x, y_card

    # Iterate labels & executed dates, making a card per executed date
    for label in labels:
        for d in calendar_days:
            rec = recs.get((label, d), {})
            lines = rec.get("lines")
            funcs = rec.get("funcs")
            bsucc = rec.get("bsucc")
            bconc = (rec.get("bconc") or "").strip()
            bdur  = rec.get("bdur")

            ttot  = rec.get("ttot")
            tpass = rec.get("tpass")
            tfail = rec.get("tfail")
            terr  = rec.get("terr")
            tskip = rec.get("tskip")

            executed = (lines is not None and lines >= 0)
            if details_only_executed and not executed:
                continue

            # Compute card height dynamically
            computed_card_h = calc_card_height(
                lines=lines,
                funcs=funcs,
                bsucc=bsucc,
                bconc=bconc,
                bdur=bdur,
                ttot=ttot
            )

            # Card placement using computed height
            x_card, y_card = place_next_card(computed_card_h)

            # Draw card background
            _card(c, x_card, y_card, card_w, computed_card_h)

            # Inside padding
            pad = CARD_PAD
            tx  = x_card + pad
            ty  = y_card + computed_card_h - pad - 2

            # Title: Repository / Level
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(colors.black)
            c.drawString(tx, ty, label)

            # Date
            c.setFont("Helvetica", 9)
            ty -= 14
            c.drawString(tx, ty, f"Date: {d.isoformat()}")

            # Build line
            ty -= 14
            if bsucc is None:
                build_text = "Build: (unknown)"
            else:
                if int(bsucc) == 1:
                    build_text = "Build: Success"
                else:
                    pretty = (bconc.capitalize() if bconc else "Failed")
                    build_text = f"Build: {pretty}"
            dur_text = _fmt_duration_hms(bdur)
            if dur_text:
                build_text += f" · {dur_text}"
            c.drawString(tx, ty, build_text)

            # Lines (with color square)
            ty -= 14
            _draw_color_box(c, tx, ty + 2, BOX_COLOR_SIZE, lines_bucket_color_rgb(lines))
            if lines is None or lines < 0:
                c.drawString(tx + BOX_COLOR_SIZE + 4, ty, "Lines: NE")
            else:
                c.drawString(tx + BOX_COLOR_SIZE + 4, ty, f"Lines: {float(lines):.2f}%")

            # Functions (with color square)
            ty -= 14
            _draw_color_box(c, tx, ty + 2, BOX_COLOR_SIZE, lines_bucket_color_rgb(funcs))
            if funcs is None or funcs < 0:
                c.drawString(tx + BOX_COLOR_SIZE + 4, ty, "Functions: NE")
            else:
                c.drawString(tx + BOX_COLOR_SIZE + 4, ty, f"Functions: {float(funcs):.2f}%")

            # ---- Tests ----
            ty -= 16
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(colors.black)
            c.drawString(tx, ty, "Tests:")
            c.setFont("Helvetica", 9)

            if ttot is None:
                ty -= 12
                c.setFillColor(colors.HexColor("#777777"))
                c.drawString(tx, ty, "Test cases information not available")
                c.setFillColor(colors.black)
            else:
                # Line 1: Total + Passed + Failed badges
                ty -= 12
                line_x = tx
                c.setFillColor(colors.black)
                total_txt = f"Total: {ttot}"
                c.drawString(line_x, ty, total_txt)
                line_x += c.stringWidth(total_txt, "Helvetica", 9) + 10

                # Passed
                w = _badge(c, line_x, ty - 9, f"Passed: {tpass or 0}", BADGE_PASS_BG, BADGE_PASS_FG)
                line_x += w + 6

                # Failed (failures + errors)
                failed_total = (tfail or 0) + (terr or 0)
                _ = _badge(c, line_x, ty - 9, f"Failed: {failed_total}", BADGE_FAIL_BG, BADGE_FAIL_FG)

                # Line 2: Errors + Skipped badges
                ty -= 14
                line_x = tx
                w = _badge(c, line_x, ty - 9, f"Errors: {terr or 0}", BADGE_WARN_BG, BADGE_WARN_FG)
                line_x += w + 6
                _ = _badge(c, line_x, ty - 9, f"Skipped: {tskip or 0}", BADGE_SKIP_BG, BADGE_SKIP_FG)

    # =========================
    # Save safely (avoid lock issues)
    # =========================
    c.save()
    try:
        os.replace(tmp_pdf, output_pdf)
        print(f"Saved PDF: {output_pdf.resolve()}")
    except PermissionError:
        ts_name = f"{output_pdf.stem}_{datetime.now():%Y%m%d_%H%M%S}{output_pdf.suffix}"
        alt_pdf = output_pdf.with_name(ts_name)
        os.replace(tmp_pdf, alt_pdf)
        print(f"[PDF] '{output_pdf.name}' is locked. Saved as '{alt_pdf.name}'.")
        print(f"Full path: {alt_pdf.resolve()}")