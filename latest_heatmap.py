# latest_heatmap.py
from __future__ import annotations

import os
import re
import io
import csv
import json
import math
import time
import zipfile
import shutil
import tempfile
import textwrap
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from pdf_download import build_pdf_from_csv  # your existing module

# -----------------------
# Files & parameters
# -----------------------
CONFIG_FILE = Path("config_api.json")
ENV_FILE = Path(".env")

HISTORY_CSV = Path("coverage_history_api.csv")
EXECUTION_ALL_HTML = Path("execution_grid_final.html")
EXECUTION_ALL_PDF  = Path("execution_grid_final.pdf")

NUM_CAL_DAYS = 3
ANCHOR_MODE = "today"           # "max_report_date" | "today"
CUSTOM_ANCHOR_DATE: Optional[str] = None  # e.g., "2026-02-04"

MAX_RUNS_TO_SCAN = 50  # slightly larger to be safe

ARTIFACT_NAME_PATTERNS = [
    "combinedcoverage-report", "coverage", "lcov", "report", "coverage-report", "artifacts", "artifact","l2-coverage-report"
]
TEST_ARTIFACT_PATTERNS = [
    "ctest", "junit", "test-results", "test_result", "tests", "utests", "unit-test", "unittest","l2-test-results"
]

VERBOSE = False
def vprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)

# -----------------------
# Utils / helpers
# -----------------------

def _coerce_date_to_window(date_str: Optional[str], target_dates: set[str]) -> str:
    """
    Return date_str if it is within target_dates; otherwise map to the latest day in window.
    If date_str is None, also map to latest day.
    """
    if target_dates:
        latest = sorted(target_dates)[-1]
    else:
        latest = dt.date.today().isoformat()
    if not date_str or date_str not in target_dates:
        return latest
    return date_str
# -----------------------
# Pages (GitHub Pages) coverage fetch
# -----------------------
from urllib.parse import urljoin

def _discover_coverage_index_urls(html: str, base_url: str) -> List[str]:
    """
    From a landing page HTML, find likely coverage index links.
    Heuristics:
      - <a href> whose href contains 'coverage_report' and 'index.html'
      - link text contains 'coverage' (case-insensitive)
      - fallback to '<base>/coverage_report/index.html'
    Returns absolute URLs (deduped, in priority order).
    """
    soup = BeautifulSoup(html, "lxml")
    cands = []

    # Anchors with href matching coverage_report/index.html
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        txt = (a.get_text(" ", strip=True) or "").lower()
        absu = urljoin(base_url, href)

        if "coverage_report" in href and "index.html" in href:
            cands.append(absu)
            continue

        # If anchor text mentions "coverage", try it
        if "coverage" in txt and href:
            cands.append(absu)

    # Fallback hard guess
    cands.append(urljoin(base_url, "coverage_report/index.html"))

    # Deduplicate preserving order
    out, seen = [], set()
    for u in cands:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def fetch_parse_coverage_from_urls(urls: List[str], timeout: int = 30) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Try URLs in order; supports both:
      - direct coverage index html
      - landing pages that link to coverage index
    Returns (date_str, lines_pct, funcs_pct).
    """
    for url in urls or []:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                vprint(f"[PAGES] GET {url} -> {r.status_code}")
                continue

            # If it's probably a landing page, try to discover the coverage index
            is_landing = (url.rstrip("/").endswith(".html") is False)
            child_urls = _discover_coverage_index_urls(r.text, url) if is_landing else [url]

            for u in child_urls:
                try:
                    rr = requests.get(u, timeout=timeout)
                    if rr.status_code != 200:
                        vprint(f"[PAGES] GET {u} -> {rr.status_code}")
                        continue
                    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".html") as tmp:
                        tmp.write(rr.content)
                        temp_path = Path(tmp.name)
                    try:
                        date_str = parse_date_from_index(temp_path) or None
                        lines, funcs = parse_lines_funcs_from_index(temp_path)
                        if lines is not None or funcs is not None:
                            vprint(f"[PAGES] Parsed coverage from {u}: lines={lines}, funcs={funcs}, date={date_str}")
                            return date_str, lines, funcs
                    finally:
                        try: os.remove(temp_path)
                        except: pass
                except Exception as inner_ex:
                    vprint(f"[PAGES] Failed to parse {u}: {inner_ex}")

        except Exception as ex:
            vprint(f"[PAGES] Failed to fetch {url}: {ex}")

    return None, None, None


def load_env_if_present(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

def truncate_one_decimal(x: float) -> float:
    return math.trunc(x * 10.0) / 10.0

def lines_bucket_color(lines: Optional[float]) -> str:
    if lines is None or lines < 0: return "#d62728"
    if lines < 25.0: return "#ff7f0e"
    if lines < 50.0: return "#ffbb78"
    if lines < 75.0: return "#ffd700"
    return "#2ca02c"

def find_file(root: Path, filename_lower: str) -> Optional[Path]:
    """
    Recursively find a file by exact name (case-insensitive).
    """
    root = Path(root)
    fl = filename_lower.lower()
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == fl:
            return p
    return None

def safe_label(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s or "")

# Build target dates = last N days (YYYY-MM-DD)
def build_target_date_set(num_days: int, anchor_mode: str, csv_path: Path) -> set[str]:
    """
    Build the set of YYYY-MM-DD we care about (N consecutive days).
    - If CSV exists and anchor_mode == 'max_report_date', anchor = max date in CSV; else anchor = today.
    """
    if anchor_mode == "max_report_date" and csv_path.exists():
        try:
            _df = pd.read_csv(csv_path, usecols=["date"])
            _df["date"] = pd.to_datetime(_df["date"], errors="coerce").dt.date
            max_d = _df["date"].dropna().max()
            anchor = max_d or dt.date.today()
        except Exception:
            anchor = dt.date.today()
    else:
        anchor = dt.date.today()
    days = [anchor - dt.timedelta(days=i) for i in range(num_days)]
    return {d.isoformat() for d in sorted(days)}

# -----------------------
# Coverage parsers
# -----------------------
def parse_lcov_info(lcov_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not lcov_path.exists():
        return None, None
    lh = lf = fnh = fnf = 0
    with lcov_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("LH:"):
                try: lh += int(line.split(":", 1)[1].strip())
                except: pass
            elif line.startswith("LF:"):
                try: lf += int(line.split(":", 1)[1].strip())
                except: pass
            elif line.startswith("FNH:"):
                try: fnh += int(line.split(":", 1)[1].strip())
                except: pass
            elif line.startswith("FNF:"):
                try: fnf += int(line.split(":", 1)[1].strip())
                except: pass
    lines_pct = (lh / lf * 100.0) if lf > 0 else None
    funcs_pct = (fnh / fnf * 100.0) if fnf > 0 else None
    return lines_pct, funcs_pct

def parse_date_from_index(index_path: Path) -> Optional[str]:
    """
    Return 'YYYY-MM-DD' from index.html by scanning a few common patterns.
    """
    if not index_path.exists():
        return None
    html = index_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True).replace("\xa0", " ")

    patterns = [
        r"\bDate:\s*(\d{4}-\d{2}-\d{2})\b",
        r"\bGenerated on\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})\b",
        r"\bCoverage report generated.*?(\d{4}-\d{2}-\d{2})\b",
        r"\bGenerated by lcov.*?on\s*(\d{4}-\d{2}-\d{2})\b",
        r"\bReport date\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def _run_date_fallback(run: dict) -> Optional[str]:
    # Prefer run_started_at; fallback created_at
    raw = run.get("run_started_at") or run.get("created_at")
    if not raw:
        return None
    try:
        dt_ = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt_.date().isoformat()

def _pct_or_hit_total(block: str, word: str) -> Optional[float]:
    m = re.search(rf"{word}\s*[:\-].*?([0-9]+(?:\.[0-9]+)?)\s*%", block, re.IGNORECASE)
    if m:
        try: return float(m.group(1))
        except: pass
    mh = re.search(rf"{word}.*?\bHit\b\s*([0-9,]+)", block, re.IGNORECASE)
    mt = re.search(rf"{word}.*?\bTotal\b\s*([0-9,]+)", block, re.IGNORECASE)
    if mh and mt:
        try:
            hit = int(mh.group(1).replace(",", ""))
            tot = int(mt.group(1).replace(",", ""))
            if tot > 0: return (hit / tot) * 100.0
        except: pass
    return None

def parse_lines_funcs_from_index(index_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not index_path.exists():
        return None, None
    html = index_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True).replace("\xa0", " ")
    idx = text.lower().find("current view: top level")
    scan = text[idx: idx + 1500] if idx >= 0 else text[:2000]
    lines_pct = _pct_or_hit_total(scan, "Lines")
    funcs_pct = _pct_or_hit_total(scan, "Functions")
    return lines_pct, funcs_pct

# -----------------------
# Test discovery & parsing
# -----------------------
def detect_test_files(extract_dir: str):
    junit_files, ctest_files = [], []
    import xml.etree.ElementTree as ET
    for dp, _, files in os.walk(extract_dir):
        for f in files:
            if not f.lower().endswith(".xml"):
                continue
            p = os.path.join(dp, f)
            try:
                root = ET.parse(p).getroot()
                tag = root.tag.lower()
                if tag.endswith("testsuites") or tag.endswith("testsuite"):
                    junit_files.append(p)
                elif tag in ("site", "testing") or "testing" in tag:
                    ctest_files.append(p)
            except Exception:
                pass
    return junit_files, ctest_files

def parse_junit(files: List[str]):
    out = {"total":0, "failures":0, "errors":0, "skipped":0}
    import xml.etree.ElementTree as ET
    for p in files:
        try:
            r = ET.parse(p).getroot()
            suites = r.findall(".//testsuite")
            if not suites:
                suites = [r] if r.tag.endswith("testsuite") else []
            if suites:
                for s in suites:
                    out["total"]    += int(s.attrib.get("tests", 0))
                    out["failures"] += int(s.attrib.get("failures", 0))
                    out["errors"]   += int(s.attrib.get("errors", 0))
                    out["skipped"]  += int(s.attrib.get("skipped", 0))
        except Exception:
            pass
    out["passed"] = max(0, out["total"] - out["failures"] - out["errors"] - out["skipped"])
    return out

def parse_ctest(files: List[str]):
    out = {"total":0, "failures":0, "errors":0, "skipped":0}
    import xml.etree.ElementTree as ET
    for p in files:
        try:
            r = ET.parse(p).getroot()
            tests = r.findall(".//Test") or r.findall(".//test")
            for t in tests:
                status = (t.attrib.get("Status") or t.attrib.get("status") or "").lower()
                out["total"] += 1
                if status in ("passed", "success"):
                    pass
                elif status in ("skipped", "notrun", "disabled"):
                    out["skipped"] += 1
                elif status in ("failed", "fail"):
                    out["failures"] += 1
                else:
                    out["errors"] += 1
        except Exception:
            pass
    out["passed"] = max(0, out["total"] - out["failures"] - out["errors"] - out["skipped"])
    return out

# -----------------------
# CSV helpers
# -----------------------
def ensure_csv_schema() -> None:
    desired = ["date","label","lines_pct","functions_pct",
               "build_status","build_conclusion","build_success","build_duration_sec",
               "tests_total","tests_passed","tests_failures","tests_errors","tests_skipped"]
    if not HISTORY_CSV.exists():
        with HISTORY_CSV.open("w", newline="") as f:
            f.write(",".join(desired) + "\n")
        return
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception as ex:
        raise SystemExit(f"Failed to read {HISTORY_CSV}: {ex}")
    for c in desired:
        if c not in df.columns:
            df[c] = np.nan
    df = df[desired]
    df.to_csv(HISTORY_CSV, index=False)
    print("Ensured CSV schema with build & tests fields.")

def add_row(date_str: str, label: str, lines: float, funcs: float,
            build_status: Optional[str] = None,
            build_conclusion: Optional[str] = None,
            build_success: Optional[int] = None,
            build_duration_sec: Optional[int] = None,
            tests_total: Optional[int] = None,
            tests_passed: Optional[int] = None,
            tests_failures: Optional[int] = None,
            tests_errors: Optional[int] = None,
            tests_skipped: Optional[int] = None) -> None:
    with HISTORY_CSV.open("a", newline="") as f:
        csv.writer(f).writerow([
            date_str,
            label,
            f"{lines:.4f}",
            f"{funcs:.4f}",
            (build_status or ""),
            (build_conclusion or ""),
            ("" if build_success is None else str(int(build_success))),
            ("" if build_duration_sec is None else str(int(build_duration_sec))),
            ("" if tests_total    is None else str(int(tests_total))),
            ("" if tests_passed   is None else str(int(tests_passed))),
            ("" if tests_failures is None else str(int(tests_failures))),
            ("" if tests_errors   is None else str(int(tests_errors))),
            ("" if tests_skipped  is None else str(int(tests_skipped))),
        ])

# -----------------------
# GitHub API helpers
# -----------------------
def gh_headers(token: str) -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def req_get(url: str, headers: Dict[str, str], params: dict | None = None, timeout: int = 30, allow_redirects: bool = True):
    for attempt in range(3):
        r = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=allow_redirects)
        if r.status_code != 429:
            return r
        wait = 2 ** attempt
        print(f"[RATE] 429 Too Many Requests; sleeping {wait}s...")
        time.sleep(wait)
    return r

def list_runs(owner: str, repo: str, token: str, per_page: int = 100, branch: str = "") -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    params = {"per_page": per_page}
    if branch:
        params["branch"] = branch
    r = req_get(url, gh_headers(token), params=params, timeout=30)
    if r.status_code == 401:
        raise requests.HTTPError("401 Unauthorized.", response=r)
    r.raise_for_status()
    return r.json().get("workflow_runs", []) or []

def filter_and_sort_runs(runs: List[dict], name_contains: str) -> List[dict]:
    key = (name_contains or "").lower().strip()
    if key:
        runs = [r for r in runs if key in (r.get("name","").lower())]
    # Run start time is the best indicator of "when it ran"
    def _key(r):
        return r.get("run_started_at") or r.get("created_at") or ""
    runs = sorted(runs, key=lambda x: _key(x), reverse=True)
    return runs

def list_artifacts_for_run(owner: str, repo: str, run_id: int, token: str) -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    r = req_get(url, gh_headers(token), timeout=30)
    if r.status_code == 401:
        raise requests.HTTPError("401 Unauthorized when listing artifacts.", response=r)
    r.raise_for_status()
    return r.json().get("artifacts", []) or []

def artifact_matches(name: str, patterns: List[str]) -> bool:
    nm = (name or "").lower()
    return any(p.lower() in nm for p in patterns)

# -----------------------
# HTML writer (centered, clean popup)
# -----------------------
def write_combined_execution_grid_html(labels: List[str],
                                       calendar_days: List[dt.date],
                                       daymaps: Dict[str, Dict[str, Dict[str, Optional[float]]]],
                                       html_path: Path) -> None:
    ths = "".join(f"<th>{d.isoformat()}</th>" for d in calendar_days)

    rows_html = []
    for label in labels:
        tds = []
        for d in calendar_days:
            ds = d.isoformat()
            info = daymaps[label][ds]
            lines = info["lines"]; funcs = info["funcs"]; avg = info["avg"]
            bsucc = info.get("build_success")
            bstatus = info.get("build_status") or ""
            bconcl = info.get("build_conclusion") or ""
            bdur   = info.get("build_duration_sec")
            executed = (lines is not None and lines >= 0)

            pct_text = f"{truncate_one_decimal(float(lines)):.1f}%" if executed else "NE"
            bg = lines_bucket_color(lines)
            color = "#ffffff"
            tds.append(textwrap.dedent(f"""
                <td class="cell" style="background:{bg}; color:{color}"
                    data-label="{label}"
                    data-date="{ds}"
                    data-exists="{str(executed).lower()}"
                    data-lines="{'' if lines is None else f'{lines:.2f}'}"
                    data-funcs="{'' if funcs is None else f'{funcs:.2f}'}"
                    data-bsuccess="{'' if bsucc is None else int(bsucc)}"
                    data-bstatus="{bstatus}"
                    data-bconcl="{bconcl}"
                    data-bdur="{'' if bdur is None else int(bdur)}"
                    data-ttotal="{'' if info.get('tests_total')    is None else int(info.get('tests_total'))}"
                    data-tpass="{''  if info.get('tests_passed')   is None else int(info.get('tests_passed'))}"
                    data-tfail="{''  if info.get('tests_failures') is None else int(info.get('tests_failures'))}"
                    data-terror="{'' if info.get('tests_errors')   is None else int(info.get('tests_errors'))}"
                    data-tskip="{''  if info.get('tests_skipped')  is None else int(info.get('tests_skipped'))}"
                >{pct_text}</td>
            """).strip())
        rows_html.append(f"<tr><td class='rlabel'>{label}</td>{''.join(tds)}</tr>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Execution Grid — All</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    :root {{
      --bg:#0f1115; --panel:#151925; --text:#e9edf3; --muted:#aab4c3;
      --line:#2b3244; --success:#2ecc71; --danger:#e74c3c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--text);
      font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
      display:flex; align-items:flex-start; justify-content:center;
    }}
    .container {{ width:min(1100px,96vw); margin:28px auto 64px; }}
    h2 {{ text-align:center; margin:0 0 18px 0; font-weight:600; }}
    .grid-wrap {{
      background: var(--panel); border:1px solid var(--line); border-radius:10px; overflow:auto;
      box-shadow:0 10px 30px rgba(0,0,0,.35);
    }}
    table.grid {{ border-collapse:collapse; width:100%; min-width:760px; }}
    table.grid th, table.grid td {{ border-bottom:1px solid var(--line); padding:10px 12px; text-align:center; }}
    table.grid th {{ position:sticky; top:0; background:#1a2030; z-index:1; font-weight:600; }}
    td.rlabel {{ font-weight:700; text-align:left; min-width:260px; white-space:nowrap; }}
    .cell {{ cursor:pointer; font-weight:700; border-left:1px solid var(--line); }}
    .modal {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:9999; align-items:center; justify-content:center; padding:16px; }}
    .modal-content {{ width:min(640px,96vw); background:#101524; border:1px solid var(--line); color:var(--text); border-radius:12px; padding:16px 18px; box-shadow:0 16px 40px rgba(0,0,0,.5); }}
    .modal-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }}
    .modal-title {{ font-weight:600; font-size:16px; }}
    .close-btn {{ cursor:pointer; font-size:20px; color:#aab4c3; }}
    .metric {{ margin-top:8px; font-size:14px; color:var(--text); }}
    .muted {{ color:#aab4c3; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-left:6px; background:#2b3244; color:#d1d6e3; }}
    .badge.pass {{ background:rgba(46,204,113,.15); color:var(--success); }}
    .badge.fail {{ background:rgba(231,76,60,.15); color:var(--danger); }}
    .kbox {{ width:12px; height:12px; display:inline-block; margin-right:6px; vertical-align:middle; border:1px solid #444; }}
    .k-red {{ background:#d62728; }} .k-orange {{ background:#ff7f0e; }} .k-lorange {{ background:#ffbb78; }} .k-yellow {{ background:#ffd700; }} .k-green {{ background:#2ca02c; }}
    .legend {{ margin-top:10px; display:grid; grid-template-columns:auto 1fr; gap:6px 10px; color:#aab4c3; }}
  </style>
</head>
<body>
  <div class="container">
    <h2>Execution Grid — All Repos / Levels</h2>
    <div class="grid-wrap">
      <table class="grid">
        <thead><tr><th>Repository / Level</th>{ths}</tr></thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="modal" id="modal">
    <div class="modal-content">
      <div class="modal-header">
        <div class="modal-title" id="modal-title">Details</div>
        <div class="close-btn" onclick="closeModal()">✖</div>
      </div>
      <div id="modal-body"></div>
    </div>
  </div>

  <script>
    const modal = document.getElementById('modal');
    const bodyEl = document.getElementById('modal-body');
    function openModal(html) {{ bodyEl.innerHTML = html; modal.style.display = 'flex'; }}
    function closeModal() {{ modal.style.display = 'none'; }}
    window.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeModal(); }});
    modal.addEventListener('click', (e) => {{ if (e.target.id === 'modal') closeModal(); }});

    document.addEventListener('click', function(ev) {{
      const td = ev.target.closest('.cell');
      if (!td) return;

      const label = td.dataset.label;
      const date  = td.dataset.date;
      const exists = td.dataset.exists === 'true';
      const lines = td.dataset.lines;
      const funcs = td.dataset.funcs;
      const bsucc = td.dataset.bsuccess;
      const bconc = td.dataset.bconcl || '';
      const bdur  = td.dataset.bdur;

      const ttotal = td.dataset.ttotal;
      const tpass  = td.dataset.tpass;
      const tfail  = td.dataset.tfail;
      const terror = td.dataset.terror;
      const tskip  = td.dataset.tskip;

      function fmtDur(sec) {{
        if (!sec) return '';
        const s = Number(sec);
        const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = s%60;
        return (h>0? (h+'h '):'') + (m>0? (m+'m '):'') + ss + 's';
      }}

      function testsLine() {{
        if (ttotal === undefined || ttotal === null || ttotal === '') {{
          return '<span class="muted">Test cases information not available</span>';
        }}
        const parts = [];
        parts.push(`${{ttotal}} total`);
        parts.push(`✅ ${{tpass || 0}} passed`);
        parts.push(`❌ ${{tfail || 0}} failed`);
        parts.push(`⚠️ ${{terror || 0}} errors`);
        parts.push(`⏭️ ${{tskip || 0}} skipped`);
        return parts.join(' · ');
      }}

      let html = '';
      html += '<div class="metric"><b>Repository / Level:</b> ' + label + '</div>';
      html += '<div class="metric"><b>Date:</b> ' + date + '</div>';

      if (bsucc === '1') {{
        html += '<div class="metric"><b>Build:</b> ✔ Success';
        if (bdur) html += ' · ' + fmtDur(bdur);
        html += '</div>';
      }} else if (bsucc === '0') {{
        const t = (bconc || 'Unknown');
        html += '<div class="metric"><b>Build:</b> ✖ ' + t.charAt(0).toUpperCase() + t.slice(1);
        if (bdur) html += ' · ' + fmtDur(bdur);
        html += '</div>';
      }}

      if (exists) {{
        html += '<div class="metric"><b>Lines:</b> ' + lines + '%</div>';
        if (funcs) html += '<div class="metric"><b>Functions:</b> ' + funcs + '%</div>';
      }} else {{
        html += '<div class="metric" style="color:#d62728;"><b>Not Executed</b></div>';
      }}

      const failedTotal = (Number(tfail||0) + Number(terror||0));
      if (ttotal === undefined || ttotal === null || ttotal === '') {{
        html += '<div class="metric"><b>Tests:</b> ' + testsLine() + '</div>';
      }} else {{
        html += '<div class="metric"><b>Tests:</b> ' + testsLine()
             +  ' <span class="badge pass">Passed: ' + (tpass||0) + '</span>'
             +  ' <span class="badge fail">Failed: ' + failedTotal + '</span>'
             +  '</div>';
      }}

        
        html += '';
    

      document.getElementById('modal-title').innerText = exists ? 'Execution Details' : 'Not Executed';
      openModal(html);
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved {html_path.resolve()}")

# -----------------------
# Calendar & matrix
# -----------------------
def build_calendar_window(all_report_dates: List[dt.date]) -> List[dt.date]:
    if CUSTOM_ANCHOR_DATE:
        anchor = dt.date.fromisoformat(CUSTOM_ANCHOR_DATE)
    elif ANCHOR_MODE == "today":
        anchor = dt.date.today()
    else:
        anchor = max(all_report_dates) if all_report_dates else dt.date.today()
    return [anchor - dt.timedelta(days=i) for i in range(NUM_CAL_DAYS)][::-1]

def build_day_map_for_label(df: pd.DataFrame, label: str, calendar_days: List[dt.date]) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {}
    sub = df[df["label"] == label].groupby("date", as_index=False).last()
    look: Dict[str, dict] = {}

    def _num_or_none(row: pd.Series, col: str):
        try:
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        except Exception:
            pass
        return None

    for _, row in sub.iterrows():
        d = row["date"].isoformat()
        look[d] = {
            "lines": _num_or_none(row, "lines_pct"),
            "funcs": _num_or_none(row, "functions_pct"),
            "build_status": row["build_status"] if "build_status" in row.index else None,
            "build_conclusion": row["build_conclusion"] if "build_conclusion" in row.index else None,
            "build_success": _num_or_none(row, "build_success"),
            "build_duration_sec": _num_or_none(row, "build_duration_sec"),
            "tests_total": _num_or_none(row, "tests_total"),
            "tests_passed": _num_or_none(row, "tests_passed"),
            "tests_failures": _num_or_none(row, "tests_failures"),
            "tests_errors": _num_or_none(row, "tests_errors"),
            "tests_skipped": _num_or_none(row, "tests_skipped"),
        }

    for d in calendar_days:
        ds = d.isoformat()
        rec = look.get(ds, {})
        lines = rec.get("lines")
        funcs = rec.get("funcs")
        executed = (lines is not None and lines >= 0)
        avg = None
        if (lines is not None and funcs is not None and lines >= 0 and funcs >= 0):
            avg = (lines + funcs) / 2.0
        result[ds] = {
            "exists": executed or ((funcs is not None) and funcs >= 0),
            "lines": lines,
            "funcs": funcs,
            "avg": avg,
            "build_status": rec.get("build_status"),
            "build_conclusion": rec.get("build_conclusion"),
            "build_success": rec.get("build_success"),
            "build_duration_sec": rec.get("build_duration_sec"),
            "tests_total": rec.get("tests_total"),
            "tests_passed": rec.get("tests_passed"),
            "tests_failures": rec.get("tests_failures"),
            "tests_errors": rec.get("tests_errors"),
            "tests_skipped": rec.get("tests_skipped"),
        }
    return result

def build_matrix_and_daymaps(labels: List[str], calendar_days: List[dt.date], df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    return {label: build_day_map_for_label(df, label, calendar_days) for label in labels}

# -----------------------
# MAIN
# -----------------------
def main():
    load_env_if_present(ENV_FILE)
    ensure_csv_schema()

    # Precompute the exact dates we want to fill in the grid
    target_dates = build_target_date_set(NUM_CAL_DAYS, ANCHOR_MODE, HISTORY_CSV)
    print(f"[INFO] Target dates: {sorted(target_dates)}")

    if not CONFIG_FILE.exists():
        raise SystemExit(f"Missing {CONFIG_FILE}.")
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as ex:
        raise SystemExit(f"Failed to parse {CONFIG_FILE}: {ex}")

    # Ingest all items
    for item in config:
        if not item.get("enabled", False):
            continue

        user  = (item.get("user") or "").strip()
        repo  = (item.get("repo") or "").strip()
        level = (item.get("test_level") or "").strip() or "L1"
        label = f"{user}/{repo} [{level}]"

        workflow_name  = (item.get("workflow") or "").strip()
        branch_filter  = (item.get("branch") or "").strip()
        #patterns       = item.get("artifact_patterns") or ARTIFACT_NAME_PATTERNS
        patterns       = item.get("artifact_patterns") or ARTIFACT_NAME_PATTERNS
        test_patterns  = item.get("test_artifact_patterns") or TEST_ARTIFACT_PATTERNS

        coverage_urls  = item.get("coverage_urls") or []
        cov_tpl        = (item.get("coverage_url_template") or "").strip()
        use_pages_fallback = item.get("use_pages_when_no_artifacts")
        if use_pages_fallback is None:
            use_pages_fallback = True

        # Build the final list of candidate coverage URLs (template first, then explicit list)
        pages_candidates: List[str] = []
        if cov_tpl:
            try:
                pages_candidates.append(cov_tpl.format(owner=user, repo=repo, level=level))
            except Exception:
                # Ignore bad templates
                pass
        pages_candidates.extend([u for u in coverage_urls if u])
        token_env      = (item.get("token_env") or "").strip()
        if token_env:
            token = (os.getenv(token_env, "") or "").strip()
        else:
            token = (os.getenv("GITHUB_TOKEN", "") or os.getenv("PAT", "") or "").strip()

        if not user or not repo:
            print(f"[SKIP] Invalid item (missing user/repo): {item}")
            continue
        if not token:
            print(f"[WARN] {label}: No token found. Set GITHUB_TOKEN/PAT in env. Skipping.")
            continue

        try:
            runs = list_runs(user, repo, token, per_page=100, branch=branch_filter)
            runs = filter_and_sort_runs(runs, workflow_name)
            print(f"[INFO] {label}: found {len(runs)} runs after workflow filter.")

            if not runs:
                print(f"[INFO] {label}: no workflow runs found (workflow='{workflow_name}', branch='{branch_filter}').")
                if pages_candidates:
                    # Attempt pages-only ingest (single snapshot)
                    p_date, p_lines, p_funcs = fetch_parse_coverage_from_urls(pages_candidates)
                    # If page didn’t expose a date, anchor to today
                    date_str = _coerce_date_to_window(p_date, target_dates)
                    if p_lines is not None or p_funcs is not None:
                        add_row(date_str, label,
                                (p_lines if p_lines is not None else -1.0),
                                (p_funcs if p_funcs is not None else -1.0),
                                build_status=None, build_conclusion=None, build_success=None, build_duration_sec=None,
                                tests_total=None, tests_passed=None, tests_failures=None, tests_errors=None, tests_skipped=None)
                        print(f"[OK] {label} (pages-only) :: {date_str} :: Lines={'NE' if p_lines is None else f'{p_lines:.2f}%'} "
                  f":: Functions={'NE' if p_funcs is None else f'{p_funcs:.2f}%'}")
                continue

            seen_dates_for_label: set[str] = set()
            processed = 0

            for run in runs[:MAX_RUNS_TO_SCAN]:
                run_id = run["id"]
                run_name = run.get("name","")
                created_at = run.get("created_at","")
                run_started_at = run.get("run_started_at", "")
                run_status = run.get("status", "")
                run_conclusion = run.get("conclusion", "")
                run_date_hint = _run_date_fallback(run)
                # # ============= DEBUG LOGGING =============
                # print(f"\n[RUN-CANDIDATE] {label}")
                # print(f"    Date:       {run_date_hint}")
                # print(f"    Started:    {run_started_at}")
                # print(f"    Created:    {created_at}")
                # print(f"    Run ID:     {run_id}")
                # print(f"    Name:       {run_name}")
                # print(f"    Status:     {run_status}")
                # print(f"    Conclusion: {run_conclusion}")
                # print(f"    URL:        https://github.com/{user}/{repo}/actions/runs/{run_id}")
                # ADD THIS DEBUG OUTPUT
                #print(f"[RUN-CANDIDATE] {label} | Date: {run_date_hint} | Started: {run_started_at} | ID: {run_id} | Name: {run_name}")
                #DEBUG
                # ===== SIMPLE DEBUG =====
                print(f"[RUN] {run_date_hint} | {run_started_at} | ID: {run_id} | {run_name}")
                print(f"      URL: https://github.com/{user}/{repo}/actions/runs/{run_id}")


                # print(f"[RUN-INFO] Workflow Name: {run_name}, Run ID: {run_id}")
                # print(f"[RUN-INFO] URL: https://github.com/{user}/{repo}/actions/runs/{run_id}")
                # vprint(f"[RUN] {label}: considering run '{run_name}' @ {created_at} (id={run_id})")

                # QUICK RUN-DATE PRE-FILTER to reduce noise:
                # run_date_hint = _run_date_fallback(run)
                if run_date_hint and run_date_hint not in target_dates:
                    # print(f"    ↳ SKIPPED: outside window")
                    print(f"      ↳ ❌ SKIPPED: outside window (target: {sorted(target_dates)})")
                    vprint(f"[SKIP] {label}: run date {run_date_hint} outside target window.")
                    continue
                # PRE-FILTER: Check if date already captured
                if run_date_hint in seen_dates_for_label:
                    print(f"      ↳ ❌ SKIPPED: duplicate (already have data for {run_date_hint})")
                    continue


                run_status = (run.get("status") or "").lower()
                run_conclusion = (run.get("conclusion") or "").lower()
                build_success = 1 if run_conclusion == "success" else 0

                def _parse_iso(s: Optional[str]) -> Optional[dt.datetime]:
                    if not s: return None
                    try:
                        return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
                    except Exception:
                        return None

                t_started = _parse_iso(run.get("run_started_at") or run.get("created_at"))
                t_updated = _parse_iso(run.get("updated_at"))
                build_duration_sec = None
                if t_started and t_updated:
                    build_duration_sec = int((t_updated - t_started).total_seconds())

                arts = list_artifacts_for_run(user, repo, run_id, token)

                # ===== ARTIFACT DEBUG =====
                print(f"      Artifacts total: {len(arts)}")
                for a in arts:
                    exp_str = "EXPIRED" if a.get("expired", False) else "valid"
                    print(f"        - {a.get('name', 'unknown')} ({exp_str})")
                # ==========================



                # select coverage+test artifacts (non-expired)
                def _is_selected(a) -> bool:
                    if a.get("expired", False): return False
                    nm = (a.get("name","") or "").lower()
                    return artifact_matches(nm, patterns) or artifact_matches(nm, test_patterns)
                selected = [a for a in arts if _is_selected(a)]
                if not selected:
                    selected = [a for a in arts if not a.get("expired", False)]
                if not selected:
                    if use_pages_fallback and pages_candidates:
                        # Try to parse coverage from GitHub Pages instead
                        p_date, p_lines, p_funcs = fetch_parse_coverage_from_urls(pages_candidates)
                        date_str = _coerce_date_to_window(p_date or run_date_hint, target_dates)

                        if date_str in seen_dates_for_label:
                            print(f"      ↳ ❌ SKIPPED: pages date {date_str} not in target set")
                            continue

                        if not date_str or date_str not in target_dates:
                            print(f"      ↳ ❌ SKIPPED: pages date {date_str} not in target set")
                            continue

                        # if not date_str:
                        #     print(f"[INFO] {label}: no date available for pages coverage; skipping run.")
                        #     continue
                        # if date_str in seen_dates_for_label or date_str not in target_dates:
                        #     print(f"    ↳ SKIPPED: duplicate (older run)")
                        #     vprint(f"[SKIP] {label}: pages date {date_str} not in target set or already captured.")
                        #     continue

                        add_row(date_str, label,
                            (p_lines if p_lines is not None else -1.0),
                            (p_funcs if p_funcs is not None else -1.0),
                            build_status=run_status,
                            build_conclusion=run_conclusion,
                            build_success=build_success,
                            build_duration_sec=build_duration_sec,
                            tests_total=None, tests_passed=None, tests_failures=None, tests_errors=None, tests_skipped=None)
                        # print(f"    ↳ ✓ SELECTED")
                        # print(f"[OK] {label} (pages-fallback) :: {date_str} :: Lines={'NE' if p_lines is None else f'{p_lines:.2f}%'} "
                        #       f":: Functions={'NE' if p_funcs is None else f'{p_funcs:.2f}%'} "
                        #       f":: Build={'✔' if build_success==1 else '✖'} ({run_conclusion or 'unknown'})")
                        print(f"      ↳ ✅ SELECTED (pages-fallback)")
                        print(f"         Date: {date_str}")
                        print(f"         Lines: {'NE' if p_lines is None else f'{p_lines:.2f}%'}")
                        print(f"         Functions: {'NE' if p_funcs is None else f'{p_funcs:.2f}%'}")

                        seen_dates_for_label.add(date_str)
                        processed += 1
                        if seen_dates_for_label >= target_dates:
                            break
                        continue

                    print(f"      ↳ ❌ SKIPPED: no usable artifacts")
                    print(f"[INFO] {label}: no usable artifacts in this run (id={run_id}). Skipping run.")
                    continue

                print(f"      Using artifacts: {', '.join([a.get('name','') for a in selected])}")
                print(f"[ARTIFACT] {label}: using {len(selected)} artifact(s): " + ", ".join([a.get('name','') for a in selected]))

                temp_root = Path(tempfile.mkdtemp(prefix="cov_zip_all_"))
                try:
                    # extract
                    for a in selected:
                        try:
                            zip_bytes = req_get(
                                f"https://api.github.com/repos/{user}/{repo}/actions/artifacts/{a['id']}/zip",
                                gh_headers(token), timeout=60, allow_redirects=True
                            )
                            zip_bytes.raise_for_status()
                            with zipfile.ZipFile(io.BytesIO(zip_bytes.content)) as zf:
                                subdir = temp_root / safe_label(a.get("name","artifact"))
                                subdir.mkdir(parents=True, exist_ok=True)
                                zf.extractall(subdir)
                        except Exception as ex:
                            print(f"[WARN] {label}: failed to extract artifact '{a.get('name','')}' ({a.get('id')}): {ex}")

                    index_html = find_file(temp_root, "index.html")
                    date_str = None
                    if index_html:
                        date_str = parse_date_from_index(index_html)
                    if not date_str:
                        date_str = _run_date_fallback(run)
                        if date_str:
                            vprint(f"[INFO] {label}: using run date fallback {date_str} (no usable date in index.html).")
                        else:
                            print(f"[INFO] {label}: no usable date (index.html absent or unparsable, and no run date). Skipping run.")
                            continue

                    if date_str in seen_dates_for_label:
                        vprint(f"[SKIP] {label}: already captured date {date_str}; skipping duplicate run.")
                        continue
                    if date_str not in target_dates:
                        vprint(f"[SKIP] {label}: parsed date {date_str} not in target date set.")
                        continue

                    # coverage
                    lcov_path = find_file(temp_root, "lcov.info") or Path("nope")
                    lines_pct, funcs_pct = parse_lcov_info(lcov_path)
                    if lines_pct is None or funcs_pct is None:
                        l2, f2 = parse_lines_funcs_from_index(index_html or Path("nope"))
                        if lines_pct is None: lines_pct = l2
                        if funcs_pct is None: funcs_pct = f2
                        # If still unresolved, try pages as a final fallback
                        if (lines_pct is None or funcs_pct is None) and use_pages_fallback and pages_candidates:
                            p_date, p_lines, p_funcs = fetch_parse_coverage_from_urls(pages_candidates)
                            # Keep the run's date (for grid alignment), but fill missing metrics
                            if lines_pct is None: lines_pct = p_lines
                            if funcs_pct is None: funcs_pct = p_funcs
                    if lines_pct is None: lines_pct = -1.0
                    if funcs_pct is None: funcs_pct = -1.0

                    # tests
                    junit_files, ctest_files = detect_test_files(str(temp_root))
                    print(f"[DEBUG] {label} {date_str}: JUnit files={len(junit_files)}, CTest files={len(ctest_files)}")
                    #DEBUG
                    if junit_files or ctest_files:
                        print(f"[TEST-SOURCE] Tests came from workflow '{run_name}'")    
                        print(f"[TEST-SOURCE] Tests came from workflow '{run_name}' (Run ID {run_id})")



                    if not junit_files and not ctest_files:
                        tests_total = {"total": None, "passed": None, "failures": None, "errors": None, "skipped": None}
                    else:
                        tests_junit = parse_junit(junit_files)
                        tests_ctest = parse_ctest(ctest_files)
                        tests_total = {
                            "total":   tests_junit.get("total",0)    + tests_ctest.get("total",0),
                            "passed":  tests_junit.get("passed",0)   + tests_ctest.get("passed",0),
                            "failures":tests_junit.get("failures",0) + tests_ctest.get("failures",0),
                            "errors":  tests_junit.get("errors",0)   + tests_ctest.get("errors",0),
                            "skipped": tests_junit.get("skipped",0)  + tests_ctest.get("skipped",0),
                        }

                    add_row(date_str, label, lines_pct, funcs_pct,
                            build_status=run_status,
                            build_conclusion=run_conclusion,
                            build_success=build_success,
                            build_duration_sec=build_duration_sec,
                            tests_total=tests_total["total"],
                            tests_passed=tests_total["passed"],
                            tests_failures=tests_total["failures"],
                            tests_errors=tests_total["errors"],
                            tests_skipped=tests_total["skipped"])

                    print(f"[OK] {label} :: {date_str} :: Lines={'NE' if lines_pct<0 else f'{lines_pct:.2f}%'} "
                          f":: Functions={'NE' if funcs_pct<0 else f'{funcs_pct:.2f}%'} "
                          f":: Tests T/P/F/E/S = "
                          f"{tests_total['total'] if tests_total['total'] is not None else 'NA'}/"
                          f"{tests_total['passed'] if tests_total['passed'] is not None else 'NA'}/"
                          f"{tests_total['failures'] if tests_total['failures'] is not None else 'NA'}/"
                          f"{tests_total['errors'] if tests_total['errors'] is not None else 'NA'}/"
                          f"{tests_total['skipped'] if tests_total['skipped'] is not None else 'NA'} "
                          f":: Build={'✔' if build_success==1 else '✖'} ({run_conclusion or 'unknown'})")

                    print(f"      ↳ ✅ SELECTED")
                    print(f"         Date: {date_str}")
                    print(f"         Lines: {'NE' if lines_pct < 0 else f'{lines_pct:.2f}%'}")
                    print(f"         Functions: {'NE' if funcs_pct < 0 else f'{funcs_pct:.2f}%'}")
                    print(f"         Tests: T={tests_total.get('total', 'NA')} P={tests_total.get('passed', 'NA')} F={tests_total.get('failures', 'NA')}")
                    print(f"         Build: {'✔ Success' if build_success == 1 else f'✖ {run_conclusion}'}")

                    seen_dates_for_label.add(date_str)
                    processed += 1
                    # Stop once we've filled all target dates for this repo/label
                    if seen_dates_for_label >= target_dates:
                        break

                finally:
                    shutil.rmtree(temp_root, ignore_errors=True)

        

        except requests.HTTPError as ex:
            body = ""
            try:
                body = f" - {ex.response.text[:200]}"
            except Exception:
                pass
            print(f"[ERROR] {label}: GitHub API HTTP {getattr(ex.response,'status_code', '???')}{body}")
        except Exception as ex:
            print(f"[ERROR] {label}: API ingestion failed: {ex}")

    # ---------- Normalize CSV, dedupe, and build window ----------
    if not HISTORY_CSV.exists():
        print("[INFO] No CSV produced; nothing to plot.")
        return

    df = pd.read_csv(HISTORY_CSV)
    if "label" not in df.columns and "repo" in df.columns:
        df = df.rename(columns={"repo":"label"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["lines_pct"] = pd.to_numeric(df["lines_pct"], errors="coerce")
    df["functions_pct"] = pd.to_numeric(df["functions_pct"], errors="coerce")
    if "build_success" in df.columns:
        df["build_success"] = pd.to_numeric(df["build_success"], errors="coerce")
    if "build_duration_sec" in df.columns:
        df["build_duration_sec"] = pd.to_numeric(df["build_duration_sec"], errors="coerce")
    for col in ["tests_total","tests_passed","tests_failures","tests_errors","tests_skipped"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date","label"])
    df = (df.sort_values(["label","date"])
            .drop_duplicates(subset=["label","date"], keep="last"))
    df.to_csv(HISTORY_CSV, index=False)

    all_dates = sorted(df["date"].unique())
    if not all_dates:
        anchor = dt.date.today()
        calendar_days = [anchor - dt.timedelta(days=i) for i in range(NUM_CAL_DAYS)][::-1]
    else:
        calendar_days = build_calendar_window(all_dates)
    
    labels = sorted(df["label"].unique())
    
    # Labels from the current config (so we always show them, even if no rows)
    labels_cfg = set()
    for item in config:
        if not item.get("enabled", False):
            continue
        user  = (item.get("user") or "").strip()
        repo  = (item.get("repo") or "").strip()
        level = (item.get("test_level") or "").strip() or "L1"
        if user and repo:
            labels_cfg.add(f"{user}/{repo} [{level}]")

    labels = sorted(labels_cfg.union(labels_cfg))


    # Build daymaps
    daymaps = build_matrix_and_daymaps(labels, calendar_days, df)

    # HTML
    write_combined_execution_grid_html(labels, calendar_days, daymaps, EXECUTION_ALL_HTML)

    # PDF
    try:
        build_pdf_from_csv(
            csv_path=HISTORY_CSV,
            output_pdf=EXECUTION_ALL_PDF,
            num_days=NUM_CAL_DAYS,
            anchor_mode=ANCHOR_MODE,
            custom_anchor_date=CUSTOM_ANCHOR_DATE,
            title="Execution Grid — All Repos / Levels",
            include_status_in_grid=True,
            details_only_executed=True
        )
    except Exception as ex:
        print(f"[WARN] PDF generation failed: {ex}")

if __name__ == "__main__":
    main()