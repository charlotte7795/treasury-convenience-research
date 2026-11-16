"""Archive the Swedish Debt Office's original quarter-end conventional stock reports."""

import calendar
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from lxml import html
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1] / "data/official/stress_supply_2026-11-03"
BASE = "https://www.riksgalden.se/en/press-and-publications/publications/government-debt/"


def main():
    out = ROOT / "sweden_reports"
    reports = {}
    for archive in sorted(out.glob("archive_*.html")):
        page = html.fromstring(archive.read_bytes())
        for box in page.xpath('//div[contains(@class,"news-block__news__item__container")]'):
            heading = " ".join(box.xpath(".//h2/text()")).strip()
            match = re.search(r"(March|June|September|December) (202[1-6])$", heading)
            if not match:
                continue
            month = list(calendar.month_name).index(match[1])
            year = int(match[2])
            period = dt.date(year, month, calendar.monthrange(year, month)[1])
            if not dt.date(2021, 12, 31) <= period <= dt.date(2026, 6, 30):
                continue
            urls = box.xpath('.//a[contains(@href,".pdf")]/@href')
            dates = box.xpath('.//span[contains(@class,"__date")]/text()')
            if len(urls) == 1 and dates:
                reports[period] = (urljoin(BASE, urls[0]), dt.datetime.strptime(dates[0].strip(), "%d %B %Y").date())
        if len(reports) == 19:
            break
    if "--manifest" in sys.argv:
        print(json.dumps({str(period): url for period, (url, _) in reports.items()}))
        return
    rows = []
    for period, (url, published) in sorted(reports.items()):
        file = out / f"{period}.pdf"
        if not file.exists():
            print("Missing PDF", period)
            continue
        hits = set()
        for page in PdfReader(file).pages:
            for line in page.extract_text(extraction_mode="layout").splitlines():
                match = re.fullmatch(r"\s*Government bonds\s{3,}(\d{1,3}(?:[ \u00a0]\d{3}){2,})\s*", line)
                if match:
                    hits.add(int(re.sub(r"\s", "", match[1])))
        if len(hits) == 1:
            rows.append((period, published, hits.pop() / 1e9, url))
        else:
            print("Review", period, hits)
    with (ROOT / "sweden_conventional_stock.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quarter_end", "public_date", "outstanding_lcu_bn", "source"])
        writer.writerows(rows)
    print(f"Sweden: {len(rows)} of 19 quarter-end reports parsed.")


if __name__ == "__main__":
    main()
