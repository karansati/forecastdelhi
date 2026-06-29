"""Scrape the current Delhi system load from the Delhi SLDC website.

Delhi SLDC publishes 5-minute system load on
    https://www.delhisldc.org/Loaddata.aspx?mode=DD/MM/YYYY
in an HTML table whose columns are roughly:
    TIMESLOT | DELHI | BRPL | BYPL | NDPL/TPDDL | NDMC | MES

We fetch today's page and read the latest row that has a valid DELHI value.
There is no official API, so this scraper depends on the page layout; if Delhi
SLDC changes its site, adjust the parsing in `parse_latest` below. A clear error
is raised (rather than a silent wrong number) when the value cannot be found.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

import config

IST = ZoneInfo(config.TIMEZONE)
HEADERS = {"User-Agent": "Mozilla/5.0 (load-forecast bot; contact: you@example.com)"}


def _to_float(s):
    try:
        return float(str(s).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def parse_latest(html):
    """Return (timeslot_str, delhi_load_mw) from the Loaddata table, latest valid row."""
    soup = BeautifulSoup(html, "html.parser")
    # the data grid id has been stable as ContentPlaceHolder3_DGGrid; fall back to
    # any table whose header mentions DELHI.
    table = soup.find("table", id=lambda x: x and "DGGrid" in x)
    if table is None:
        for tbl in soup.find_all("table"):
            if "DELHI" in tbl.get_text().upper():
                table = tbl
                break
    if table is None:
        raise RuntimeError("Could not locate the SLDC load table on the page.")

    rows = table.find_all("tr")
    last = None
    for tr in rows[1:]:                       # skip header
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        slot, delhi = cells[0], _to_float(cells[1])
        if delhi and delhi > 0:
            last = (slot, delhi)
    if last is None:
        raise RuntimeError("No valid DELHI load value found in the SLDC table yet.")
    return last


def fetch_load():
    """Return dict with the current Delhi system load (MW) and its timeslot."""
    today = datetime.now(IST).strftime("%d/%m/%Y")
    r = requests.get(config.SLDC_LOADDATA_URL, params={"mode": today},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    slot, load_mw = parse_latest(r.text)
    return {"Load": round(load_mw, 2), "slot": slot, "date": today}


if __name__ == "__main__":
    print(fetch_load())
