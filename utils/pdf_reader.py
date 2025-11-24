from typing import List, Dict
import os
import re
import logging
import pdfplumber
from config.config import PDF_DIR

logger = logging.getLogger(__name__)
_TIME_PAT = re.compile(r"(\d{1,2}[:.]\d{2})(/\d{1,2}[:.]\d{2})?$")

def _normalize_cell(c):
    return (c or "").strip().replace("\n", " ").replace("\r", " ").strip()

def _find_header_indices(header_row: List[str]):
    mapping = {}
    for i, h in enumerate(header_row):
        if not h:
            continue
        text = h.lower()
        if "from" in text:
            mapping["from"] = i
        elif "to" in text and "via" not in text:
            mapping["to"] = i
        elif "via" in text:
            mapping["via"] = i
        elif "service" in text or "type" in text:
            mapping["service"] = i
        elif "dep" in text or "time" in text or "departure" in text:
            mapping["dep_time"] = i
    return mapping

def _row_from_table(row: List[str], header_map: Dict[str,int], default_station: str, source_pdf: str) -> Dict:
    # Safeguard index lookups
    def get_col(key):
        idx = header_map.get(key)
        if idx is None or idx >= len(row):
            return ""
        return _normalize_cell(row[idx])
    parsed = {
        "raw": " | ".join([_normalize_cell(c) for c in row]),
        "from": get_col("from"),
        "to": get_col("to"),
        "via": get_col("via") or "",
        "service": get_col("service") or "",
        "dep_time": get_col("dep_time") or "",
        "station": default_station,
        "source_pdf": source_pdf
    }
    return parsed

def parse_pdf_with_tables(pdf_path: str, station_name: str) -> List[Dict]:
    rows = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try tables extraction
                tables = page.extract_tables()
                if not tables:
                    # fallback: try extracting simple table-like lines from text
                    text = page.extract_text() or ""
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if re.match(r"^\d+\s+", ln):
                            parsed = _parse_plain_line(ln, station_name, pdf_path)
                            rows.append(parsed)
                    continue
                # For each table captured, try to deduce header and parse rows
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [ _normalize_cell(c) for c in table[0] ]
                    header_map = _find_header_indices(header)
                    # If header_map missing 'to' or 'dep_time', still attempt best-effort
                    for r in table[1:]:
                        # r might be shorter/longer; normalize to strings
                        parsed = _row_from_table(r, header_map, station_name, pdf_path)
                        # Only include if we get at least a 'to' or 'from' or dep_time
                        if parsed["to"] or parsed["from"] or parsed["dep_time"]:
                            rows.append(parsed)
    except Exception as e:
        logger.exception("pdfplumber failed on %s: %s", pdf_path, e)
    return rows

def _parse_plain_line(line: str, station_name: str, source_pdf: str) -> Dict:
    """
    Fallback parser for lines starting with serial number.
    Heuristic similar to prior parse_bus_row but more conservative for fields.
    """
    raw = line.strip()
    m = re.match(r"^\d+\s+(.*)$", raw)
    core = m.group(1).strip() if m else raw
    # split by two+ spaces where table spacing likely present
    parts = re.split(r"\s{2,}", core)
    # If parts long enough, take first few as from/to/via/service/time heuristics
    from_col = to_col = via_col = service_col = dep_col = ""
    if len(parts) >= 4:
        # common layout: FROM  TO  VIA  SERVICE DEP.TIME
        from_col = _normalize_cell(parts[0])
        to_col = _normalize_cell(parts[1])
        # combine subsequent columns except last as VIA or service candidate
        middle = [ _normalize_cell(p) for p in parts[2:-1] ]
        # last part possibly includes service and time, try to extract time
        last = _normalize_cell(parts[-1])
        # extract time if present
        time_match = _TIME_PAT.search(last)
        if time_match:
            dep_col = time_match.group(1)
            # anything preceding time in last part is candidate for service
            service_col = last[:time_match.start()].strip()
        # via could be middle joined
        via_col = " | ".join(middle).strip()
    else:
        # fallback splitting by whitespace and searching for time token
        toks = core.split()
        idx_time = None
        for i in range(len(toks)-1, -1, -1):
            if _TIME_PAT.match(toks[i]):
                idx_time = i
                break
        dep_col = toks[idx_time] if idx_time is not None else ""
        if idx_time and idx_time >= 2:
            from_col = toks[0]
            to_col = toks[1]
            if idx_time > 2:
                via_col = " ".join(toks[2:idx_time])
            # service maybe tokens before time
            service_col = " ".join(toks[max(2, idx_time-2):idx_time]).strip()
        else:
            # coarse fallback
            parts2 = re.split(r"\s{1,}", core)
            from_col = parts2[0] if parts2 else ""
            to_col = parts2[1] if len(parts2)>1 else ""
            via_col = " ".join(parts2[2:]) if len(parts2)>2 else ""
    parsed = {
        "raw": raw,
        "from": from_col.strip().strip(","),
        "to": to_col.strip().strip(","),
        "via": via_col.strip().strip(","),
        "service": service_col.strip().strip(","),
        "dep_time": dep_col.strip(),
        "station": station_name,
        "source_pdf": source_pdf
    }
    return parsed

def parse_pdf_to_chunks(pdf_path: str, station_name: str) -> List[Dict]:
    """
    Public function to parse one PDF path into normalized rows.
    It first tries table parsing; if no table rows found, it uses fallback.
    """
    rows = parse_pdf_with_tables(pdf_path, station_name)
    if not rows:
        # fallback: parse entire file text for serial-number lines
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if re.match(r"^\d+\s+", ln):
                            rows.append(_parse_plain_line(ln, station_name, pdf_path))
        except Exception as e:
            logger.exception("Fallback parse failed for %s: %s", pdf_path, e)
    return rows