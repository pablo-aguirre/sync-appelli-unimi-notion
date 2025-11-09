import os
import json
import time
import math
import random
import re
import unicodedata
from datetime import datetime
from typing import Dict, Any, Optional

import requests
import pandas as pd

# --- Config ambiente ---
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATA_SOURCE_ID = os.environ["DATA_SOURCE_ID"]

API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}

# Override tipi (cdl e descrIns sono select; descrIns verrà sanificato prima dell'invio)
COL_OVERRIDES: Dict[str, str] = {
    "dataStr": "date",
    "aperturaStr": "date",
    "chiusuraStr": "date",
    "ora": "rich_text",
    "idAppello": "rich_text",
    "cdl": "select",
    "descrIns": "select",
    "codIns": "select",
}

# Formati per le date
DATE_FORMATS: Dict[str, str] = {
    "dataStr": "%d/%m/%Y",
    "aperturaStr": "%d/%m/%Y",
    "chiusuraStr": "%d/%m/%Y",
}

# Pattern per inferenza date
_DATE_PATTERNS = [
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y", False),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d", False),
    (re.compile(r"^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}$"), "%d/%m/%Y %H:%M", True),
    (re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"), "%Y-%m-%dT%H:%M", True),
    (re.compile(r"^\d{2}:\d{2}$"), "%H:%M", "time-only"),
]


# Sanitizzazione label per select (per descrIns)
def sanitize_select_label(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    # rimuovi tutta la punteggiatura Unicode
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    # collassa spazi multipli
    s = " ".join(s.split())
    return s


# HTTP con backoff
def req(method: str, url: str, **kw) -> requests.Response:
    for i in range(6):
        r = requests.request(method, url, timeout=60, **kw)
        if r.status_code != 429 and r.status_code < 500:
            return r
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else (2 ** i) + random.random()
        time.sleep(min(wait, 20))
    r.raise_for_status()
    return r


# Notion: schema
def retrieve_data_source(ds_id: str) -> Dict[str, Any]:
    r = req("GET", f"{API_BASE}/data_sources/{ds_id}", headers=HEADERS)
    r.raise_for_status()
    return r.json()


def update_data_source_properties(ds_id: str, props: Dict[str, Any]) -> None:
    if not props:
        return
    body = {"properties": props}
    r = req("PATCH", f"{API_BASE}/data_sources/{ds_id}", headers=HEADERS, data=json.dumps(body))
    if r.status_code >= 300:
        raise RuntimeError(r.text)


def schema_for(ptype: str) -> Dict[str, Any]:
    # Tipi gestiti: rich_text, number, checkbox, date, select, multi_select
    return {ptype: {}}


# Inferenza tipi
def _detect_date_format(sample_strings):
    counts = {}
    for s in sample_strings:
        s = s.strip()
        for rx, fmt, has_time in _DATE_PATTERNS:
            if rx.match(s):
                counts[(fmt, has_time)] = counts.get((fmt, has_time), 0) + 1
                break
    if not counts:
        return (None, False)
    (fmt, has_time), best = max(counts.items(), key=lambda kv: kv[1])
    if best >= (len(sample_strings) / 2):
        if fmt == "%H:%M":
            return (None, False)
        return (fmt, has_time)
    return (None, False)


def infer_property_type(series: pd.Series, name: str) -> str:
    if name in COL_OVERRIDES:
        return COL_OVERRIDES[name]
    if series.dropna().map(lambda x: isinstance(x, (bool,))).all():
        return "checkbox"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    sample = series.dropna().astype(str).head(50).tolist()
    if sample:
        fmt, _ = _detect_date_format(sample)
        if fmt:
            DATE_FORMATS.setdefault(name, fmt)
            return "date"
    return "rich_text"


def ensure_properties_from_df(df: pd.DataFrame, ds_id: str) -> Dict[str, str]:
    ds = retrieve_data_source(ds_id)
    existing = ds.get("properties", {})  # {name: {type:...}}
    desired: Dict[str, str] = {}
    for col in df.columns:
        if col == "Name":
            continue
        desired[col] = infer_property_type(df[col], col)
    if "External ID" not in existing:
        desired["External ID"] = "rich_text"

    to_add, to_update = {}, {}
    for col, ptype in desired.items():
        if col not in existing:
            to_add[col] = schema_for(ptype)
        else:
            current = existing[col].get("type")
            if current != ptype:
                to_update[col] = schema_for(ptype)

    payload = {**to_add, **to_update}
    if payload:
        update_data_source_properties(ds_id, payload)
        existing = retrieve_data_source(ds_id).get("properties", {})

    final_types: Dict[str, str] = {}
    for col in desired:
        final_types[col] = existing[col]["type"] if col in existing else desired[col]
    return final_types


# Conversione valori verso Notion
def to_notion_value(value: Any, ptype: str, colname: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if ptype == "number":
        try:
            return {"number": float(value)}
        except Exception:
            return {"number": None}
    if ptype == "checkbox":
        return {"checkbox": bool(value)}
    if ptype == "date":
        fmt = DATE_FORMATS.get(colname) if colname else None
        if fmt:
            try:
                dt = datetime.strptime(str(value).strip(), fmt)
                return {"date": {"start": dt.isoformat() if "%H" in fmt else dt.date().isoformat()}}
            except Exception:
                return None
        dt = pd.to_datetime(str(value), errors="coerce")
        if pd.isna(dt):
            return None
        return {"date": {"start": pd.Timestamp(dt).isoformat()}}
    if ptype == "select":
        name = str(value)
        if colname == "descrIns":
            name = sanitize_select_label(name)
            if not name:
                return None
        return {"select": {"name": name}}
    if ptype == "multi_select":
        if isinstance(value, (list, tuple, set)):
            opts = [{"name": str(v)} for v in value if v is not None and not (isinstance(v, float) and math.isnan(v))]
        else:
            opts = [{"name": str(value)}]
        return {"multi_select": opts}
    return {"rich_text": [{"text": {"content": str(value)[:2000]}}]}


# Costruzione properties per riga
def build_properties_for_row(row: pd.Series, prop_types: Dict[str, str], cdl_value: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    # Title obbligatorio: usa descrIns originale per leggibilità
    title = str(row.get("descrIns") or "Senza nome")
    props["Name"] = {"title": [{"text": {"content": title[:2000]}}]}
    # External ID per upsert
    ext = row.get("idAppello")
    if pd.notna(ext):
        props["External ID"] = {"rich_text": [{"text": {"content": str(ext)}}]}
    # cdl come select
    props["cdl"] = {"select": {"name": cdl_value}}
    # Altre proprietà
    for col, ptype in prop_types.items():
        if col in ("Name", "External ID", "cdl"):
            continue
        v = row.get(col)
        nv = to_notion_value(v, ptype, colname=col)
        if nv is not None:
            props[col] = nv
    return props


# Notion: upsert
def find_page_by_external_id(ds_id: str, ext_id: str) -> Optional[str]:
    url = f"{API_BASE}/data_sources/{ds_id}/query"
    body = {"page_size": 1, "filter": {"property": "External ID", "rich_text": {"equals": str(ext_id)}}}
    r = req("POST", url, headers=HEADERS, data=json.dumps(body))
    r.raise_for_status()
    js = r.json()
    res = js.get("results") or []
    return res[0]["id"] if res else None


def create_page(ds_id: str, properties: Dict[str, Any]) -> None:
    body = {"parent": {"type": "data_source_id", "data_source_id": ds_id}, "properties": properties}
    r = req("POST", f"{API_BASE}/pages", headers=HEADERS, data=json.dumps(body))
    if r.status_code >= 300:
        raise RuntimeError(r.text)


def update_page(page_id: str, properties: Dict[str, Any]) -> None:
    r = req("PATCH", f"{API_BASE}/pages/{page_id}", headers=HEADERS, data=json.dumps({"properties": properties}))
    if r.status_code >= 300:
        raise RuntimeError(r.text)


# Download + normalizzazione json
def download_json(course_code: str) -> Dict[str, Any]:
    code = course_code.strip().upper()
    url = f"https://work.unimi.it/foProssimiEsami/json/{code}"
    print(f"Scarico: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def normalize_json_to_df(json_data: Dict[str, Any]) -> pd.DataFrame:
    return pd.json_normalize(
        json_data,
        record_path="appelli",
        meta=["codIns", "codW4", "descrIns", "descrInsEng"],
        sep="_",
        errors="ignore",
    )


# Pipeline principale
def push_course_appelli_to_notion(course_code: str) -> None:
    data = download_json(course_code)
    df = normalize_json_to_df(data)
    if df.empty:
        print("Nessun appello trovato.")
        return
    cdl_value = course_code.strip().upper()
    df["cdl"] = cdl_value

    prop_types = ensure_properties_from_df(df, DATA_SOURCE_ID)
    created = updated = 0
    for _, row in df.iterrows():
        props = build_properties_for_row(row, prop_types, cdl_value)
        ext = row.get("idAppello")
        page_id = find_page_by_external_id(DATA_SOURCE_ID, str(ext)) if pd.notna(ext) else None
        if page_id:
            update_page(page_id, props)
            updated += 1
        else:
            create_page(DATA_SOURCE_ID, props)
            created += 1
        time.sleep(0.35)  # ~3 rps
    print(f"Created: {created} | Updated: {updated}")


if __name__ == "__main__":
    cdl_codes = [
        'F94',  # Informatica Magistrale, vecchio manifesto
        'FBA',  # Informatica Magistrale, nuovo manifesto
        'F1X',  # Informatica Triennale, vecchio manifesto
        'FAA',  # Informatica Triennale, nuovo manifesto
        'FAD',  # Sicurezza Informatica, nuovo manifesto
        'F68',  # Sicurezza Informatica, vecchio manifesto
    ]

    for code in cdl_codes:
        push_course_appelli_to_notion(code)
