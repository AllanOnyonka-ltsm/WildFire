"""WildFire Rev B: a FIRMS-only fire watcher. No model, no third-party packages.

Polls NASA FIRMS for active-fire detections inside one bounding box, stores each
detection once in SQLite, alerts on new ones, and lets you label them real or false
so that `report` can answer the Rev B question: are FIRMS false alarms common enough
in this area to justify building a classifier (Rev C)?

Usage:
    py firms_watch.py poll              # one poll (schedule it every 3 h)
    py firms_watch.py poll --loop       # poll forever, every poll_hours
    py firms_watch.py list [--unlabeled]
    py firms_watch.py label 12 real     # or: false, unsure   [--note "gas flare"]
    py firms_watch.py report

Needs a free FIRMS MAP_KEY in the FIRMS_MAP_KEY environment variable:
https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import smtplib
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}"
VALID_SOURCES = {
    "MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "LANDSAT_NRT",
}
LABELS = ("real", "false", "unsure")
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- config

@dataclass
class Config:
    area_name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    sources: list[str]
    day_range: int = 1
    poll_hours: float = 3.0
    db_path: str = "detections.sqlite"
    email: dict | None = None

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        w, s, e, n = (float(v) for v in raw["bbox"])
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError("bbox must be [west, south, east, north] with west < east and south < north")
        sources = list(raw.get("sources", ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]))
        bad = [x for x in sources if x not in VALID_SOURCES]
        if bad:
            raise ValueError(f"unknown FIRMS sources: {bad}; valid: {sorted(VALID_SOURCES)}")
        day_range = int(raw.get("day_range", 1))
        if not 1 <= day_range <= 5:
            raise ValueError("day_range must be 1-5 (FIRMS API limit)")
        db_path = raw.get("db_path", "detections.sqlite")
        if not Path(db_path).is_absolute():
            db_path = str(path.parent / db_path)
        return cls(raw.get("area_name", "Unnamed area"), (w, s, e, n), sources, day_range,
                   float(raw.get("poll_hours", 3)), db_path, raw.get("email"))

    @property
    def area(self) -> str:
        return ",".join(f"{v:g}" for v in self.bbox)


# ---------------------------------------------------------------- parsing

def confidence_class(source: str, raw: str) -> str:
    """Put MODIS (0-100) and VIIRS/Landsat (l/n/h) confidence on one scale."""
    raw = (raw or "").strip().lower()
    if raw in ("l", "low"):
        return "low"
    if raw in ("n", "nominal"):
        return "nominal"
    if raw in ("h", "high"):
        return "high"
    try:
        value = float(raw)
    except ValueError:
        return "unknown"
    # FIRMS MODIS guidance: <30 low, 30-79 nominal, >=80 high
    return "low" if value < 30 else "high" if value >= 80 else "nominal"


def parse_firms_csv(text: str, source: str) -> list[dict]:
    """Turn a FIRMS area-API CSV response into detection dicts.

    FIRMS answers errors (bad key, bad source) with a plain-text line instead of CSV,
    so anything without a latitude header is treated as an error.
    """
    text = text.strip()
    if not text:
        return []
    first_line = text.splitlines()[0].lower()
    if "latitude" not in first_line:
        raise RuntimeError(f"FIRMS returned an error for {source}: {text[:200]}")
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
            acq_time = (r.get("acq_time") or "0").strip().zfill(4)
            acq = datetime.strptime(f'{r["acq_date"].strip()} {acq_time}', "%Y-%m-%d %H%M")
        except (KeyError, ValueError):
            continue  # skip a malformed row instead of losing the whole poll
        rows.append({
            "source": source,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "acquired_utc": acq.replace(tzinfo=timezone.utc).isoformat(),
            "confidence_raw": (r.get("confidence") or "").strip(),
            "confidence": confidence_class(source, r.get("confidence", "")),
            "frp_mw": _float_or_none(r.get("frp")),
            "daynight": (r.get("daynight") or "").strip(),
        })
    return rows


def _float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id             INTEGER PRIMARY KEY,
    source         TEXT NOT NULL,
    lat            REAL NOT NULL,
    lon            REAL NOT NULL,
    acquired_utc   TEXT NOT NULL,
    first_seen_utc TEXT NOT NULL,
    confidence_raw TEXT,
    confidence     TEXT,
    frp_mw         REAL,
    daynight       TEXT,
    label          TEXT CHECK (label IN ('real', 'false', 'unsure')),
    note           TEXT,
    UNIQUE (source, lat, lon, acquired_utc)
);
CREATE TABLE IF NOT EXISTS polls (
    id         INTEGER PRIMARY KEY,
    polled_utc TEXT NOT NULL,
    source     TEXT NOT NULL,
    ok         INTEGER NOT NULL,
    fetched    INTEGER NOT NULL,
    new        INTEGER NOT NULL,
    error      TEXT
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def store(db: sqlite3.Connection, rows: list[dict], seen_at: str) -> list[sqlite3.Row]:
    """Insert detections not seen before; return the new ones."""
    new_ids = []
    for r in rows:
        cur = db.execute(
            """INSERT OR IGNORE INTO detections
               (source, lat, lon, acquired_utc, first_seen_utc, confidence_raw, confidence, frp_mw, daynight)
               VALUES (:source, :lat, :lon, :acquired_utc, :seen, :confidence_raw, :confidence, :frp_mw, :daynight)""",
            {**r, "seen": seen_at},
        )
        if cur.rowcount:
            new_ids.append(cur.lastrowid)
    db.commit()
    if not new_ids:
        return []
    marks = ",".join("?" * len(new_ids))
    return db.execute(f"SELECT * FROM detections WHERE id IN ({marks}) ORDER BY acquired_utc", new_ids).fetchall()


# ---------------------------------------------------------------- polling

def fetch(cfg: Config, source: str, key: str, retries: int = 2) -> str:
    url = API.format(key=key, source=source, area=cfg.area, days=cfg.day_range)
    req = urllib.request.Request(url, headers={"User-Agent": "wildfire-revb/1.0"})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace").strip()[:200].rstrip(".")
            if e.code < 500:  # bad key, bad source, bad area: retrying won't help
                raise RuntimeError(f"FIRMS rejected the request for {source} (HTTP {e.code}): "
                                   f"{detail or e.reason}. Check FIRMS_MAP_KEY and config.json.") from None
            last = f"HTTP {e.code} {detail}"
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        if attempt < retries:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not reach FIRMS for {source}: {last}")


def map_links(d: sqlite3.Row) -> str:
    date = d["acquired_utc"][:10]
    pad = 0.05
    view = f"{d['lon'] - pad:.4f},{d['lat'] - pad:.4f},{d['lon'] + pad:.4f},{d['lat'] + pad:.4f}"
    return (f"https://worldview.earthdata.nasa.gov/?v={view}&t={date}  "
            f"https://www.google.com/maps?q={d['lat']},{d['lon']}")


def format_alert(cfg: Config, new: list[sqlite3.Row]) -> str:
    lines = [f"{len(new)} new FIRMS detection(s) in {cfg.area_name}:", ""]
    for d in new:
        lines.append(f"#{d['id']}  {d['acquired_utc'][:16].replace('T', ' ')} UTC  {d['lat']}, {d['lon']}  "
                     f"{d['source']}  confidence {d['confidence']}  FRP {d['frp_mw'] or '?'} MW")
        lines.append(f"     {map_links(d)}")
    lines += ["", "Label each one after checking: py firms_watch.py label <id> real|false|unsure"]
    return "\n".join(lines)


def send_email(cfg: Config, body: str, count: int) -> None:
    e = cfg.email
    password = os.environ.get("FIRMS_SMTP_PASSWORD", "")
    msg = EmailMessage()
    msg["Subject"] = f"[FIRMS] {count} new detection(s) in {cfg.area_name}"
    msg["From"], msg["To"] = e["from"], ", ".join(e["to"])
    msg.set_content(body)
    with smtplib.SMTP(e["smtp_host"], int(e.get("smtp_port", 587)), timeout=30) as s:
        s.starttls()
        if password:
            s.login(e.get("username", e["from"]), password)
        s.send_message(msg)


def poll_once(cfg: Config, db: sqlite3.Connection, key: str, fetcher=fetch) -> tuple[list[sqlite3.Row], int]:
    """Poll every source once. Returns (new detections, number of sources that failed)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_new, failed = [], 0
    for source in cfg.sources:
        try:
            rows = parse_firms_csv(fetcher(cfg, source, key), source)
            new = store(db, rows, now)
            all_new += new
            db.execute("INSERT INTO polls (polled_utc, source, ok, fetched, new) VALUES (?,?,1,?,?)",
                       (now, source, len(rows), len(new)))
        except RuntimeError as e:
            failed += 1
            print(f"[warn] {e}", file=sys.stderr)
            db.execute("INSERT INTO polls (polled_utc, source, ok, fetched, new, error) VALUES (?,?,0,0,0,?)",
                       (now, source, str(e)[:500]))
        db.commit()
    print(f"[{now}] polled {len(cfg.sources)} source(s) ({failed} failed), {len(all_new)} new detection(s)")
    if all_new:
        body = format_alert(cfg, all_new)
        print(body)
        if cfg.email:
            try:
                send_email(cfg, body, len(all_new))
            except (OSError, smtplib.SMTPException, KeyError) as e:
                print(f"[warn] email alert failed: {e}", file=sys.stderr)
    return all_new, failed


# ---------------------------------------------------------------- report

def build_report(db: sqlite3.Connection) -> dict:
    dets = db.execute("SELECT * FROM detections").fetchall()
    polls = db.execute("SELECT * FROM polls").fetchall()
    latencies = []
    for d in dets:
        lag = (datetime.fromisoformat(d["first_seen_utc"]) - datetime.fromisoformat(d["acquired_utc"]))
        latencies.append(lag.total_seconds() / 3600)

    by_conf = {}
    for c in ("high", "nominal", "low", "unknown"):
        group = [d for d in dets if d["confidence"] == c]
        real = sum(d["label"] == "real" for d in group)
        false = sum(d["label"] == "false" for d in group)
        by_conf[c] = {"total": len(group), "real": real, "false": false,
                      "false_share": (false / (real + false)) if real + false else None}

    real = sum(d["label"] == "real" for d in dets)
    false = sum(d["label"] == "false" for d in dets)
    first = min((p["polled_utc"] for p in polls), default=None)
    last = max((p["polled_utc"] for p in polls), default=None)
    days = ((datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds() / 86400) if first else 0
    return {
        "days_watched": round(days, 1),
        "polls": len({p["polled_utc"] for p in polls}),
        "failed_source_polls": sum(not p["ok"] for p in polls),
        "detections": len(dets),
        "labelled": real + false,
        "unlabelled": sum(d["label"] is None for d in dets),
        "real": real,
        "false": false,
        "false_share": (false / (real + false)) if real + false else None,
        "latency_hours_median": round(statistics.median(latencies), 1) if latencies else None,
        "latency_hours_max": round(max(latencies), 1) if latencies else None,
        "by_confidence": by_conf,
    }


def print_report(r: dict) -> None:
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    print(f"Watched {r['days_watched']} days over {r['polls']} polls ({r['failed_source_polls']} failed source calls)")
    print(f"Detections: {r['detections']}  labelled {r['labelled']}  unlabelled {r['unlabelled']}")
    print(f"Labelled real {r['real']}, false {r['false']}  ->  false-alarm share {pct(r['false_share'])}")
    if r["latency_hours_median"] is None:
        print("Time from satellite pass to first seen: n/a (no detections yet)")
    else:
        print(f"Time from satellite pass to first seen: median {r['latency_hours_median']} h, "
              f"max {r['latency_hours_max']} h")
    print("\nBy FIRMS confidence:")
    for c, g in r["by_confidence"].items():
        if g["total"]:
            print(f"  {c:<8} {g['total']:>4} total  {g['real']:>3} real  {g['false']:>3} false  false share {pct(g['false_share'])}")
    print("\nRev B decision (design log, question 5):")
    if r["labelled"] < 30:
        print("  Not enough labelled detections yet to decide. Keep watching and labelling.")
    elif r["false_share"] is not None and r["false_share"] < 0.10:
        print("  False alarms are rare. Stop here: FIRMS alone is the product, skip Rev C.")
    else:
        print("  False alarms are common enough to test Rev C. First check whether simply")
        print("  ignoring low-confidence detections already fixes it (see the table above).")


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="WildFire Rev B: FIRMS-only fire watcher")
    p.add_argument("--config", default=str(HERE / "config.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("poll", help="fetch new detections and alert")
    sp.add_argument("--loop", action="store_true", help="keep polling every poll_hours")
    sl = sub.add_parser("list", help="show stored detections")
    sl.add_argument("--unlabeled", action="store_true")
    sb = sub.add_parser("label", help="mark a detection after checking it")
    sb.add_argument("id", type=int)
    sb.add_argument("label", choices=LABELS)
    sb.add_argument("--note", default=None)
    sub.add_parser("report", help="summarise false alarms and alert delay")
    args = p.parse_args(argv)

    cfg = Config.load(Path(args.config))
    db = connect(cfg.db_path)
    try:
        return run(args, cfg, db)
    finally:
        db.close()


def run(args, cfg: Config, db: sqlite3.Connection) -> int:
    if args.cmd == "poll":
        key = os.environ.get("FIRMS_MAP_KEY", "").strip()
        if not key:
            print("Set FIRMS_MAP_KEY first. Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/",
                  file=sys.stderr)
            return 2
        while True:
            _, failed = poll_once(cfg, db, key)
            if not args.loop:
                # Non-zero only when every source failed, so a scheduler flags a broken key or no network.
                return 1 if failed == len(cfg.sources) else 0
            time.sleep(cfg.poll_hours * 3600)

    if args.cmd == "list":
        q = "SELECT * FROM detections" + (" WHERE label IS NULL" if args.unlabeled else "") + " ORDER BY acquired_utc"
        for d in db.execute(q):
            print(f"#{d['id']:<5} {d['acquired_utc'][:16].replace('T', ' ')}  {d['lat']:>8}, {d['lon']:<9} "
                  f"{d['source']:<17} {d['confidence']:<8} {d['label'] or '-'}  {d['note'] or ''}")
        return 0

    if args.cmd == "label":
        cur = db.execute("UPDATE detections SET label = ?, note = COALESCE(?, note) WHERE id = ?",
                         (args.label, args.note, args.id))
        db.commit()
        if not cur.rowcount:
            print(f"No detection #{args.id}", file=sys.stderr)
            return 1
        print(f"#{args.id} labelled {args.label}")
        return 0

    if args.cmd == "report":
        print_report(build_report(db))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
