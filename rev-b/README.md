# WildFire Rev B: FIRMS-only fire watcher

The baseline from the WildFire design log (question 3). It polls NASA FIRMS, which
detects fires from heat seen by the MODIS and VIIRS satellites, for one area and alerts
on new detections. There's no ML model. Python standard library only, so there's nothing
to install.

The point of Rev B is to **measure**. After about 30 days of polling and labelling,
`report` answers one question: are FIRMS false alarms common enough in this area to
justify building a classifier (Rev C)?

## Setup

1. Get a free FIRMS MAP_KEY: <https://firms.modaps.eosdis.nasa.gov/api/map_key/>
2. Copy the config and set your area as `[west, south, east, north]` in degrees:
   ```
   copy config.example.json config.json
   ```
   The example box covers the Mt Kenya forest. Replace it with your own area.
3. Set the key for the current shell (PowerShell):
   ```
   $env:FIRMS_MAP_KEY = "your-key"
   ```

## Use

| Command | What it does |
|---|---|
| `py firms_watch.py poll` | Fetch new detections once and print an alert for each |
| `py firms_watch.py poll --loop` | Keep polling every `poll_hours` (default 3) |
| `py firms_watch.py list --unlabeled` | Show detections you haven't checked yet |
| `py firms_watch.py label 12 false --note "gas flare"` | Record what a detection turned out to be (`real`, `false` or `unsure`) |
| `py firms_watch.py report` | False-alarm share overall and by FIRMS confidence, plus alert delay |

Each alert includes a NASA Worldview link and a Google Maps link, so you can check the
spot before labelling it.

To run it unattended on Windows, schedule `py firms_watch.py poll` every 3 hours in Task
Scheduler instead of leaving `--loop` running. `poll` exits with code 1 when every source
failed (bad key or no network), so the scheduler's history shows broken runs.

### Email alerts (optional)

Set `"email"` in `config.json`, and put the SMTP password in the `FIRMS_SMTP_PASSWORD`
environment variable, never in the file:

```json
"email": {"smtp_host": "smtp.gmail.com", "smtp_port": 587,
          "from": "you@example.com", "to": ["ranger@example.com"]}
```

## How the decision is made

`report` follows the kill criteria from the design log:

- **Fewer than 30 labelled detections:** keep watching.
- **False alarms under 10%:** stop. FIRMS alone is the product, so skip Rev C.
- **Otherwise:** first check the by-confidence table. If ignoring `low` confidence
  detections already removes most false alarms, that one-line filter beats any classifier.

## Tests

```
py -m unittest -v
```

The tests run offline against FIRMS-shaped sample data and a fake HTTP layer.

## Data notes

- Detections are stored once each, by source, position (4 decimal places) and pass
  time. FIRMS re-sends the same day's data on every call, and the duplicates are ignored.
- Confidence is put on one scale: VIIRS reports `l/n/h`, MODIS reports 0–100
  (under 30 is low, 80 and over is high).
- "Alert delay" is the time from the satellite pass to the first poll that saw the
  detection. With 3-hour polling, part of that delay comes from the polling interval itself.
