"""Offline tests for firms_watch. Run: py -m unittest -v"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import firms_watch as fw

# Shaped like real FIRMS area-API responses (column order as FIRMS publishes them).
VIIRS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
-0.12345,37.30001,330.5,0.4,0.37,2026-11-08,1042,N,VIIRS,n,2.0NRT,295.1,4.2,D
-0.20000,37.40000,367.0,0.4,0.37,2026-11-08,1042,N,VIIRS,h,2.0NRT,300.2,18.9,D
0.10000,37.10000,301.0,0.4,0.37,2026-11-08,45,N,VIIRS,l,2.0NRT,290.0,1.1,N
"""
MODIS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight
-0.3000,37.5000,320.1,1.0,1.0,2026-11-08,0750,T,MODIS,85,6.1NRT,290.0,22.0,D
-0.3500,37.5500,310.0,1.0,1.0,2026-11-08,0750,T,MODIS,20,6.1NRT,289.0,5.0,D
"""
HEADER_ONLY = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"


def make_config(tmp: Path, **overrides) -> Path:
    raw = {"area_name": "Test forest", "bbox": [36.95, -0.55, 37.75, 0.25],
           "sources": ["VIIRS_SNPP_NRT", "MODIS_NRT"], "db_path": "t.sqlite"}
    raw.update(overrides)
    path = tmp / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


class ParseTests(unittest.TestCase):
    def test_viirs_rows(self):
        rows = fw.parse_firms_csv(VIIRS_CSV, "VIIRS_SNPP_NRT")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["lat"], -0.1235)  # rounded to 4 dp for de-duplication
        self.assertEqual(rows[0]["acquired_utc"], "2026-11-08T10:42:00+00:00")
        self.assertEqual([r["confidence"] for r in rows], ["nominal", "high", "low"])
        self.assertEqual(rows[1]["frp_mw"], 18.9)

    def test_short_acq_time_is_zero_padded(self):
        rows = fw.parse_firms_csv(VIIRS_CSV, "VIIRS_SNPP_NRT")
        self.assertEqual(rows[2]["acquired_utc"], "2026-11-08T00:45:00+00:00")

    def test_modis_numeric_confidence(self):
        rows = fw.parse_firms_csv(MODIS_CSV, "MODIS_NRT")
        self.assertEqual([r["confidence"] for r in rows], ["high", "low"])

    def test_confidence_boundaries(self):
        self.assertEqual(fw.confidence_class("MODIS_NRT", "29"), "low")
        self.assertEqual(fw.confidence_class("MODIS_NRT", "30"), "nominal")
        self.assertEqual(fw.confidence_class("MODIS_NRT", "79"), "nominal")
        self.assertEqual(fw.confidence_class("MODIS_NRT", "80"), "high")
        self.assertEqual(fw.confidence_class("MODIS_NRT", ""), "unknown")

    def test_empty_and_header_only(self):
        self.assertEqual(fw.parse_firms_csv("", "MODIS_NRT"), [])
        self.assertEqual(fw.parse_firms_csv(HEADER_ONLY, "VIIRS_SNPP_NRT"), [])

    def test_plain_text_error_raises(self):
        with self.assertRaises(RuntimeError):
            fw.parse_firms_csv("Invalid MAP_KEY.", "MODIS_NRT")

    def test_malformed_row_is_skipped(self):
        bad = HEADER_ONLY + "notanumber,37.1,300,0.4,0.4,2026-11-08,1000,N,VIIRS,n,2,290,1,D\n" + VIIRS_CSV.splitlines()[1]
        self.assertEqual(len(fw.parse_firms_csv(bad, "VIIRS_SNPP_NRT")), 1)


class ConfigTests(unittest.TestCase):
    def test_rejects_bad_bbox_source_and_days(self):
        with tempfile.TemporaryDirectory() as t:
            for override in ({"bbox": [37.75, -0.55, 36.95, 0.25]}, {"sources": ["VIIRS"]}, {"day_range": 7}):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    fw.Config.load(make_config(Path(t), **override))

    def test_area_string_and_relative_db(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = fw.Config.load(make_config(Path(t)))
            self.assertEqual(cfg.area, "36.95,-0.55,37.75,0.25")
            self.assertEqual(Path(cfg.db_path).parent, Path(t))


class PollTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = fw.Config.load(make_config(Path(self.tmp.name)))
        self.db = fw.connect(self.cfg.db_path)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def fake(self, responses):
        def fetcher(cfg, source, key):
            r = responses[source]
            if isinstance(r, Exception):
                raise r
            return r
        return fetcher

    def poll(self, responses):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            new, self.failed = fw.poll_once(self.cfg, self.db, "KEY", fetcher=self.fake(responses))
        return new, out.getvalue(), err.getvalue()

    def test_new_detections_stored_once(self):
        responses = {"VIIRS_SNPP_NRT": VIIRS_CSV, "MODIS_NRT": MODIS_CSV}
        new, out, _ = self.poll(responses)
        self.assertEqual(len(new), 5)
        self.assertIn("5 new FIRMS detection(s) in Test forest", out)
        self.assertIn("worldview.earthdata.nasa.gov", out)
        again, out2, _ = self.poll(responses)  # FIRMS re-sends the same day's data
        self.assertEqual(again, [])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM detections").fetchone()[0], 5)

    def test_one_source_failing_does_not_lose_the_other(self):
        new, _, err = self.poll({"VIIRS_SNPP_NRT": RuntimeError("could not reach FIRMS"), "MODIS_NRT": MODIS_CSV})
        self.assertEqual(len(new), 2)
        self.assertIn("could not reach FIRMS", err)
        failed = self.db.execute("SELECT COUNT(*) FROM polls WHERE ok = 0").fetchone()[0]
        self.assertEqual((failed, self.failed), (1, 1))

    def test_report_false_share_by_confidence(self):
        self.poll({"VIIRS_SNPP_NRT": VIIRS_CSV, "MODIS_NRT": MODIS_CSV})
        ids = {r["confidence_raw"]: r["id"] for r in self.db.execute("SELECT id, confidence_raw FROM detections")}
        for raw, label in {"h": "real", "85": "real", "n": "false", "l": "false", "20": "false"}.items():
            self.db.execute("UPDATE detections SET label = ? WHERE id = ?", (label, ids[raw]))
        self.db.commit()
        r = fw.build_report(self.db)
        self.assertEqual((r["detections"], r["real"], r["false"]), (5, 2, 3))
        self.assertAlmostEqual(r["false_share"], 0.6)
        self.assertEqual(r["by_confidence"]["high"]["false_share"], 0.0)
        self.assertEqual(r["by_confidence"]["low"]["false_share"], 1.0)
        self.assertIsNotNone(r["latency_hours_median"])


class FetchTests(unittest.TestCase):
    """fetch() against a fake urlopen, so no network is used."""

    def run_fetch(self, errors):
        import urllib.error
        calls = self.calls = []

        def fake_urlopen(req, timeout):
            calls.append(req.full_url)
            err = errors[min(len(calls) - 1, len(errors) - 1)]
            if err is None:
                return io.BytesIO(HEADER_ONLY.encode())
            if isinstance(err, int):
                raise urllib.error.HTTPError(req.full_url, err, "err", {}, io.BytesIO(b"Invalid MAP_KEY."))
            raise err

        cfg = fw.Config("A", (36.95, -0.55, 37.75, 0.25), ["MODIS_NRT"])
        orig_open, orig_sleep = fw.urllib.request.urlopen, fw.time.sleep
        fw.urllib.request.urlopen, fw.time.sleep = fake_urlopen, lambda s: None
        try:
            return fw.fetch(cfg, "MODIS_NRT", "KEY"), calls
        finally:
            fw.urllib.request.urlopen, fw.time.sleep = orig_open, orig_sleep

    def test_client_error_fails_fast_with_server_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_fetch([400])
        self.assertIn("Invalid MAP_KEY", str(ctx.exception))
        self.assertIn("HTTP 400", str(ctx.exception))

    def test_client_error_is_not_retried(self):
        with self.assertRaises(RuntimeError):
            self.run_fetch([403, None])  # a retry would succeed, so one call proves no retry
        self.assertEqual(len(self.calls), 1)

    def test_network_errors_give_up_after_retries(self):
        import urllib.error
        with self.assertRaises(RuntimeError) as ctx:
            self.run_fetch([urllib.error.URLError("no route")])
        self.assertEqual(len(self.calls), 3)  # first try + 2 retries
        self.assertIn("could not reach FIRMS", str(ctx.exception))

    def test_server_error_then_success_is_retried(self):
        text, calls = self.run_fetch([503, None])
        self.assertEqual(len(calls), 2)
        self.assertIn("latitude", text)
        self.assertIn("/MODIS_NRT/36.95,-0.55,37.75,0.25/1", calls[0])


class CliTests(unittest.TestCase):
    def test_label_list_report_and_missing_key(self):
        with tempfile.TemporaryDirectory() as t:
            cfgp = make_config(Path(t))
            cfg = fw.Config.load(cfgp)
            db = fw.connect(cfg.db_path)
            fw.store(db, fw.parse_firms_csv(MODIS_CSV, "MODIS_NRT"), "2026-11-08T12:00:00+00:00")
            db.close()
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                self.assertEqual(fw.main(["--config", str(cfgp), "label", "1", "false", "--note", "gas flare"]), 0)
                self.assertEqual(fw.main(["--config", str(cfgp), "label", "99", "real"]), 1)
                self.assertEqual(fw.main(["--config", str(cfgp), "list"]), 0)
                self.assertEqual(fw.main(["--config", str(cfgp), "report"]), 0)
            text = out.getvalue()
            self.assertIn("#1 labelled false", text)
            self.assertIn("gas flare", text)
            self.assertIn("Not enough labelled detections", text)
            self.assertIn("No detection #99", err.getvalue())

            import os
            saved = os.environ.pop("FIRMS_MAP_KEY", None)
            try:
                with redirect_stderr(io.StringIO()) as err2:
                    self.assertEqual(fw.main(["--config", str(cfgp), "poll"]), 2)
                self.assertIn("FIRMS_MAP_KEY", err2.getvalue())
            finally:
                if saved is not None:
                    os.environ["FIRMS_MAP_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
