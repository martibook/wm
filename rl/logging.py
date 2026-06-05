"""JSONL run logging (config / metrics / eval / games) per docs/logging.md."""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class RunLogger:
    def __init__(self, run_dir, run_id=None):
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(run_dir) / self.run_id
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._metrics = open(self.dir / "metrics.jsonl", "a")
        self._eval = open(self.dir / "eval.jsonl", "a")
        self._games = open(self.dir / "games.jsonl", "a")

    def write_config(self, cfg: dict) -> None:
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    def _w(self, f, rec) -> None:
        f.write(json.dumps(rec) + "\n")
        f.flush()

    def log_metrics(self, rec):
        self._w(self._metrics, rec)

    def log_eval(self, rec):
        self._w(self._eval, rec)

    def log_games(self, recs):
        for r in recs:
            self._w(self._games, r)

    def close(self):
        for f in (self._metrics, self._eval, self._games):
            f.close()
