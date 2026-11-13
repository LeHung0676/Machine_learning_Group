"""Ghi log, metrics, lưu kết quả thí nghiệm."""
import json
import logging
from datetime import datetime
from pathlib import Path


class Tracker:
    def __init__(self, exp_dir: Path, run_name: str, member: str = "team"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{member}_{run_name}"
        self.dir = Path(exp_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record = {"run_id": self.run_id, "member": member, "params": {}, "metrics": {}}
        self.logger = logging.getLogger(self.run_id)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(fh)
        self.logger.addHandler(logging.StreamHandler())

    def log_params(self, **params):
        self.record["params"].update(params)

    def log_metrics(self, prefix: str = "", **metrics):
        self.record["metrics"].update({f"{prefix}{k}": v for k, v in metrics.items()})
        self.logger.info(f"{prefix} {metrics}")

    def save(self):
        with open(self.dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(self.record, f, indent=2, ensure_ascii=False)
        return self.dir / "result.json"
