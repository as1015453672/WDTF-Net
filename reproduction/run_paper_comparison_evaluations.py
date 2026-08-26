"""Sequentially re-evaluate every locked paper-table checkpoint.

No training code is invoked.  Each subprocess handles exactly one model and
releases GPU memory before the next one starts.  Failures are recorded and the
remaining methods continue, so the final table can explicitly mark N/A rather
than silently substituting a different result.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from paper_comparison_inventory import METHODS, OUT, ROOT


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_status(value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation_workflow_status.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def slug(method: str) -> str:
    # Keep the historical helper's WDTF output directory spelling.
    if method == "WDTF-Net":
        return "wdtfnet"
    return method.lower().replace("-", "_").replace(" ", "_")


def is_done(dataset: str, method: str) -> bool:
    path = OUT / "raw" / dataset.lower() / slug(method) / "metrics_summary.csv"
    if not path.is_file():
        return False
    return f"\n{slug(method)}," in path.read_text(encoding="utf-8-sig")


def main() -> None:
    status = {
        "state": "running", "started": now(), "training_started": False,
        "order": [f"{x['dataset']}/{x['method']}" for x in METHODS if x["method"] != "Mask"],
        "completed": [], "failed": [], "current": None,
    }
    save_status(status)
    for item in METHODS:
        dataset, method = item["dataset"], item["method"]
        if method == "Mask":
            continue
        tag = f"{dataset}/{method}"
        if is_done(dataset, method):
            status["completed"].append({"item": tag, "status": "already_complete"})
            save_status(status)
            continue
        out = OUT / "raw" / dataset.lower() / slug(method)
        out.mkdir(parents=True, exist_ok=True)
        status["current"] = tag
        save_status(status)
        command = [sys.executable, "-u", "paper_compare_single.py", "--dataset", dataset, "--method", method]
        with (out / "batch.stdout.log").open("w", encoding="utf-8") as stdout, (out / "batch.stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        if result.returncode == 0 and is_done(dataset, method):
            status["completed"].append({"item": tag, "status": "evaluated"})
        else:
            status["failed"].append({"item": tag, "returncode": result.returncode,
                                     "stdout": str(out / "batch.stdout.log"), "stderr": str(out / "batch.stderr.log")})
        status["current"] = None
        save_status(status)
    status.update({"state": "complete", "finished": now(), "current": None})
    save_status(status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


