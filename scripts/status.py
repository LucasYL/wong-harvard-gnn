"""Read saved run status without loading molecules or model weights."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "outputs").glob("*/run_started.json")):
    started = json.loads(path.read_text())
    result_path = path.parent / "results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    progress_path = path.parent / "epoch_progress.json"
    epochs = json.loads(progress_path.read_text()) if progress_path.exists() else []
    print(json.dumps({"run_id": started["run_id"], "pilot_only": started["pilot_only"],
                      "complete": result.get("complete", False),
                      "last_recorded_epoch": epochs[-1] if epochs else None,
                      "total_seconds": result.get("total_seconds"),
                      "note": "Incomplete records do not establish that a process is still running."}, ensure_ascii=False))
