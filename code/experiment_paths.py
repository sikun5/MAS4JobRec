"""Paths and source receipts for the frozen v2 experiments."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def phase(number):
    return ROOT / "runtime" / "runs" / f"v2-phase{number}-20261121"


# In addition to each stage's own scripts, pin the shared implementations it uses.
SHARED = {
    4: ("review_windows", "phase2_data_prep", "phase3_probe"),
    5: (),
    6: (),
    7: ("phase2_data_prep", "phase3_probe", "phase3_awq", "review_windows"),
    8: ("review_windows", "phase2_data_prep", "phase3_probe", "phase5_calibration", "phase5_io",
        "phase5_ranking", "phase5_verify", "phase6_score"),
}


def source_hashes(number):
    sources = set(CODE.glob(f"phase{number}_*.py"))
    sources.update(CODE / f"{name}.py" for name in
                   ("experiment_paths", "experiment_handoff", "launch_stage", *SHARED[number]))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)}


def lock_sources(number):
    """Pin code before the first stage, and reject source drift on resume."""
    run = phase(number)
    if (run / f"phase{number}.sha256").exists():
        raise SystemExit(f"Sealed run: {run}. Use a separate replay checkout.")
    receipt = run / "evidence" / "source_hashes.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    current = source_hashes(number)
    if not receipt.exists():
        # Publish a complete file exactly once, even for concurrent launchers.
        with tempfile.NamedTemporaryFile(dir=receipt.parent, mode="w", delete=False) as tmp:
            json.dump(current, tmp, indent=2)
            tmp.write("\n")
        try:
            os.link(tmp.name, receipt)
        except FileExistsError:
            pass
        finally:
            Path(tmp.name).unlink()
    if json.loads(receipt.read_text()) != current:
        raise SystemExit(f"Sources differ from {receipt}; use a separate run.")
