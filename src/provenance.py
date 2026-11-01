"""Record what a result was produced under.

A results directory accumulates one subdirectory per arm, and arms are usually
run over several sessions. Without this, a table could silently mix an arm
trained for 40 epochs with one trained for 2 during a smoke test, and nothing in
the output would show it. The fingerprint makes that visible: figures.py warns
when the arms it is tabulating were not all run under the same settings.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone

# Only the settings that affect comparability. results_dir or verbosity
# differing between runs is harmless and should not flag the table.
COMPARABILITY_KEYS = (
    "city", "patch", "bands", "depth", "base_filters", "dropout",
    "batch_size", "epochs", "patience", "learning_rate", "seed",
)


def fingerprint(cfg: dict) -> dict:
    subset = {k: cfg.get(k) for k in COMPARABILITY_KEYS}
    digest = hashlib.sha256(
        json.dumps(subset, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "hash": digest,
        "settings": subset,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }
