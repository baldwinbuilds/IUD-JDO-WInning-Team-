"""Tabulate every blend report (reports/blend_<name>_vN.json) for the final decision:
proxy score after assignment, batch-9 score, unforced test-histogram distance to 600/class, fraction of rows moved
by the assignment, agreement with the best other family, and the submission file (if written)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift.data import REPORTS, SUBS  # noqa: E402


def main():
    rows = []
    for p in sorted(REPORTS.glob("blend_*_v*.json")):
        r = json.load(open(p))
        if "test_hist_pre" not in r:          # report from the old blend script
            continue
        m = re.match(r"blend_(.+)_v(\d+)\.json", p.name)
        name, v = m.group(1), int(m.group(2))
        hist = np.asarray(r["test_hist_pre"], float)
        agree = r.get("agreement", {})
        tb = [a["post"] for k, a in agree.items() if k.startswith("trackB:")]
        nn = [a["post"] for k, a in agree.items() if k.startswith("nn:")]
        w = r["weights"]
        rows.append({
            "candidate": f"{name}_v{v}",
            "score_post": round(r["score_post"], 4),
            "b9_post": round(r["per_fold_post"].get("9", r["per_fold_post"].get(9, float("nan"))), 4),
            "b7_post": round(r["per_fold_post"].get("7", r["per_fold_post"].get(7, float("nan"))), 4),
            "hist_l1_pre": round(float(np.abs(hist - 600).sum() / hist.sum()), 3),
            "moved": round(r["moved"], 3),
            "agree_trackB_max": round(max(tb), 3) if tb else float("nan"),
            "agree_nn_max": round(max(nn), 3) if nn else float("nan"),
            "nn_weight": round(sum(x for k, x in w.items() if k.startswith("nn:")), 2),
            "n_members": len(w),
            "assign": f"{r['assign']}@{r['tau']}",
            "gates": "ok" if not r["problems"] else "FAIL",
            "csv": (SUBS / f"sub_{name}_v{v}.csv").exists(),
        })
    if not rows:
        sys.exit("no blend reports")
    df = pd.DataFrame(rows).set_index("candidate").sort_values(["score_post", "hist_l1_pre"], ascending=[False, True])
    pd.set_option("display.width", 220)
    print(df.to_string())


if __name__ == "__main__":
    main()
