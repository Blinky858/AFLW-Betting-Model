"""
Tune the disposal model on 2023-2025 and test on the current season.

    python backtest_disp.py            # tune + test, writes disp_params.json and output/disp_*
    python backtest_disp.py --no-tune
"""
import argparse
import json
import os
import random
from dataclasses import asdict

import numpy as np
import pandas as pd

from model import load
from model_disp import DParams, walk_forward, nll, p_at_least, LINES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TUNE = ["2023", "2024", "2025"]


def sub(res, seasons):
    return res[res["season"].astype(str).isin(seasons)]


def best_r(mu, y, grid=(4, 6, 8, 10, 12, 15, 20, 30, 50)):
    s = [(nll(mu, y, r), r) for r in grid]
    return min(s)[1], min(s)[0]


def baselines(pm, seasons):
    d = pm.sort_values("date").copy()
    pl = d.groupby("playerId")
    n = pl.cumcount()
    prev = (pl["disposals"].cumsum() - d["disposals"]) / n.replace(0, np.nan)
    grp = d.groupby("pos_group")
    pos = (grp["disposals"].cumsum() - d["disposals"]) / grp.cumcount().replace(0, np.nan)
    d["naive"] = prev.fillna(pos).fillna(11.0)
    d["pos"] = pos.fillna(11.0)
    b = d[d["season"].astype(str).isin(seasons)]
    out = {}
    for c in ("naive", "pos"):
        r, s = best_r(b[c], b["disposals"])
        out[c] = {"nll": s, "mae": float(np.mean(np.abs(b[c] - b["disposals"]))), "nb_size": r}
    return out


def calibration(res, seasons, r):
    c = sub(res, seasons)
    rows = []
    for L in LINES:
        pr = p_at_least(c["d_mu"], L, r)
        rows.append(pd.DataFrame({"line": L, "p": pr, "hit": (c["disposals"] >= L).astype(int)}))
    a = pd.concat(rows)
    a["bucket"] = pd.cut(a["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], include_lowest=True)
    t = a.groupby("bucket", observed=True).agg(n=("hit", "size"), pred=("p", "mean"), act=("hit", "mean")).reset_index()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pm, _ = load()
    test = [str(pm["season"].astype(str).max())]
    pf = os.path.join(HERE, "disp_params.json")
    if a.no_tune and os.path.exists(pf):
        p = DParams(**json.load(open(pf)))
    else:
        rng = random.Random(7)
        space = {"player_decay": [0.85, 0.9, 0.93, 0.96], "season_carry": [0.4, 0.6, 0.8],
                 "k_rate": [1, 2, 3, 5, 8], "k_tog": [1, 2, 4, 8], "team_decay": [0.85, 0.9, 0.95],
                 "k_team": [2, 4, 8], "g_opp": [0, 0.5, 1.0], "g_team": [0, 0.5, 1.0]}
        cands = [DParams()] + [DParams(**{k: rng.choice(v) for k, v in space.items()}) for _ in range(a.iters)]
        best = (1e9, None)
        for i, p in enumerate(cands):
            res, _ = walk_forward(pm, p)
            t = sub(res, TUNE)
            r, s = best_r(t["d_mu"], t["disposals"])
            p.nb_size = r
            if s < best[0]:
                best = (s, p)
            print(f"[{i+1}/{len(cands)}] nll={s:.5f} best={best[0]:.5f}")
        p = best[1]
        json.dump(asdict(p), open(pf, "w"), indent=2)
    print("Params:", asdict(p))
    res, _ = walk_forward(pm, p)
    rep = {"params": asdict(p)}
    for name, seas in (("tune", TUNE), ("test", test)):
        s = sub(res, seas)
        rep[name] = {"model": {"nll": nll(s["d_mu"], s["disposals"], p.nb_size),
                               "mae": float(np.mean(np.abs(s["d_mu"] - s["disposals"]))),
                               "mean_pred": float(s["d_mu"].mean()), "mean_act": float(s["disposals"].mean())},
                     "baselines": baselines(pm, seas)}
    cal = calibration(res, test, p.nb_size)
    cal.to_csv(os.path.join(OUT, "disp_calibration.csv"), index=False)
    json.dump(rep, open(os.path.join(OUT, "disp_report.json"), "w"), indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
    print(cal.to_string(index=False))


if __name__ == "__main__":
    main()
