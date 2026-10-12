"""
Tune the model on 2023-2025 and test it out-of-sample on 2026.

    python backtest.py            # tune + test, writes output/ and params.json
    python backtest.py --no-tune  # just test with the saved params.json
"""
import argparse
import json
import os
import random
from dataclasses import asdict

import numpy as np
import pandas as pd

from model import Params, load, walk_forward, logloss, goal_nll, p_at_least

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TUNE_SEASONS = ["2023", "2024", "2025"]   # 2022 is burn-in only
TEST_SEASONS = ["2026"]


def score(res, seasons, nb):
    r = res[res["season"].astype(str).isin(seasons)]
    y1 = (r["goals"] >= 1).astype(int)
    return {
        "n": len(r),
        "goal_nll": goal_nll(r["lam"], r["goals"], nb),
        "logloss_1plus": logloss(r["p1"], y1),
        "logloss_2plus": logloss(r["p2"], (r["goals"] >= 2).astype(int)),
        "brier_1plus": float(np.mean((r["p1"] - y1) ** 2)),
        "mean_pred_goals": float(r["lam"].mean()),
        "mean_actual_goals": float(r["goals"].mean()),
    }


def baselines(pm, seasons):
    """Two naive comparisons: position base rate, and raw career goals-per-game (no shrinkage)."""
    pm = pm.copy()
    pm["scored"] = (pm["goals"] >= 1).astype(int)
    # position base rate from all prior games (expanding, by date)
    pm = pm.sort_values("date")
    grp = pm.groupby("pos_group")
    pm["pos_rate"] = (grp["scored"].cumsum() - pm["scored"]) / grp.cumcount().replace(0, np.nan)
    pm["pos_rate"] = pm["pos_rate"].fillna(0.15)
    pl = pm.groupby("playerId")
    prior_games = pl.cumcount()
    prior_goals = pl["goals"].cumsum() - pm["goals"]
    gpg = (prior_goals / prior_games.replace(0, np.nan))
    pm["naive_p1"] = np.where(prior_games >= 1, 1 - np.exp(-gpg.fillna(0)), pm["pos_rate"])
    r = pm[pm["season"].astype(str).isin(seasons)]
    return {
        "position_base_rate": logloss(r["pos_rate"], r["scored"]),
        "naive_goals_per_game": logloss(r["naive_p1"], r["scored"]),
    }


def calibration(res, seasons, col="p1", k=1, bins=(0, .05, .1, .15, .2, .3, .4, .5, .6, .8, 1)):
    r = res[res["season"].astype(str).isin(seasons)].copy()
    r["hit"] = (r["goals"] >= k).astype(int)
    r["bucket"] = pd.cut(r[col], bins, include_lowest=True)
    t = r.groupby("bucket", observed=True).agg(n=("hit", "size"), predicted=(col, "mean"), actual=("hit", "mean"))
    return t.reset_index()


def tune(pm, n_iter=40, seed=1):
    rng = random.Random(seed)
    space = {
        "player_decay": [0.8, 0.85, 0.9, 0.93, 0.96],
        "team_decay": [0.8, 0.85, 0.9, 0.95],
        "season_carry": [0.3, 0.5, 0.7, 0.9],
        "k_share": [1, 2, 3, 4, 6, 9],
        "k_team": [2, 4, 8, 15],
        "k_acc": [10, 20, 35, 60, 100],
        "share_power": [1.0, 1.1, 1.2],
    }
    best, best_p, log = 1e9, None, []
    cands = [Params()] + [Params(**{k: rng.choice(v) for k, v in space.items()}) for _ in range(n_iter)]
    for i, p in enumerate(cands):
        p.nb_size = 1e9  # tune structure under Poisson, then fit dispersion
        res, _ = walk_forward(pm, p)
        s = score(res, TUNE_SEASONS, p.nb_size)["goal_nll"]
        log.append({**asdict(p), "goal_nll": s})
        if s < best:
            best, best_p = s, p
        print(f"[{i+1}/{len(cands)}] nll={s:.5f} best={best:.5f}")
    # dispersion
    res, _ = walk_forward(pm, best_p)
    best_nb, best_s = 1e9, best
    for nb in [2, 3, 5, 8, 12, 20, 40, 1e9]:
        s = goal_nll(res[res["season"].astype(str).isin(TUNE_SEASONS)]["lam"],
                     res[res["season"].astype(str).isin(TUNE_SEASONS)]["goals"], nb)
        if s < best_s - 1e-9:
            best_nb, best_s = nb, s
    best_p.nb_size = best_nb
    pd.DataFrame(log).sort_values("goal_nll").to_csv(os.path.join(OUT, "tuning_log.csv"), index=False)
    return best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pm, _ = load()
    pfile = os.path.join(HERE, "params.json")
    if a.no_tune and os.path.exists(pfile):
        p = Params(**json.load(open(pfile)))
    else:
        p = tune(pm, a.iters)
        json.dump(asdict(p), open(pfile, "w"), indent=2)
    print("Params:", asdict(p))

    res, _ = walk_forward(pm, p)
    res.to_csv(os.path.join(OUT, "backtest_predictions.csv"), index=False)

    report = {"params": asdict(p)}
    for name, seas in (("tune_2023_2025", TUNE_SEASONS), ("test_2026", TEST_SEASONS)):
        report[name] = {"model": score(res, seas, p.nb_size), "baselines_logloss_1plus": baselines(pm, seas)}
    cal1 = calibration(res, TEST_SEASONS, "p1", 1)
    cal2 = calibration(res, TEST_SEASONS, "p2", 2, bins=(0, .02, .05, .1, .15, .2, .3, .5, 1))
    cal1.to_csv(os.path.join(OUT, "calibration_1plus_2026.csv"), index=False)
    cal2.to_csv(os.path.join(OUT, "calibration_2plus_2026.csv"), index=False)
    json.dump(report, open(os.path.join(OUT, "backtest_report.json"), "w"), indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print("\nCalibration, anytime goalkicker, 2026 (out of sample):")
    print(cal1.to_string(index=False))
    print("\nCalibration, 2+ goals, 2026:")
    print(cal2.to_string(index=False))


if __name__ == "__main__":
    main()
