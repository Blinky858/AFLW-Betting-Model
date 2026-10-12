"""
Build the website: site/index.html (one self-contained file).

    python build_site.py

Runs the model forward through every completed game, predicts the next round, works out the
model's track record for the current season, and bakes it all into site_template.html.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fair_odds import compute_round
from model import walk_forward, logloss

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def track_record(pm, p, season):
    res, _ = walk_forward(pm, p)
    cur = res[res["season"].astype(str) == season].copy()
    if cur.empty:
        return None
    cur["hit"] = (cur["goals"] >= 1).astype(int)

    # baselines, computed walk-forward on the same games
    d = pm.sort_values("date").copy()
    d["scored"] = (d["goals"] >= 1).astype(int)
    grp = d.groupby("pos_group")
    d["pos_rate"] = ((grp["scored"].cumsum() - d["scored"]) / grp.cumcount().replace(0, np.nan)).fillna(0.15)
    pl = d.groupby("playerId")
    n_prev = pl.cumcount()
    gpg = (pl["goals"].cumsum() - d["goals"]) / n_prev.replace(0, np.nan)
    d["naive"] = np.where(n_prev >= 1, 1 - np.exp(-gpg.fillna(0)), d["pos_rate"])
    b = d[d["season"].astype(str) == season]

    bins = [0, .05, .1, .15, .2, .3, .4, .5, .6, .8, 1]
    cur["bucket"] = pd.cut(cur["p1"], bins, include_lowest=True)
    cal = cur.groupby("bucket", observed=True).agg(n=("hit", "size"), pred=("p1", "mean"), act=("hit", "mean")).reset_index()
    calib = [{"lo": float(iv.left if iv.left > 0 else 0), "hi": float(iv.right), "n": int(n), "pred": r(pr), "act": r(ac)}
             for iv, n, pr, ac in zip(cal["bucket"], cal["n"], cal["pred"], cal["act"])]

    # last completed round: the model's top-rated players and whether they kicked one
    last_round = int(cur["round"].max())
    lr = cur[cur["round"] == last_round].sort_values("p1", ascending=False)
    top = lr.head(20)
    last = {
        "round": last_round,
        "top": [{"n": x.name, "t": x.team, "o": x.opp, "p1": r(x.p1), "g": int(x.goals)} for x in top.itertuples()],
        "top_expected": r(top["p1"].sum(), 1),
        "top_actual": int((top["goals"] >= 1).sum()),
        "round_expected": r(lr["p1"].sum(), 1),
        "round_actual": int((lr["goals"] >= 1).sum()),
    }
    return {
        "season": season,
        "n": int(len(cur)),
        "games": int(cur["mid"].nunique()),
        "logloss": {"model": r(logloss(cur["p1"], cur["hit"]), 4),
                    "position": r(logloss(b["pos_rate"], b["scored"]), 4),
                    "naive": r(logloss(b["naive"], b["scored"]), 4)},
        "mean_pred": r(cur["lam"].mean()), "mean_act": r(cur["goals"].mean()),
        "calib": calib,
        "last": last,
    }


def build():
    res, rnd, up, p, pm = compute_round()
    mt = pd.read_csv(os.path.join(HERE, "data", "matches.csv"))
    season = str(pm["season"].astype(str).max())
    data = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": season, "round": rnd, "matches": [], "track": None,
            "settings": {"min_edge": 0.10, "kelly_fraction": 0.25, "max_stake_pct": 0.02}}

    if res is not None:
        meta = mt.set_index("mid")
        for m in up.sort_values("date").itertuples():
            name = f"{m.home} v {m.away}"
            g = res[res["match"] == name].sort_values("lam", ascending=False)
            if g.empty:
                continue
            row = meta.loc[m.mid] if m.mid in meta.index else None
            teams = {}
            for t in (m.home, m.away):
                tg = g[g["team"] == t]
                teams[t] = {"named": bool((tg["lineup_source"] == "named").all()) if len(tg) else False,
                            "xs": r(tg["team_exp_shots"].iloc[0], 1) if len(tg) else None}
            data["matches"].append({
                "id": m.mid, "home": m.home, "away": m.away, "kickoff": str(m.date),
                "venue": None if row is None else (row.get("venue") if isinstance(row.get("venue"), str) else None),
                "weather": None if row is None else (row.get("wdesc") if isinstance(row.get("wdesc"), str) else None),
                "teams": teams,
                "players": [{"n": x.name, "t": x.team, "pos": x.position, "gp": int(x.games_hist),
                             "xs": r(x.exp_shots, 2), "acc": r(x.acc), "lam": r(x.lam),
                             "p1": r(x.p1, 4), "p2": r(x.p2, 4), "p3": r(x.p3, 4)} for x in g.itertuples()],
            })

    data["track"] = track_record(pm, p, season)

    os.makedirs(SITE, exist_ok=True)
    tpl = open(os.path.join(HERE, "site_template.html"), encoding="utf-8").read()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", blob)
    open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(html)
    json.dump(data, open(os.path.join(SITE, "data.json"), "w"), indent=1)
    print(f"Built site for {season} round {rnd}: {len(data['matches'])} games -> {SITE}/index.html")


if __name__ == "__main__":
    build()
