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
import model_disp as MD

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


def disp_track(pm, dp, season):
    res, _ = MD.walk_forward(pm, dp)
    cur = res[res["season"].astype(str) == season]
    if cur.empty:
        return None
    d = pm.sort_values("date").copy()
    pl = d.groupby("playerId")
    n = pl.cumcount()
    prev = (pl["disposals"].cumsum() - d["disposals"]) / n.replace(0, np.nan)
    grp = d.groupby("pos_group")
    pos = (grp["disposals"].cumsum() - d["disposals"]) / grp.cumcount().replace(0, np.nan)
    d["naive"], d["pos"] = prev.fillna(pos).fillna(11.0), pos.fillna(11.0)
    b = d[d["season"].astype(str) == season]
    rows = []
    for L in MD.LINES:
        rows.append(pd.DataFrame({"p": MD.p_at_least(cur["d_mu"], L, cur["d_size"]), "hit": (cur["disposals"] >= L).astype(int)}))
    a = pd.concat(rows)
    a["bucket"] = pd.cut(a["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], include_lowest=True)
    cal = a.groupby("bucket", observed=True).agg(n=("hit", "size"), pred=("p", "mean"), act=("hit", "mean")).reset_index()
    return {
        "season": season, "n": int(len(cur)),
        "mae": {"model": r(np.mean(np.abs(cur["d_mu"] - cur["disposals"])), 2),
                "naive": r(np.mean(np.abs(b["naive"] - b["disposals"])), 2),
                "position": r(np.mean(np.abs(b["pos"] - b["disposals"])), 2)},
        "within3": r(np.mean(np.abs(cur["d_mu"] - cur["disposals"]) <= 3), 3),
        "calib": [{"lo": float(max(iv.left, 0)), "hi": float(iv.right), "n": int(n), "pred": r(pr), "act": r(ac)}
                  for iv, n, pr, ac in zip(cal["bucket"], cal["n"], cal["pred"], cal["act"])],
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
                             "p1": r(x.p1, 4), "p2": r(x.p2, 4), "p3": r(x.p3, 4),
                             "dm": r(x.d_mu, 2), "dr": r(x.d_size, 1), "dtog": r(x.d_tog, 0)} for x in g.itertuples()],
            })

    data["track"] = track_record(pm, p, season)
    dpf = os.path.join(HERE, "disp_params.json")
    dp = MD.DParams(**json.load(open(dpf))) if os.path.exists(dpf) else MD.DParams()
    data["settings"]["d_size"] = dp.nb_size
    data["settings"]["g_size"] = p.nb_size
    # points per shot (current + previous season), to turn expected shots into an expected score
    seasons = sorted(pm["season"].astype(str).unique())[-2:]
    done = mt[(mt["status"] == "CONCLUDED") & (mt["season"].astype(str).isin(seasons))]
    sh = pm[pm["mid"].isin(done["mid"])]["shots"].sum()
    data["settings"]["pts_per_shot"] = r((done["hs"].sum() + done["as"].sum()) / sh, 3) if sh else 2.9
    data["disp_track"] = disp_track(pm, dp, season)

    os.makedirs(SITE, exist_ok=True)
    tpl = open(os.path.join(HERE, "site_template.html"), encoding="utf-8").read()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", blob)
    open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(html)
    json.dump(data, open(os.path.join(SITE, "data.json"), "w"), indent=1)
    print(f"Built site for {season} round {rnd}: {len(data['matches'])} games -> {SITE}/index.html")


if __name__ == "__main__":
    build()
