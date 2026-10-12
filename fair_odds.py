"""
Fair goalkicker odds for the next round, and value flags against bookie prices.

    python fetch_data.py --lineups-only   # Thursday/Friday once teams are named
    python fair_odds.py                   # writes output/fair_odds_round_N.csv
    python fair_odds.py --odds my_odds.csv --bank 500

my_odds.csv (fill in from your bookie; any subset of players/markets):
    player,market,odds
    Tayla Harris,1+,2.10
    Tayla Harris,2+,5.50

market is one of 1+, 2+, 3+. Rows are matched to players by name.
"""
import argparse
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from model import Params, load, walk_forward, p_at_least
from model_disp import DParams, walk_forward as disp_walk_forward

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")

MIN_EDGE = 0.10        # only flag bets where model_prob x odds >= 1.10
KELLY_FRACTION = 0.25  # quarter Kelly
MAX_STAKE_PCT = 0.02   # never more than 2% of bank on one bet


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def last_lineups(pm):
    """Each team's most recent 21 as a stand-in until real teams are named."""
    last = pm.sort_values("date").groupby("teamId")["mid"].last()
    rows = []
    for team, mid in last.items():
        g = pm[(pm["mid"] == mid) & (pm["teamId"] == team)]
        rows += [{"teamId": team, "playerId": r.playerId, "position": r.position} for r in g.itertuples()]
    return pd.DataFrame(rows)


def compute_round(rnd=None):
    """Walk the model forward through every completed game, then predict the next round.
    Returns (predictions DataFrame, round number, fixtures DataFrame, params, player-match data)."""
    p = Params(**json.load(open(os.path.join(HERE, "params.json")))) if os.path.exists(os.path.join(HERE, "params.json")) else Params()
    pm, mt = load()
    _, eng = walk_forward(pm, p)
    dpf = os.path.join(HERE, "disp_params.json")
    dp = DParams(**json.load(open(dpf))) if os.path.exists(dpf) else DParams()
    _, deng = disp_walk_forward(pm, dp)

    cur = mt["season"].astype(str).max()
    up = mt[(mt["season"].astype(str) == cur) & (mt["status"] != "CONCLUDED")].copy()
    if up.empty:
        return None, None, up, p, pm
    rnd = rnd or int(up["round"].min())
    up = up[up["round"] == rnd]
    eng.new_season(cur)
    deng.new_season(cur)

    lpath = os.path.join(DATA, "lineups.csv")
    named = pd.read_csv(lpath) if os.path.exists(lpath) else pd.DataFrame(columns=["mid", "teamId", "playerId", "position"])
    fallback = last_lineups(pm)

    out = []
    for m in up.itertuples():
        for team_id, opp_id, team, opp, is_home in ((m.homeId, m.awayId, m.home, m.away, 1), (m.awayId, m.homeId, m.away, m.home, 0)):
            lu = named[(named["mid"] == m.mid) & (named["teamId"] == team_id)]
            src = "named"
            if len(lu) < 16:
                lu, src = fallback[fallback["teamId"] == team_id], "last game (teams not named yet)"
            lineup = list(zip(lu["playerId"], lu["position"]))
            wx = getattr(m, "weather", "") or ""
            df = eng.predict_team(team_id, opp_id, is_home, wx, lineup)
            df = df.merge(deng.predict_team(team_id, opp_id, is_home, wx, lineup), on="playerId", how="left")
            df["d_size"] = dp.nb_size
            df["name"] = df["playerId"].map(eng.pl_name).fillna(
                df["playerId"].map(dict(zip(named.get("playerId", []), named.get("name", [])))))
            df["team"], df["opp"], df["match"], df["lineup_source"] = team, opp, f"{m.home} v {m.away}", src
            df["kickoff_utc"] = m.date
            out.append(df)

    res = pd.concat(out, ignore_index=True)
    for k in (1, 2, 3):
        pr = p_at_least(res["lam"], k, p.nb_size)
        res[f"p{k}"] = pr
        res[f"fair_{k}+"] = np.where(pr > 0, 1 / pr, np.inf)
    res = res.sort_values(["match", "lam"], ascending=[True, False])
    return res, rnd, up, p, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", help="CSV of bookie odds: player,market,odds")
    ap.add_argument("--bank", type=float, default=100.0)
    ap.add_argument("--round", type=int, help="round number (default: next unplayed)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    res, rnd, up, p, pm = compute_round(a.round)
    if res is None:
        print("No upcoming games."); return
    cols = ["match", "team", "name", "position", "games_hist", "exp_shots", "acc", "lam",
            "p1", "fair_1+", "p2", "fair_2+", "p3", "fair_3+", "lineup_source"]
    sheet = res[cols].round({"exp_shots": 2, "acc": 3, "lam": 3, "p1": 3, "p2": 3, "p3": 3,
                             "fair_1+": 2, "fair_2+": 2, "fair_3+": 2})
    path = os.path.join(OUT, f"fair_odds_round_{rnd}.csv")
    sheet.to_csv(path, index=False)
    print(f"Round {rnd} fair odds -> {path}")
    for mname, g in sheet.groupby("match", sort=False):
        print(f"\n{mname}  [{g['lineup_source'].iloc[0]}]")
        print(g[g["p1"] >= 0.15][["team", "name", "position", "lam", "p1", "fair_1+", "fair_2+"]].to_string(index=False))

    if a.odds:
        odds = pd.read_csv(a.odds)
        odds["key"] = odds["player"].map(norm)
        res["key"] = res["name"].map(norm)
        kmap = {"1+": "p1", "2+": "p2", "3+": "p3"}
        bets = []
        for r in odds.itertuples():
            hit = res[res["key"] == r.key]
            if hit.empty:
                print(f"  ! no match for '{r.player}' (check spelling / named team)"); continue
            h = hit.iloc[0]
            prob = float(h[kmap[str(r.market)]])
            ev = prob * r.odds - 1
            b = r.odds - 1
            kelly = max(0.0, (prob * b - (1 - prob)) / b) * KELLY_FRACTION
            stake = min(kelly, MAX_STAKE_PCT) * a.bank
            bets.append({"match": h["match"], "player": h["name"], "market": r.market, "odds": r.odds,
                         "model_prob": round(prob, 3), "fair_odds": round(1 / prob, 2) if prob else None,
                         "edge": round(ev, 3), "bet": ev >= MIN_EDGE, "stake": round(stake, 2) if ev >= MIN_EDGE else 0})
        bets = pd.DataFrame(bets).sort_values("edge", ascending=False)
        bpath = os.path.join(OUT, f"value_round_{rnd}.csv")
        bets.to_csv(bpath, index=False)
        print(f"\nValue check -> {bpath}")
        print(bets.to_string(index=False))


if __name__ == "__main__":
    main()
