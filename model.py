"""
AFLW goalkicker model.

Expected goals for a player in a game:

    lambda = team_expected_shots  x  player_share_of_team_shots  x  player_goal_accuracy

  * team_expected_shots : league average, scaled by the team's attack (shots for) and the
                          opponent's defence (shots conceded), plus home ground and weather.
  * share               : the player's recent share of her team's shots, shrunk toward the
                          average for the position she is named in this week. Shares across
                          the named 21 are re-normalised to sum to 1, so if a key forward is
                          out, her shots get redistributed to whoever is named.
  * accuracy            : goals / shots, shrunk hard toward the league rate (accuracy is noisy).

Goals ~ Negative Binomial(mean=lambda, size=r)  (r -> infinity is Poisson).
Everything is computed walk-forward: each game is predicted only from games before it.
"""
import math
import os
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

POS_GROUP = {
    "FF": "KEY_FWD", "CHF": "KEY_FWD",
    "FPL": "FWD_POCKET", "FPR": "FWD_POCKET",
    "HFFL": "HALF_FWD", "HFFR": "HALF_FWD",
    "C": "MID", "WL": "MID", "WR": "MID", "R": "MID", "RR": "MID",
    "RK": "RUCK",
    "FB": "BACK", "BPL": "BACK", "BPR": "BACK", "CHB": "BACK", "HBFL": "BACK", "HBFR": "BACK",
    "INT": "BENCH", "SUB": "BENCH", "EMERG": "BENCH",
}
WET = {"RAIN", "THUNDERSTORMS"}
ZONE = {"KEY_FWD": "FWD", "FWD_POCKET": "FWD", "HALF_FWD": "FWD", "MID": "MID", "RUCK": "MID",
        "BACK": "BACK", "BENCH": "BENCH"}


@dataclass
class Params:
    player_decay: float = 0.90   # weight on a player's previous games, per game
    team_decay: float = 0.90     # same for team attack/defence
    season_carry: float = 0.60   # extra multiplier on all history at a new season
    k_share: float = 4.0         # pseudo-games pulling share toward positional average
    k_team: float = 4.0          # pseudo-games pulling team rates toward league average
    k_acc: float = 25.0          # pseudo-shots pulling accuracy toward league rate
    nb_size: float = 8.0         # negative binomial size (dispersion); large = Poisson
    lg_decay: float = 0.995      # recency weighting of the league-average shots per team-game
    share_power: float = 1.0     # >1 sharpens differences between players' shares before re-normalising
    zone_weight: float = 1.0     # weight on games played in a different zone (fwd/mid/back) to this week's named one; 1 = ignore role


# ----------------------------------------------------------------------------- data
def load(data_dir=DATA):
    pm = pd.read_csv(os.path.join(data_dir, "player_matches.csv"))
    mt = pd.read_csv(os.path.join(data_dir, "matches.csv"))
    mt["date"] = pd.to_datetime(mt["date"], utc=True, format="mixed")
    mt["season_year"] = mt["season"].astype(str).str[:4].astype(int)
    for c in ("weather", "wdesc"):
        if c not in mt:
            mt[c] = ""
    mt["weather"] = mt["weather"].fillna("")
    pm["pos_group"] = pm["position"].map(POS_GROUP).fillna("BENCH")
    pm = pm.merge(mt[["mid", "date", "season", "round", "homeId", "awayId", "home", "away", "weather"]],
                  on=["mid", "season"], how="left")
    pm["oppId"] = np.where(pm["teamId"] == pm["homeId"], pm["awayId"], pm["homeId"])
    pm["is_home"] = (pm["teamId"] == pm["homeId"]).astype(int)
    pm["team"] = np.where(pm["is_home"] == 1, pm["home"], pm["away"])
    pm["opp"] = np.where(pm["is_home"] == 1, pm["away"], pm["home"])
    team_shots = pm.groupby(["mid", "teamId"])["shots"].transform("sum")
    pm["team_shots"] = team_shots
    pm["share"] = np.where(team_shots > 0, pm["shots"] / team_shots.replace(0, np.nan), 0.0)
    pm = pm.sort_values(["date", "mid", "teamId"]).reset_index(drop=True)
    return pm, mt


# ----------------------------------------------------------------------------- engine
class Engine:
    """Holds exponentially weighted state; predict() then update() one game at a time."""

    def __init__(self, p: Params):
        self.p = p
        self.season = None
        # player state
        self.pl_share = defaultdict(float)
        self.pl_n = defaultdict(float)
        self.pl_goals = defaultdict(float)
        self.pl_shots = defaultdict(float)
        # per-zone decayed share sums, so games in a different role can count for less
        self.pl_zshare = defaultdict(lambda: defaultdict(float))
        self.pl_zn = defaultdict(lambda: defaultdict(float))
        self.pl_games = defaultdict(int)
        self.pl_name, self.pl_team, self.pl_pos = {}, {}, {}
        # team state
        self.t_for = defaultdict(float)
        self.t_against = defaultdict(float)
        self.t_n = defaultdict(float)
        # league totals: shots per team-game is recency weighted (scoring levels drift by season)
        self.lg_shots, self.lg_team_games = 0.0, 0.0
        self.lg_goals, self.lg_shot_total = 0.0, 0.0
        self.pos_share_sum = defaultdict(float)
        self.pos_n = defaultdict(int)
        self.home_shots, self.away_shots = 0.0, 0.0
        self.wet_shots, self.wet_n, self.dry_shots, self.dry_n = 0.0, 0, 0.0, 0

    # --- league-level quantities
    def lg_avg_shots(self):
        return self.lg_shots / self.lg_team_games if self.lg_team_games else 20.0

    def lg_acc(self):
        return self.lg_goals / self.lg_shot_total if self.lg_shot_total else 0.42

    def pos_prior(self, g):
        if self.pos_n[g] < 30:
            return {"KEY_FWD": .16, "FWD_POCKET": .12, "HALF_FWD": .08, "MID": .04,
                    "RUCK": .03, "BACK": .005, "BENCH": .03}[g]
        return self.pos_share_sum[g] / self.pos_n[g]

    def home_mult(self):
        tot = self.home_shots + self.away_shots
        return (self.home_shots / self.away_shots) ** 0.5 if tot > 500 and self.away_shots else 1.0

    def wet_mult(self, weather):
        if self.wet_n < 40 or self.dry_n < 40:
            return 1.0
        wet, dry = self.wet_shots / self.wet_n, self.dry_shots / self.dry_n
        avg = (self.wet_shots + self.dry_shots) / (self.wet_n + self.dry_n)
        return (wet if weather in WET else dry) / avg

    # --- season rollover
    def new_season(self, season):
        if self.season is not None and season != self.season:
            c = self.p.season_carry
            for z in (self.pl_zshare, self.pl_zn):
                for k in z:
                    for zz in z[k]:
                        z[k][zz] *= c
            for d in (self.pl_share, self.pl_n, self.pl_goals, self.pl_shots,
                      self.t_for, self.t_against, self.t_n):
                for k in d:
                    d[k] *= c
        self.season = season

    # --- prediction for one team in one game
    def team_expected_shots(self, team, opp, is_home, weather):
        p, lg = self.p, self.lg_avg_shots()
        att = (self.t_for[team] + p.k_team * lg) / (self.t_n[team] + p.k_team)
        dfn = (self.t_against[opp] + p.k_team * lg) / (self.t_n[opp] + p.k_team)
        hm = self.home_mult()
        hm = hm if is_home else 1 / hm
        return lg * (att / lg) * (dfn / lg) * hm * self.wet_mult(weather)

    def predict_team(self, team, opp, is_home, weather, lineup):
        """lineup: list of (playerId, position_code). Returns DataFrame of player lambdas."""
        p = self.p
        exp_shots = self.team_expected_shots(team, opp, is_home, weather)
        lg_acc = self.lg_acc()
        rows = []
        for pid, pos in lineup:
            g = POS_GROUP.get(pos, "BENCH")
            prior = self.pos_prior(g)
            if p.zone_weight < 1.0:
                zn = ZONE[g]
                s_sum = n_sum = 0.0
                for z, v in self.pl_zshare[pid].items():
                    w = 1.0 if (z == zn or z == "BENCH" or zn == "BENCH") else p.zone_weight
                    s_sum += w * v
                    n_sum += w * self.pl_zn[pid][z]
                raw = (s_sum + p.k_share * prior) / (n_sum + p.k_share)
            else:
                raw = (self.pl_share[pid] + p.k_share * prior) / (self.pl_n[pid] + p.k_share)
            acc = (self.pl_goals[pid] + p.k_acc * lg_acc) / (self.pl_shots[pid] + p.k_acc)
            rows.append((pid, pos, g, raw, acc, self.pl_games[pid]))
        df = pd.DataFrame(rows, columns=["playerId", "position", "pos_group", "raw_share", "acc", "games_hist"])
        sharp = df["raw_share"] ** p.share_power
        tot = sharp.sum()
        df["share"] = sharp / tot if tot > 0 else 0
        df["exp_shots"] = exp_shots * df["share"]
        df["lam"] = df["exp_shots"] * df["acc"]
        df["team_exp_shots"] = exp_shots
        return df

    # --- update with a completed team-game
    def update_team(self, team, opp, is_home, weather, players: pd.DataFrame, opp_shots):
        p = self.p
        ts = players["shots"].sum()
        for r in players.itertuples(index=False):
            pid = r.playerId
            d = p.player_decay
            self.pl_share[pid] = d * self.pl_share[pid] + r.share
            self.pl_n[pid] = d * self.pl_n[pid] + 1
            self.pl_goals[pid] = d * self.pl_goals[pid] + r.goals
            self.pl_shots[pid] = d * self.pl_shots[pid] + r.shots
            zn = ZONE[POS_GROUP.get(r.position, "BENCH")]
            for z in list(self.pl_zshare[pid].keys()) + [zn]:
                self.pl_zshare[pid][z] *= d
                self.pl_zn[pid][z] *= d
            self.pl_zshare[pid][zn] += r.share
            self.pl_zn[pid][zn] += 1
            self.pl_games[pid] += 1
            self.pl_name[pid], self.pl_team[pid], self.pl_pos[pid] = r.name, team, r.position
            g = POS_GROUP.get(r.position, "BENCH")
            self.pos_share_sum[g] += r.share
            self.pos_n[g] += 1
            self.lg_goals += r.goals
            self.lg_shot_total += r.shots
        d = p.team_decay
        self.t_for[team] = d * self.t_for[team] + ts
        self.t_against[team] = d * self.t_against[team] + opp_shots
        self.t_n[team] = d * self.t_n[team] + 1
        self.lg_shots = p.lg_decay * self.lg_shots + ts
        self.lg_team_games = p.lg_decay * self.lg_team_games + 1
        if is_home:
            self.home_shots += ts
        else:
            self.away_shots += ts
        if weather in WET:
            self.wet_shots += ts; self.wet_n += 1
        elif weather:
            self.dry_shots += ts; self.dry_n += 1


# ----------------------------------------------------------------------------- probabilities
def p_at_least(lam, k, nb_size):
    lam = np.asarray(lam, dtype=float)
    if nb_size is None or nb_size >= 1e6:
        return stats.poisson.sf(k - 1, lam)
    pr = nb_size / (nb_size + lam)
    return stats.nbinom.sf(k - 1, nb_size, pr)


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(pm: pd.DataFrame, p: Params):
    """Predict every player-game using only prior games. Returns predictions DataFrame."""
    eng = Engine(p)
    out = []
    for mid, g in pm.groupby("mid", sort=False):
        eng.new_season(g["season"].iloc[0])
        teams = g["teamId"].unique()
        if len(teams) != 2:
            continue
        preds = []
        for t in teams:
            tg = g[g["teamId"] == t]
            r0 = tg.iloc[0]
            df = eng.predict_team(t, r0.oppId, r0.is_home, r0.weather,
                                  list(zip(tg["playerId"], tg["position"])))
            df = df.merge(tg[["playerId", "name", "goals", "shots", "team", "opp", "season", "round", "date", "mid"]],
                          on="playerId", how="left")
            preds.append(df)
        out.extend(preds)
        shots = {t: g.loc[g["teamId"] == t, "shots"].sum() for t in teams}
        for t in teams:
            tg = g[g["teamId"] == t]
            o = [x for x in teams if x != t][0]
            r0 = tg.iloc[0]
            eng.update_team(t, o, r0.is_home, r0.weather, tg, shots[o])
    res = pd.concat(out, ignore_index=True)
    res["p1"] = p_at_least(res["lam"], 1, p.nb_size)
    res["p2"] = p_at_least(res["lam"], 2, p.nb_size)
    res["p3"] = p_at_least(res["lam"], 3, p.nb_size)
    return res, eng


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def goal_nll(lam, goals, nb_size):
    lam = np.clip(np.asarray(lam, float), 1e-6, None)
    if nb_size >= 1e6:
        return float(-np.mean(stats.poisson.logpmf(goals, lam)))
    return float(-np.mean(stats.nbinom.logpmf(goals, nb_size, nb_size / (nb_size + lam))))
