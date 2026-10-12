"""
AFLW disposal model.

Expected disposals for a player in a game:

    mu = disposal_rate (per full game on the ground)
         x expected time on ground
         x opponent factor (disposals the opponent concedes vs league)
         x team factor (her team's disposal volume vs league)
         x home / weather adjustments

  * rate : recency-weighted disposals per 100% time on ground, shrunk toward the average
           for the position she's named in. Using a rate means a game cut short by injury
           or a sub doesn't drag her numbers down.
  * expected TOG : her recent time on ground, shrunk toward the average for her named position
           (bench players get less).

Disposals ~ Negative Binomial(mean=mu, size=r). Walk-forward: each game is predicted only from
games before it.
"""
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from model import POS_GROUP, WET, load

LINES = [10, 15, 20, 25, 30, 35]   # milestone markets shown on the site (X+ disposals)


@dataclass
class DParams:
    player_decay: float = 0.92   # per game played
    season_carry: float = 0.6    # extra multiplier on history at a new season
    k_rate: float = 3.0          # pseudo full-games pulling rate toward positional average
    k_tog: float = 3.0           # pseudo games pulling TOG toward positional average
    team_decay: float = 0.92
    k_team: float = 4.0
    g_opp: float = 1.0           # exponent on opponent factor (0 = ignore)
    g_team: float = 0.5          # exponent on own-team factor
    lg_decay: float = 0.998
    sos: float = 0.0             # schedule adjustment of team/opponent disposal ratings
    nb_size: float = 15.0        # fallback dispersion if no variance curve is fitted
    recal_g: float = 0.0         # stretch: mu' = mu * (mu/12)^g, lifts high-volume players the shrinkage pulls down too far
    var_c: tuple = ()            # predictive variance as a quadratic in mu' (c2, c1, c0); spread differs for low vs high volume


class DEngine:
    def __init__(self, p: DParams):
        self.p = p
        self.season = None
        self.pl_d = defaultdict(float)     # decayed disposals
        self.pl_t = defaultdict(float)     # decayed full-games on ground (tog/100)
        self.pl_tog = defaultdict(float)   # decayed tog %
        self.pl_n = defaultdict(float)     # decayed games
        self.pl_games = defaultdict(int)
        self.t_for = defaultdict(float)
        self.t_against = defaultdict(float)
        self.t_n = defaultdict(float)
        self.lg_d, self.lg_n = 0.0, 0.0
        self.pos_d = defaultdict(float); self.pos_t = defaultdict(float)
        self.pos_tog = defaultdict(float); self.pos_n = defaultdict(int)
        self.home_d = self.away_d = 0.0
        self.wet_d = self.dry_d = 0.0; self.wet_n = self.dry_n = 0

    def new_season(self, season):
        if self.season is not None and season != self.season:
            c = self.p.season_carry
            for d in (self.pl_d, self.pl_t, self.pl_tog, self.pl_n, self.t_for, self.t_against, self.t_n):
                for k in d:
                    d[k] *= c
        self.season = season

    def lg_avg(self):
        return self.lg_d / self.lg_n if self.lg_n else 230.0

    def pos_rate(self, g):
        return self.pos_d[g] / self.pos_t[g] if self.pos_t[g] > 20 else 14.0

    def pos_tog_avg(self, g):
        return self.pos_tog[g] / self.pos_n[g] if self.pos_n[g] > 20 else 75.0

    def home_mult(self):
        return (self.home_d / self.away_d) ** 0.5 if self.away_d > 5000 else 1.0

    def wet_mult(self, w):
        if self.wet_n < 40 or self.dry_n < 40:
            return 1.0
        wet, dry = self.wet_d / self.wet_n, self.dry_d / self.dry_n
        avg = (self.wet_d + self.dry_d) / (self.wet_n + self.dry_n)
        return (wet if w in WET else dry) / avg

    def team_factor(self, team, opp, is_home, weather):
        p, lg = self.p, self.lg_avg()
        att = (self.t_for[team] + p.k_team * lg) / (self.t_n[team] + p.k_team)
        dfn = (self.t_against[opp] + p.k_team * lg) / (self.t_n[opp] + p.k_team)
        hm = self.home_mult()
        hm = hm if is_home else 1 / hm
        return (att / lg) ** p.g_team * (dfn / lg) ** p.g_opp * hm * self.wet_mult(weather)

    def predict_team(self, team, opp, is_home, weather, lineup):
        p = self.p
        f = self.team_factor(team, opp, is_home, weather)
        rows = []
        for pid, pos in lineup:
            g = POS_GROUP.get(pos, "BENCH")
            rate = (self.pl_d[pid] + p.k_rate * self.pos_rate(g)) / (self.pl_t[pid] + p.k_rate)
            etog = (self.pl_tog[pid] + p.k_tog * self.pos_tog_avg(g)) / (self.pl_n[pid] + p.k_tog)
            rows.append((pid, rate, etog, rate * etog / 100 * f))
        df = pd.DataFrame(rows, columns=["playerId", "d_rate", "d_tog", "d_mu_raw"])
        df["d_mu"] = adjust(df["d_mu_raw"], p)
        df["d_size"] = size_for(df["d_mu"], p)
        return df

    def ratings(self, team):
        p, lg = self.p, self.lg_avg()
        return ((self.t_for[team] + p.k_team * lg) / (self.t_n[team] + p.k_team) / lg,
                (self.t_against[team] + p.k_team * lg) / (self.t_n[team] + p.k_team) / lg)

    def update_team(self, team, opp, is_home, weather, players, opp_disp, opp_att=1.0, opp_def=1.0):
        p = self.p
        td = players["disposals"].sum()
        d = p.player_decay
        for r in players.itertuples(index=False):
            pid, tog = r.playerId, float(r.tog or 0)
            self.pl_d[pid] = d * self.pl_d[pid] + r.disposals
            self.pl_t[pid] = d * self.pl_t[pid] + tog / 100
            self.pl_tog[pid] = d * self.pl_tog[pid] + tog
            self.pl_n[pid] = d * self.pl_n[pid] + 1
            self.pl_games[pid] += 1
            g = POS_GROUP.get(r.position, "BENCH")
            self.pos_d[g] += r.disposals; self.pos_t[g] += tog / 100
            self.pos_tog[g] += tog; self.pos_n[g] += 1
        dt = p.team_decay
        self.t_for[team] = dt * self.t_for[team] + td / (opp_def ** p.sos)
        self.t_against[team] = dt * self.t_against[team] + opp_disp / (opp_att ** p.sos)
        self.t_n[team] = dt * self.t_n[team] + 1
        self.lg_d = p.lg_decay * self.lg_d + td
        self.lg_n = p.lg_decay * self.lg_n + 1
        if is_home:
            self.home_d += td
        else:
            self.away_d += td
        if weather in WET:
            self.wet_d += td; self.wet_n += 1
        elif weather:
            self.dry_d += td; self.dry_n += 1


def adjust(mu, p: DParams):
    mu = np.maximum(0.5, np.asarray(mu, float))
    return mu * (mu / 12.0) ** p.recal_g


def size_for(mu, p: DParams):
    mu = np.asarray(mu, float)
    if not p.var_c:
        return np.full_like(mu, p.nb_size)
    var = np.maximum(np.polyval(p.var_c, mu), 1.08 * mu)
    return mu ** 2 / (var - mu)


def p_at_least(mu, k, r):
    mu = np.asarray(mu, float)
    r = np.asarray(r, float)
    return stats.nbinom.sf(k - 1, r, r / (r + mu))


def nll(mu, y, r):
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    r = np.asarray(r, float)
    return float(-np.mean(stats.nbinom.logpmf(y, r, r / (r + mu))))


def walk_forward(pm, p: DParams):
    eng = DEngine(p)
    out = []
    for mid, g in pm.groupby("mid", sort=False):
        eng.new_season(g["season"].iloc[0])
        teams = g["teamId"].unique()
        if len(teams) != 2:
            continue
        for t in teams:
            tg = g[g["teamId"] == t]
            r0 = tg.iloc[0]
            df = eng.predict_team(t, r0.oppId, r0.is_home, r0.weather, list(zip(tg["playerId"], tg["position"])))
            df = df.merge(tg[["playerId", "name", "position", "pos_group", "disposals", "tog", "team", "opp",
                              "season", "round", "mid"]], on="playerId", how="left")
            out.append(df)
        disp = {t: g.loc[g["teamId"] == t, "disposals"].sum() for t in teams}
        snap = {t: eng.ratings(t) for t in teams}
        for t in teams:
            tg = g[g["teamId"] == t]
            o = [x for x in teams if x != t][0]
            r0 = tg.iloc[0]
            eng.update_team(t, o, r0.is_home, r0.weather, tg, disp[o], opp_att=snap[o][0], opp_def=snap[o][1])
    return pd.concat(out, ignore_index=True), eng
