"""
Pull AFLW match + player data from the AFL's public API (same source fitzRoy uses).

Run on your own PC (the API is reachable from normal home internet):
    python fetch_data.py                # refresh everything (2022 onward)
    python fetch_data.py --lineups-only # just refresh named teams for upcoming games

Writes into ./data/:
    player_matches.csv   one row per player per completed game
    matches.csv          every fixture (completed + upcoming) with score + weather
    lineups.csv          named teams for upcoming games (once clubs announce them, usually Thursday)
"""
import argparse
import csv
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

AFLW_COMP_ID = 3
# AFLW comp-season ids (aflapi). 2022 had two seasons (S6 Jan-Apr, S7 Aug-Nov).
SEASONS = [(41, "2022a"), (51, "2022b"), (61, "2023"), (72, "2024"), (84, "2025"), (96, "2026")]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url, headers=None, method="GET", retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=b"" if method == "POST" else None,
                                         headers={**UA, **(headers or {})}, method=method)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def token():
    return _get("https://api.afl.com.au/cfs/afl/WMCTok", method="POST")["token"]


def season_matches(comp_season_id):
    url = (f"https://aflapi.afl.com.au/afl/v2/matches?competitionId={AFLW_COMP_ID}"
           f"&compSeasonId={comp_season_id}&pageSize=300")
    return _get(url)["matches"]


def player_stats(match_pid, tok):
    return _get(f"https://api.afl.com.au/cfs/afl/playerStats/match/{match_pid}",
                {"x-media-mis-token": tok})


def roster(match_pid, tok):
    return _get(f"https://api.afl.com.au/cfs/afl/matchRoster/full/{match_pid}",
                {"x-media-mis-token": tok})


def _name(p):
    n = p.get("playerName", {})
    return f"{n.get('givenName', '')} {n.get('surname', '')}".strip()


def discover_seasons(first_year=2022):
    """All AFLW seasons from first_year on, labelled like SEASONS (2022 has two)."""
    try:
        cs = _get(f"https://aflapi.afl.com.au/afl/v2/competitions/{AFLW_COMP_ID}/compseasons?pageSize=50")["compSeasons"]
    except Exception:
        return SEASONS
    known = dict(SEASONS)
    out = []
    for c in sorted(cs, key=lambda c: c["id"]):
        if c["id"] in known:
            out.append((c["id"], known[c["id"]]))
        else:
            yr = int(c["name"][:4])
            if yr >= first_year:
                out.append((c["id"], str(yr)))
    return out or SEASONS


def fetch_all(seasons=None, workers=8):
    os.makedirs(DATA, exist_ok=True)
    seasons = seasons or discover_seasons()
    tok = token()
    matches, completed = [], []
    for sid, label in seasons:
        for x in season_matches(sid):
            m = {
                "season": label, "mid": x["providerId"],
                "round": (x.get("round") or {}).get("roundNumber"),
                "rname": (x.get("round") or {}).get("name"),
                "date": x.get("utcStartTime"), "status": x.get("status"),
                "home": x["home"]["team"]["name"], "homeId": x["home"]["team"]["providerId"],
                "away": x["away"]["team"]["name"], "awayId": x["away"]["team"]["providerId"],
                "venue": (x.get("venue") or {}).get("name"),
                "hs": ((x["home"].get("score") or {}).get("totalScore")),
                "as": ((x["away"].get("score") or {}).get("totalScore")),
            }
            matches.append(m)
            if m["status"] == "CONCLUDED":
                completed.append(m)
        print(f"{label}: {sum(1 for m in matches if m['season'] == label)} fixtures")

    # incremental: keep what we already have, only fetch new games
    old_m, old_rows = {}, []
    mp, pp = os.path.join(DATA, "matches.csv"), os.path.join(DATA, "player_matches.csv")
    if os.path.exists(mp):
        old_m = {r["mid"]: r for r in csv.DictReader(open(mp, encoding="utf-8"))}
    if os.path.exists(pp):
        old_rows = list(csv.DictReader(open(pp, encoding="utf-8")))
    have_stats = {r["mid"] for r in old_rows}

    def do_weather(m):
        o = old_m.get(m["mid"])
        if o and o.get("status") == "CONCLUDED" and m["status"] == "CONCLUDED":
            m["weather"], m["temp"], m["wdesc"] = o.get("weather", ""), o.get("temp", ""), o.get("wdesc", "")
            return
        try:
            w = (roster(m["mid"], tok).get("matchRoster") or {}).get("weather") or {}
            m["weather"], m["temp"], m["wdesc"] = w.get("weatherType", ""), w.get("tempInCelsius", ""), w.get("description", "")
        except Exception:
            m["weather"] = m["temp"] = m["wdesc"] = ""

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(do_weather, matches))

    rows = [r for r in old_rows if r["mid"] in {m["mid"] for m in completed}]

    def do_stats(m):
        out = []
        ps = player_stats(m["mid"], tok)
        for side, arr in (("H", ps.get("homeTeamPlayerStats")), ("A", ps.get("awayTeamPlayerStats"))):
            for p in arr or []:
                s = p["playerStats"]["stats"]
                pl = p["player"]["player"]
                out.append({
                    "season": m["season"], "mid": m["mid"], "side": side, "teamId": p["teamId"],
                    "playerId": pl["player"]["playerId"], "name": _name(pl["player"]),
                    "position": pl.get("position"), "goals": s.get("goals") or 0,
                    "behinds": s.get("behinds") or 0, "shots": s.get("shotsAtGoal") or 0,
                    "mi50": s.get("marksInside50") or 0, "i50": s.get("inside50s") or 0,
                    "disposals": s.get("disposals") or 0, "si": s.get("scoreInvolvements") or 0,
                    "goal_assists": s.get("goalAssists") or 0,
                    "tog": p["playerStats"].get("timeOnGroundPercentage"),
                })
        return out

    todo = [m for m in completed if m["mid"] not in have_stats]
    with ThreadPoolExecutor(workers) as ex:
        for out in ex.map(do_stats, todo):
            rows.extend(out)

    _write(mp, matches)
    _write(pp, rows)
    print(f"Fetched {len(todo)} new games. Saved {len(matches)} fixtures and {len(rows)} player-games to {DATA}")
    fetch_lineups(tok, matches)


def fetch_lineups(tok=None, matches=None):
    """Named teams for the next round of scheduled games."""
    tok = tok or token()
    if matches is None:
        matches = list(csv.DictReader(open(os.path.join(DATA, "matches.csv"), encoding="utf-8")))
    cur = max(str(m["season"]) for m in matches)
    upcoming = [m for m in matches if m["status"] != "CONCLUDED" and str(m["season"]) == cur]
    if not upcoming:
        print("No upcoming games.")
        return
    next_round = min(int(m["round"]) for m in upcoming if m["round"] not in (None, ""))
    rows = []
    for m in [m for m in upcoming if int(m["round"]) == next_round]:
        mr = roster(m["mid"], tok).get("matchRoster") or {}
        for side_key in ("homeTeam", "awayTeam"):
            t = mr.get(side_key) or {}
            for p in t.get("positions") or []:
                rows.append({
                    "mid": m["mid"], "teamId": t.get("teamId"), "team": (t.get("teamName") or {}).get("teamName"),
                    "playerId": p["player"]["playerId"], "name": _name(p["player"]),
                    "position": p.get("position"), "teamStatus": t.get("teamStatus"),
                })
    path = os.path.join(DATA, "lineups.csv")
    if rows:
        _write(path, rows)
        print(f"Round {next_round}: saved {len(rows)} named players to {path}")
    else:
        if os.path.exists(path):
            os.remove(path)
        print(f"Round {next_round}: teams not named yet - fair_odds.py will use each side's last lineup.")


def _write(path, rows):
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineups-only", action="store_true")
    a = ap.parse_args()
    fetch_lineups() if a.lineups_only else fetch_all()
