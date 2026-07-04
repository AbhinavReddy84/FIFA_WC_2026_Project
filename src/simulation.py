"""Match prediction and full tournament simulation engine."""
from __future__ import annotations

import itertools
import pickle
from functools import lru_cache
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    ALL_MODELS_PKL, BEST_MODEL_PKL, CONFEDERATION_STRENGTH,
    CONF_CALIBRATION, DRAW_TEMPERATURE, FEATURE_COLS_PKL,
    FIFA_RANK_CSV, FORM_CALIBRATION, GROUP_STAGES_CSV,
    H2H_CALIBRATION, KNOCKOUT_FIXTURES_CSV, METADATA_JSON,
    PERM_IMP_CSV, RANK_CALIBRATION, STATIC_CONFEDERATION_MAP,
    WC_FIXTURES_CSV, WC_GROUPS_CSV, CLASS_NAMES, CLASS_TO_IDX,
)
from src.feature_engineering import (
    build_elo_lookup,
    build_team_stats,
    compute_h2h,
    get_team_stat,
    load_raw_history,
)

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_model_bundle():
    with open(BEST_MODEL_PKL,   "rb") as f: model = pickle.load(f)
    with open(FEATURE_COLS_PKL, "rb") as f: feature_cols = pickle.load(f)

    import json
    metadata = {}
    if METADATA_JSON.exists():
        with open(METADATA_JSON) as f: metadata = json.load(f)

    perm_df = pd.read_csv(PERM_IMP_CSV) if PERM_IMP_CSV.exists() else pd.DataFrame()

    all_models = {}
    if ALL_MODELS_PKL.exists():
        with open(ALL_MODELS_PKL, "rb") as f: all_models = pickle.load(f)

    return model, feature_cols, metadata, perm_df, all_models


@lru_cache(maxsize=1)
def load_tournament_data() -> dict:
    """Load WC 2026 tournament fixtures, groups, FIFA ranks."""
    # Groups
    df_groups = pd.read_csv(GROUP_STAGES_CSV, sep=";")
    df_groups.columns = [c.strip() for c in df_groups.columns]
    # Handle encoding issues in team names
    df_groups["nation"] = df_groups["nation"].str.strip()

    # Fixtures (group stage)
    df_fixtures = pd.read_csv(WC_FIXTURES_CSV, parse_dates=["date"])
    df_fixtures["neutral"] = df_fixtures["neutral"].map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).fillna(1).astype(int)

    # Knockout fixtures
    df_knockout = pd.read_csv(KNOCKOUT_FIXTURES_CSV)

    # FIFA ranks
    print(type(FIFA_RANK_CSV))
    print(FIFA_RANK_CSV)
    print(pd.read_csv)
    df_rank = pd.read_csv(FIFA_RANK_CSV, encoding="latin1")
    # Clean up column names
    df_rank.columns = [c.strip() for c in df_rank.columns]
    # Map team -> rank
    nation_col = next((c for c in df_rank.columns if "nation" in c.lower() or "team" in c.lower()), df_rank.columns[0])
    rank_col   = next((c for c in df_rank.columns if "2026" in c.lower() or "rank" in c.lower()), df_rank.columns[1])
    team_to_rank = dict(zip(df_rank[nation_col].str.strip(), df_rank[rank_col]))

    # WC groups detail (Elo ratings)
    team_to_elo_wc: dict[str, float] = {}
    team_to_confederation: dict[str, str] = dict(STATIC_CONFEDERATION_MAP)
    try:
        df_wc = pd.read_csv(WC_GROUPS_CSV, encoding="latin1")
        if "elo_rating_2026" in df_wc.columns and "team" in df_wc.columns:
            for _, row in df_wc.iterrows():
                t = str(row["team"]).strip()
                team_to_elo_wc[t] = float(row["elo_rating_2026"]) if pd.notna(row["elo_rating_2026"]) else 1500.0
        if "confederation" in df_wc.columns and "team" in df_wc.columns:
            for _, row in df_wc.iterrows():
                team_to_confederation[str(row["team"]).strip()] = str(row["confederation"]).strip()
    except Exception:
        pass

    wc_teams   = sorted(df_groups["nation"].unique().tolist())
    team_to_group = dict(zip(df_groups["nation"], df_groups["group"]))

    # Merge Elo: prefer wc-specific, then historical
    elo_hist = build_elo_lookup()
    final_elo = {t: team_to_elo_wc.get(t, elo_hist.get(t, 1500.0)) for t in wc_teams}
    for t, e in elo_hist.items():
        final_elo.setdefault(t, e)

    # FIFA rank defaults
    for t in wc_teams:
        team_to_rank.setdefault(str(t), 100)
    for t in final_elo:
        team_to_rank.setdefault(str(t), 150)

    # All teams (for match predictor)
    hist = pd.DataFrame({
    "home_team": [],
    "away_team": []
})
    all_teams = sorted(set(hist["home_team"]) | set(hist["away_team"]) | set(wc_teams))

    return {
        "wc_teams":            wc_teams,
        "all_teams":           all_teams,
        "team_to_group":       team_to_group,
        "team_to_confederation": team_to_confederation,
        "team_to_elo":         final_elo,
        "team_to_rank":        team_to_rank,
        "df_groups":           df_groups,
        "df_fixtures":         df_fixtures,
        "df_knockout":         df_knockout,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction engine
# ─────────────────────────────────────────────────────────────────────────────

class MatchPredictor:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.model, self.feature_cols, self.metadata, self.perm_df, self.all_models = load_model_bundle()
        self.tdata      = load_tournament_data()
        self.team_stats = build_team_stats()
        self.elo_lookup = build_elo_lookup()
        self.rng        = rng or np.random.default_rng(RANDOM_STATE)

    def _get_elo(self, team: str) -> float:
        return float(self.tdata["team_to_elo"].get(team)
                     or self.elo_lookup.get(team, 1500.0))

    def _get_conf(self, team: str) -> str:
        return (self.tdata["team_to_confederation"].get(team)
                or STATIC_CONFEDERATION_MAP.get(team, "Unknown"))

    def _get_rank(self, team: str) -> float:
        return float(self.tdata["team_to_rank"].get(team, 150))

    def _conf_strength(self, team: str) -> float:
        conf = self._get_conf(team)
        return CONFEDERATION_STRENGTH.get(conf, 0.88)

    def _rank_strength(self, team: str) -> float:
        rank = self._get_rank(team)
        if rank >= 150: return 0.0
        return (70.0 - min(rank, 140.0)) / 70.0

    def _form_strength(self, team: str) -> float:
        pts = get_team_stat(team, "pts_2y", self.team_stats)
        gd  = get_team_stat(team, "gd_2y",  self.team_stats)
        return (pts - 1.4) / 1.5 + gd / 3.0

    def build_features(self, home: str, away: str, neutral: bool = True,
                       tournament_weight: int = 5) -> pd.DataFrame:
        h_elo = self._get_elo(home);  a_elo = self._get_elo(away)
        h_conf = self._get_conf(home); a_conf = self._get_conf(away)
        h_cs   = self._conf_strength(home); a_cs = self._conf_strength(away)

        def gs(team, stat): return get_team_stat(team, stat, self.team_stats)

        row = {
            "elo_diff":            h_elo - a_elo,
            "elo_abs_diff":        abs(h_elo - a_elo),
            "elo_ratio":           h_elo / a_elo if a_elo else 1.0,
            "home_elo":            h_elo,
            "away_elo":            a_elo,
            "conf_strength_diff":  h_cs - a_cs,
            "home_conf_strength":  h_cs,
            "away_conf_strength":  a_cs,
            "same_confederation":  int(h_conf == a_conf),
            "neutral":             int(neutral),
            "tournament_weight":   float(tournament_weight),
            "is_competitive":      1,
            "match_year":          2026,
            "pts_2y_diff":         gs(home, "pts_2y")    - gs(away, "pts_2y"),
            "gd_2y_diff":          gs(home, "gd_2y")     - gs(away, "gd_2y"),
            "gf_2y_diff":          gs(home, "gf_2y")     - gs(away, "gf_2y"),
            "w_rate_2y_diff":      gs(home, "w_rate_2y") - gs(away, "w_rate_2y"),
            "home_pts_2y":         gs(home, "pts_2y"),
            "away_pts_2y":         gs(away, "pts_2y"),
            "pts_l10_diff":        gs(home, "pts_l10")   - gs(away, "pts_l10"),
            "gd_l10_diff":         gs(home, "gd_l10")    - gs(away, "gd_l10"),
            "w_rate_l10_diff":     gs(home, "w_rate_l10")- gs(away, "w_rate_l10"),
            "home_confederation":  h_conf,
            "away_confederation":  a_conf,
        }
        X = pd.DataFrame([row])
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        return X[self.feature_cols]

    def calibrate(self, home: str, away: str,
                  base_proba: np.ndarray) -> tuple[np.ndarray, dict]:
        """Apply multi-factor calibration on top of model probabilities."""
        p_away, p_draw, p_home = map(float, base_proba)

        conf_edge = np.log(self._conf_strength(home) / max(self._conf_strength(away), 1e-9))
        rank_edge = self._rank_strength(home) - self._rank_strength(away)
        form_edge = self._form_strength(home)  - self._form_strength(away)
        h2h       = compute_h2h(home, away, n=10)
        h2h_edge  = h2h["h2h_score"]

        total_edge = (
            CONF_CALIBRATION  * conf_edge +
            RANK_CALIBRATION  * rank_edge +
            FORM_CALIBRATION  * form_edge +
            H2H_CALIBRATION   * h2h_edge
        )

        calibrated = np.array([
            p_away * np.exp(-total_edge),
            p_draw * DRAW_TEMPERATURE,
            p_home * np.exp(total_edge),
        ], dtype=float)
        calibrated = np.clip(calibrated / calibrated.sum(), 0.01, 0.97)

        return calibrated, {
            "edge":      float(total_edge),
            "conf_edge": float(conf_edge),
            "rank_edge": float(rank_edge),
            "form_edge": float(form_edge),
            "h2h_edge":  float(h2h_edge),
        }

    def predict(self, home: str, away: str, neutral: bool = True,
                tournament_weight: int = 5) -> dict[str, Any]:
        X = self.build_features(home, away, neutral, tournament_weight)
        base_proba = np.asarray(self.model.predict_proba(X))[0]

        # Align class order → [away_win, draw, home_win]
        classes = list(self.model.classes_) if hasattr(self.model, "classes_") else CLASS_NAMES
        ordered = np.zeros(3)
        for i, c in enumerate(classes):
            if c in CLASS_TO_IDX:
                ordered[CLASS_TO_IDX[c]] = base_proba[i]
        if ordered.sum() < 0.5:
            ordered = base_proba  # fallback

        proba, calibration = self.calibrate(home, away, ordered)

        return {
            "home_team":       home,
            "away_team":       away,
            "home_win_prob":   float(proba[CLASS_TO_IDX["home_win"]]),
            "draw_prob":       float(proba[CLASS_TO_IDX["draw"]]),
            "away_win_prob":   float(proba[CLASS_TO_IDX["away_win"]]),
            "base_home_prob":  float(ordered[CLASS_TO_IDX["home_win"]]),
            "base_draw_prob":  float(ordered[CLASS_TO_IDX["draw"]]),
            "base_away_prob":  float(ordered[CLASS_TO_IDX["away_win"]]),
            "home_elo":        self._get_elo(home),
            "away_elo":        self._get_elo(away),
            "home_rank":       self._get_rank(home),
            "away_rank":       self._get_rank(away),
            "home_conf":       self._get_conf(home),
            "away_conf":       self._get_conf(away),
            **calibration,
        }

    def sample_scoreline(self, outcome: str) -> tuple[int, int]:
        if outcome == "home_win":
            choices = [(1,0),(2,0),(2,1),(3,0),(3,1),(3,2),(4,1),(4,2)]
            probs   = [0.26,0.20,0.27,0.06,0.10,0.06,0.03,0.02]
        elif outcome == "away_win":
            choices = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
            probs   = [0.26,0.20,0.27,0.06,0.10,0.06,0.03,0.02]
        else:
            choices = [(0,0),(1,1),(2,2),(3,3)]
            probs   = [0.28,0.56,0.14,0.02]
        idx = self.rng.choice(len(choices), p=probs)
        return choices[idx]

    def simulate_match(self, home: str, away: str, neutral: bool = True,
                       tournament_weight: int = 5) -> dict[str, Any]:
        pred = self.predict(home, away, neutral, tournament_weight)
        probs = [pred["away_win_prob"], pred["draw_prob"], pred["home_win_prob"]]
        probs = np.array(probs); probs /= probs.sum()
        outcome = self.rng.choice(CLASS_NAMES, p=probs)
        hg, ag  = self.sample_scoreline(outcome)

        if outcome == "home_win":   winner = home
        elif outcome == "away_win": winner = away
        else:                       winner = None

        return {**pred, "home_goals": hg, "away_goals": ag,
                "outcome": outcome, "winner": winner}

    def simulate_knockout_match(self, home: str, away: str,
                                neutral: bool = True,
                                tournament_weight: int = 5) -> dict[str, Any]:
        m = self.simulate_match(home, away, neutral, tournament_weight)
        if m["winner"] is None:  # Draw → resolve via penalties
            p_home = m["home_win_prob"] / max(m["home_win_prob"] + m["away_win_prob"], 1e-9)
            m["winner"] = home if self.rng.random() < p_home else away
            m["loser"]  = away if m["winner"] == home else home
            m["resolved_via"] = "penalties"
        else:
            m["loser"]  = away if m["winner"] == home else home
            m["resolved_via"] = "normal"
        return m

    # ── Team profile ──────────────────────────────────────────────────────────
    def team_profile(self, team: str) -> dict:
        ts = self.team_stats
        def gs(stat): return get_team_stat(team, stat, ts)
        return {
            "team":           team,
            "confederation":  self._get_conf(team),
            "elo":            round(self._get_elo(team), 0),
            "fifa_rank":      int(self._get_rank(team)),
            "conf_strength":  round(self._conf_strength(team), 3),
            "pts_2y":         round(gs("pts_2y"),     2),
            "gd_2y":          round(gs("gd_2y"),      2),
            "w_rate_2y":      round(gs("w_rate_2y"),  3),
            "pts_l10":        round(gs("pts_l10"),    2),
            "gd_l10":         round(gs("gd_l10"),     2),
            "matches_2y":     int(gs("matches_2y")),
        }

    def strengths_weaknesses(self, home: str, away: str, pred: dict) -> dict:
        hp = self.team_profile(home)
        ap = self.team_profile(away)
        h2h = compute_h2h(home, away)

        def bullets(team, prof, opp_prof):
            s, w = [], []
            if prof["elo"] > opp_prof["elo"] + 40:
                s.append(f"Superior Elo ({int(prof['elo'])} vs {int(opp_prof['elo'])})")
            elif prof["elo"] < opp_prof["elo"] - 40:
                w.append(f"Lower Elo ({int(prof['elo'])} vs {int(opp_prof['elo'])})")

            if prof["fifa_rank"] < opp_prof["fifa_rank"]:
                s.append(f"Better FIFA rank (#{prof['fifa_rank']} vs #{opp_prof['fifa_rank']})")
            elif prof["fifa_rank"] > opp_prof["fifa_rank"] + 5:
                w.append(f"Lower FIFA rank (#{prof['fifa_rank']} vs #{opp_prof['fifa_rank']})")

            if prof["pts_2y"] > opp_prof["pts_2y"] + 0.25:
                s.append(f"Stronger 2-yr form ({prof['pts_2y']:.2f} pts/match)")
            elif prof["pts_2y"] < opp_prof["pts_2y"] - 0.25:
                w.append(f"Weaker 2-yr form ({prof['pts_2y']:.2f} pts/match)")

            if prof["gd_2y"] > opp_prof["gd_2y"] + 0.3:
                s.append(f"Better 2-yr goal diff ({prof['gd_2y']:+.2f}/match)")
            elif prof["gd_2y"] < opp_prof["gd_2y"] - 0.3:
                w.append(f"Worse goal diff trend ({prof['gd_2y']:+.2f}/match)")

            if prof["conf_strength"] > opp_prof["conf_strength"] + 0.05:
                s.append(f"Stronger confederation ({prof['confederation']})")
            elif prof["conf_strength"] < opp_prof["conf_strength"] - 0.05:
                w.append(f"Weaker confederation ({prof['confederation']})")

            if h2h["h2h_score"] > 0.3 and team == home:
                s.append(f"Favourable H2H ({h2h['wins']}W-{h2h['draws']}D-{h2h['losses']}L)")
            elif h2h["h2h_score"] < -0.3 and team == home:
                w.append(f"Unfavourable H2H ({h2h['wins']}W-{h2h['draws']}D-{h2h['losses']}L)")

            if not s: s.append("Well-balanced — no clear standout edge")
            if not w: w.append("No major weaknesses identified")
            return s, w

        hs, hw = bullets(home, hp, ap)
        as_, aw = bullets(away, ap, hp)
        return {
            home: {"strengths": hs, "weaknesses": hw, "profile": hp},
            away: {"strengths": as_, "weaknesses": aw, "profile": ap},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tournament simulation
# ─────────────────────────────────────────────────────────────────────────────

class TournamentSimulator:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.predictor = MatchPredictor(rng=rng)
        self.tdata     = self.predictor.tdata
        self.rng       = self.predictor.rng

        # Build group fixture map
        self.df_fixtures = self.tdata["df_fixtures"].copy()
        self.df_fixtures["group"] = self.df_fixtures["home_team"].map(
            self.tdata["team_to_group"])

    # ── Group stage ───────────────────────────────────────────────────────────

    def _empty_table(self, teams: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "team": teams, "played": 0, "wins": 0, "draws": 0, "losses": 0,
            "gf": 0, "ga": 0, "gd": 0, "points": 0,
        })

    def _update_table(self, tbl: pd.DataFrame, m: dict) -> pd.DataFrame:
        tbl = tbl.copy()
        h, a, hg, ag = m["home_team"], m["away_team"], m["home_goals"], m["away_goals"]
        tbl.loc[tbl["team"] == h, ["played","gf","ga"]] += [1, hg, ag]
        tbl.loc[tbl["team"] == a, ["played","gf","ga"]] += [1, ag, hg]
        if hg > ag:
            tbl.loc[tbl["team"] == h, ["wins","points"]] += [1,3]
            tbl.loc[tbl["team"] == a, "losses"] += 1
        elif ag > hg:
            tbl.loc[tbl["team"] == a, ["wins","points"]] += [1,3]
            tbl.loc[tbl["team"] == h, "losses"] += 1
        else:
            tbl.loc[tbl["team"].isin([h,a]), ["draws","points"]] += [1,1]
        tbl["gd"] = tbl["gf"] - tbl["ga"]
        return tbl

    def _rank_table(self, tbl: pd.DataFrame) -> pd.DataFrame:
        tbl = tbl.copy()
        tbl["fifa_rank"] = tbl["team"].map(
            lambda t: self.tdata["team_to_rank"].get(t, 150))
        tbl = tbl.sort_values(
            ["points","gd","gf","fifa_rank"],
            ascending=[False,False,False,True]
        ).reset_index(drop=True)
        tbl["pos"] = tbl.index + 1
        return tbl.drop(columns=["fifa_rank"])

    def simulate_group(self, group_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        teams = (
            self.tdata["df_groups"][self.tdata["df_groups"]["group"] == group_name]
            .sort_values("position")["nation"].tolist()
        )
        fixtures = self.df_fixtures[self.df_fixtures["group"] == group_name]
        table = self._empty_table(teams)
        matches = []
        for _, row in fixtures.iterrows():
            m = self.predictor.simulate_match(
                row["home_team"], row["away_team"],
                neutral=bool(row.get("neutral", 1)), tournament_weight=5)
            m["group"] = group_name
            m["venue"] = f"{row.get('city','?')}, {row.get('country','?')}"
            m["date"]  = str(row.get("date",""))
            matches.append(m)
            table = self._update_table(table, m)
        ranked = self._rank_table(table)
        ranked["group"] = group_name
        return ranked, pd.DataFrame(matches)

    def simulate_group_stage(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        tables, all_matches = [], []
        for g in sorted(self.tdata["df_groups"]["group"].unique()):
            t, m = self.simulate_group(g)
            tables.append(t); all_matches.append(m)
        return pd.concat(tables, ignore_index=True), pd.concat(all_matches, ignore_index=True)

    # ── Best third-placed ─────────────────────────────────────────────────────

    def get_qualifiers(self, group_tables: pd.DataFrame
                       ) -> tuple[dict, pd.DataFrame]:
        """Return direct qualifiers dict + best-8 third placed DataFrame."""
        direct: dict[str, str] = {}
        for _, row in group_tables[group_tables["pos"].isin([1,2])].iterrows():
            key = f"{int(row['pos'])}{row['group']}"
            direct[key] = row["team"]

        third = group_tables[group_tables["pos"] == 3].copy()
        third["fifa_rank"] = third["team"].map(
            lambda t: self.tdata["team_to_rank"].get(t, 150))
        best8 = third.sort_values(
            ["points","gd","gf","fifa_rank"],
            ascending=[False,False,False,True]
        ).head(8).reset_index(drop=True)
        best8["third_rank"] = best8.index + 1
        return direct, best8

    # ── R32 bracket builder ───────────────────────────────────────────────────

    def _assign_third_place(
        self, df_r32: pd.DataFrame, best8: pd.DataFrame
    ) -> dict[str, str]:
        """Assign best-8 third-placed teams to their R32 slots."""
        # Collect all slots that start with "3"
        third_slots = []
        for col in ["home_slot","away_slot"]:
            if col in df_r32.columns:
                for v in df_r32[col].astype(str).unique():
                    if v.startswith("3") and v not in third_slots:
                        third_slots.append(v)

        third_teams = best8[["team","group"]].to_dict("records")

        # Try every permutation (up to 8! = 40320, but slots ≤ 8 so fine)
        for perm in itertools.permutations(third_teams, len(third_slots)):
            used = set(); valid = True; assignment = {}
            for slot, info in zip(third_slots, perm):
                allowed = set(slot[1:])          # e.g. "3ABCDF" → {A,B,C,D,F}
                if info["group"] not in allowed or info["team"] in used:
                    valid = False; break
                assignment[slot] = info["team"]; used.add(info["team"])
            if valid:
                return assignment
        # Fallback: assign in order ignoring group constraint
        return {s: t["team"] for s, t in zip(third_slots, third_teams)}

    def build_r32_bracket(
        self, direct: dict, best8: pd.DataFrame
    ) -> pd.DataFrame:
        df_r32 = self.tdata["df_knockout"][
            self.tdata["df_knockout"]["round"] == "R32"].copy()
        third_assign = self._assign_third_place(df_r32, best8)

        rows = []
        for _, row in df_r32.iterrows():
            m = row.to_dict()
            for side in ["home","away"]:
                slot = str(row[f"{side}_slot"])
                if slot in direct:
                    m[f"{side}_team"] = direct[slot]
                elif slot.startswith("3"):
                    m[f"{side}_team"] = third_assign.get(slot, "TBD")
                else:
                    m[f"{side}_team"] = "TBD"
            rows.append(m)
        return pd.DataFrame(rows)

    # ── Knockout rounds ───────────────────────────────────────────────────────

    def simulate_round(self, df_round: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df_round.iterrows():
            m = self.predictor.simulate_knockout_match(
                row["home_team"], row["away_team"], neutral=True,
                tournament_weight=5)
            for col in ["match_id","round","winner_advances_to","loser_advances_to",
                        "match_date","match_time"]:
                if col in row.index:
                    m[col] = row[col]
            results.append(m)
        return pd.DataFrame(results)

    def build_next_round(self, prev_results: pd.DataFrame,
                         next_round: str) -> pd.DataFrame:
        df_next  = self.tdata["df_knockout"][
            self.tdata["df_knockout"]["round"] == next_round].copy()
        teams_by_match: dict[Any, list[str]] = {}
        for _, row in prev_results.iterrows():
            adv = row.get("winner_advances_to")
            if pd.notna(adv):
                teams_by_match.setdefault(adv, []).append(row["winner"])

        rows = []
        for _, row in df_next.iterrows():
            teams = teams_by_match.get(row["match_id"], ["TBD","TBD"])
            m = row.to_dict()
            m["home_team"] = teams[0] if len(teams) > 0 else "TBD"
            m["away_team"] = teams[1] if len(teams) > 1 else "TBD"
            rows.append(m)
        return pd.DataFrame(rows)

    def simulate_knockout_stage(
        self, r32_bracket: pd.DataFrame
    ) -> tuple[pd.DataFrame, str, str]:
        r32  = self.simulate_round(r32_bracket)
        r16  = self.simulate_round(self.build_next_round(r32,  "R16"))
        qf   = self.simulate_round(self.build_next_round(r16,  "QF"))
        sf   = self.simulate_round(self.build_next_round(qf,   "SF"))
        fin  = self.simulate_round(self.build_next_round(sf,   "Final"))
        all_ = pd.concat([r32, r16, qf, sf, fin], ignore_index=True)
        return all_, fin["winner"].iloc[0], fin["loser"].iloc[0]

    # ── Full tournament ───────────────────────────────────────────────────────

    def simulate_tournament(self) -> dict:
        group_tables, group_matches = self.simulate_group_stage()
        direct, best8               = self.get_qualifiers(group_tables)
        r32_bracket                 = self.build_r32_bracket(direct, best8)
        knockout, winner, runner_up = self.simulate_knockout_stage(r32_bracket)

        summary = {
            "winner":     winner,
            "runner_up":  runner_up,
            "r32_teams":  list(direct.values()) + best8["team"].tolist(),
            "r16_teams":  knockout[knockout["round"]=="R32"]["winner"].tolist(),
            "qf_teams":   knockout[knockout["round"]=="R16"]["winner"].tolist(),
            "sf_teams":   knockout[knockout["round"]=="QF"]["winner"].tolist(),
            "final_teams":knockout[knockout["round"]=="SF"]["winner"].tolist(),
        }
        return {
            "summary":       summary,
            "group_tables":  group_tables,
            "group_matches": group_matches,
            "r32_bracket":   r32_bracket,
            "knockout":      knockout,
        }

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    def run_monte_carlo(
        self, n: int = 200,
        callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        wc_teams = self.tdata["wc_teams"]
        counts = {t: {k:0 for k in ["r32","r16","qf","sf","final","winner"]}
                  for t in wc_teams}

        for i in range(n):
            result = self.simulate_tournament()
            s = result["summary"]
            for t in s["r32_teams"]:
                if t in counts: counts[t]["r32"]    += 1
            for t in s["r16_teams"]:
                if t in counts: counts[t]["r16"]    += 1
            for t in s["qf_teams"]:
                if t in counts: counts[t]["qf"]     += 1
            for t in s["sf_teams"]:
                if t in counts: counts[t]["sf"]     += 1
            for t in s["final_teams"]:
                if t in counts: counts[t]["final"]  += 1
            if s["winner"] in counts:
                counts[s["winner"]]["winner"] += 1
            if callback: callback(i+1, n)

        rows = []
        for team, c in counts.items():
            td = self.predictor.tdata
            rows.append({
                "team":          team,
                "confederation": td["team_to_confederation"].get(team,"?"),
                "fifa_rank":     td["team_to_rank"].get(team, 150),
                "elo":           round(td["team_to_elo"].get(team, 1500)),
                "r32_prob":      c["r32"]    / n,
                "r16_prob":      c["r16"]    / n,
                "qf_prob":       c["qf"]     / n,
                "sf_prob":       c["sf"]     / n,
                "final_prob":    c["final"]  / n,
                "winner_prob":   c["winner"] / n,
            })
        return pd.DataFrame(rows).sort_values("winner_prob", ascending=False
                                              ).reset_index(drop=True)
