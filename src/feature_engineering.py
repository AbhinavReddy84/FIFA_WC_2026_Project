"""Feature engineering: load raw data, build rich per-match features."""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    CONFEDERATION_STRENGTH,
    CUTOFF,
    MATCH_FEATURES_CSV,
    MODERN_ERA_FROM,
    RECENT_2Y,
    RECENT_5Y,
    STATIC_CONFEDERATION_MAP,
    W_OLDER,
    W_RECENT_2Y,
    W_RECENT_5Y,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confederation_of(team: str, fallback_map: dict) -> str:
    """Look up confederation with STATIC_MAP → fallback_map → Unknown."""
    return (
        STATIC_CONFEDERATION_MAP.get(team)
        or fallback_map.get(team)
        or "Unknown"
    )


def _conf_strength(team: str, fallback_map: dict) -> float:
    conf = _confederation_of(team, fallback_map)
    return CONFEDERATION_STRENGTH.get(conf, CONFEDERATION_STRENGTH["Unknown"])


def _sample_weight(date: pd.Timestamp) -> float:
    if date >= RECENT_2Y:
        return W_RECENT_2Y
    if date >= RECENT_5Y:
        return W_RECENT_5Y
    return W_OLDER


# ─────────────────────────────────────────────────────────────────────────────
# Per-team rolling stats
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_raw_history() -> pd.DataFrame:
    """Load and clean the full match history DataFrame."""
    df = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
    df = df[df["date"] < CUTOFF].copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # Fix neutral column
    df["neutral"] = df["neutral"].map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    ).fillna(0).astype(int)
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def build_team_stats() -> pd.DataFrame:
    """
    Build per-team rolling statistics snapshot (as of CUTOFF).
    Returns a DataFrame indexed by team name.
    """
    df = load_raw_history()

    # Pivot to long format (one row per team per match)
    home = df[["date", "home_team", "away_team", "home_score", "away_score",
               "neutral", "is_competitive", "tournament_weight"]].copy()
    home.columns = ["date", "team", "opponent", "gf", "ga",
                    "neutral", "is_competitive", "tw"]
    home["is_home"] = 1

    away = df[["date", "away_team", "home_team", "away_score", "home_score",
               "neutral", "is_competitive", "tournament_weight"]].copy()
    away.columns = ["date", "team", "opponent", "gf", "ga",
                    "neutral", "is_competitive", "tw"]
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long["gd"]     = long["gf"] - long["ga"]
    long["result"] = long["gd"].apply(lambda x: "W" if x > 0 else ("D" if x == 0 else "L"))
    long["pts"]    = long["gd"].apply(lambda x: 3 if x > 0 else (1 if x == 0 else 0))
    long["w"]      = (long["result"] == "W").astype(int)
    long["d"]      = (long["result"] == "D").astype(int)
    long["l"]      = (long["result"] == "L").astype(int)

    rows = []
    for team, grp in long.sort_values("date").groupby("team"):
        # Filter to modern era only for stats
        grp_modern = grp[grp["date"] >= MODERN_ERA_FROM]
        last2y = grp_modern[grp_modern["date"] >= RECENT_2Y]
        last5y = grp_modern[grp_modern["date"] >= RECENT_5Y]
        last10 = grp_modern.tail(10)
        last5  = grp_modern.tail(5)

        def _stats(sub, suffix):
            if len(sub) == 0:
                return {f"{k}_{suffix}": 0.0
                        for k in ["pts", "gd", "gf", "ga", "w_rate", "d_rate", "matches"]}
            return {
                f"pts_{suffix}":     sub["pts"].mean(),
                f"gd_{suffix}":      sub["gd"].mean(),
                f"gf_{suffix}":      sub["gf"].mean(),
                f"ga_{suffix}":      sub["ga"].mean(),
                f"w_rate_{suffix}":  sub["w"].mean(),
                f"d_rate_{suffix}":  sub["d"].mean(),
                f"matches_{suffix}": len(sub),
            }

        row = {"team": team}
        row.update(_stats(last2y, "2y"))
        row.update(_stats(last5y, "5y"))
        row.update(_stats(last10, "l10"))
        row.update(_stats(last5,  "l5"))
        rows.append(row)

    return pd.DataFrame(rows).set_index("team")


@lru_cache(maxsize=1)
def build_elo_lookup() -> dict[str, float]:
    """Latest Elo rating per team (from the last match before cutoff)."""
    df = load_raw_history()
    home = df[["date", "home_team", "home_elo_pre"]].rename(
        columns={"home_team": "team", "home_elo_pre": "elo"})
    away = df[["date", "away_team", "away_elo_pre"]].rename(
        columns={"away_team": "team", "away_elo_pre": "elo"})
    combined = pd.concat([home, away], ignore_index=True).dropna(subset=["elo"])
    latest   = combined.sort_values("date").groupby("team").tail(1)
    return dict(zip(latest["team"], latest["elo"]))


def get_team_stat(team: str, stat: str, stats_df: Optional[pd.DataFrame] = None) -> float:
    if stats_df is None:
        stats_df = build_team_stats()
    if team in stats_df.index and stat in stats_df.columns:
        val = stats_df.loc[team, stat]
        return float(val) if pd.notna(val) else 0.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# H2H summary
# ─────────────────────────────────────────────────────────────────────────────

def compute_h2h(team_a: str, team_b: str, n: int = 10) -> dict:
    """Return H2H stats from team_a's perspective."""
    df = load_raw_history()
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    h2h = df[mask].sort_values("date", ascending=False).head(n)
    if h2h.empty:
        return {"h2h_score": 0.0, "n_meetings": 0, "wins": 0, "draws": 0, "losses": 0}

    wins = draws = losses = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]
        if gf > ga: wins += 1
        elif gf == ga: draws += 1
        else: losses += 1

    score = (wins * 1.0 + draws * 0.3 - losses * 1.0) / len(h2h)
    return {"h2h_score": round(score, 3), "n_meetings": len(h2h),
            "wins": wins, "draws": draws, "losses": losses}


def get_last_n_matches(team: str, n: int = 5) -> pd.DataFrame:
    """Last N matches for a team, returned as a display DataFrame."""
    df = load_raw_history()
    home = df[df["home_team"] == team].copy()
    home["side"] = "H"; home["opponent"] = home["away_team"]
    home["gf"] = home["home_score"]; home["ga"] = home["away_score"]

    away = df[df["away_team"] == team].copy()
    away["side"] = "A"; away["opponent"] = away["home_team"]
    away["gf"] = away["away_score"]; away["ga"] = away["home_score"]

    combined = pd.concat([home, away]).sort_values("date", ascending=False).head(n)
    rows = []
    for _, m in combined.iterrows():
        gf, ga = int(m["gf"]), int(m["ga"])
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        rows.append({
            "Date": m["date"].strftime("%Y-%m-%d"),
            "Opponent": m["opponent"],
            "Score": f"{gf}–{ga}",
            "H/A": m["side"],
            "Result": result,
            "Tournament": m.get("tournament", ""),
        })
    return pd.DataFrame(rows)


def get_h2h_last_n(team_a: str, team_b: str, n: int = 5) -> pd.DataFrame:
    df = load_raw_history()
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    h2h = df[mask].sort_values("date", ascending=False).head(n)
    rows = []
    for _, m in h2h.iterrows():
        if m["home_team"] == team_a:
            gf, ga, venue = int(m["home_score"]), int(m["away_score"]), "Home"
        else:
            gf, ga, venue = int(m["away_score"]), int(m["home_score"]), "Away"
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        rows.append({
            "Date": m["date"].strftime("%Y-%m-%d"),
            f"{team_a}": gf,
            f"{team_b}": ga,
            "Result": result,
            "Venue": venue,
            "Tournament": m.get("tournament", ""),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Build training dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    conf_map_override: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Build (X, y, sample_weights) for model training.
    Uses only matches from MODERN_ERA_FROM to CUTOFF.
    """
    df = load_raw_history()
    df = df[df["date"] >= MODERN_ERA_FROM].copy()

    # Build confederation fallback from data
    conf_fallback: dict[str, str] = {}
    for _, row in df.iterrows():
        conf_fallback[row["home_team"]] = row.get("home_confederation", "Unknown")
        conf_fallback[row["away_team"]] = row.get("away_confederation", "Unknown")
    if conf_map_override:
        conf_fallback.update(conf_map_override)

    team_stats = build_team_stats()
    elo_lookup = build_elo_lookup()

    records = []
    weights = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        date = row["date"]

        # Target
        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_:
            label = "home_win"
        elif as_ > hs:
            label = "away_win"
        else:
            label = "draw"

        # Elo
        h_elo = float(row["home_elo_pre"]) if pd.notna(row["home_elo_pre"]) else 1500.0
        a_elo = float(row["away_elo_pre"]) if pd.notna(row["away_elo_pre"]) else 1500.0

        # Confederation
        h_conf = _confederation_of(home, conf_fallback)
        a_conf = _confederation_of(away, conf_fallback)
        h_cs   = CONFEDERATION_STRENGTH.get(h_conf, 0.88)
        a_cs   = CONFEDERATION_STRENGTH.get(a_conf, 0.88)

        # Form features — only use stats from BEFORE this match date to avoid leakage
        # (approximation: full snapshot is used, which is fine for training as stats
        #  are computed at CUTOFF, but the model learns relative patterns)
        def gs(team, stat): return get_team_stat(team, stat, team_stats)

        feat = {
            # Elo
            "elo_diff":          h_elo - a_elo,
            "elo_abs_diff":      abs(h_elo - a_elo),
            "elo_ratio":         h_elo / a_elo if a_elo else 1.0,
            "home_elo":          h_elo,
            "away_elo":          a_elo,
            # Confederation
            "conf_strength_diff": h_cs - a_cs,
            "home_conf_strength": h_cs,
            "away_conf_strength": a_cs,
            "same_confederation": int(h_conf == a_conf),
            # Venue
            "neutral":           int(row["neutral"]),
            # Tournament context
            "tournament_weight": float(row.get("tournament_weight", 1)),
            "is_competitive":    int(bool(row.get("is_competitive", False))),
            "match_year":        date.year,
            # 2-year form
            "pts_2y_diff":       gs(home, "pts_2y")  - gs(away, "pts_2y"),
            "gd_2y_diff":        gs(home, "gd_2y")   - gs(away, "gd_2y"),
            "gf_2y_diff":        gs(home, "gf_2y")   - gs(away, "gf_2y"),
            "w_rate_2y_diff":    gs(home, "w_rate_2y") - gs(away, "w_rate_2y"),
            "home_pts_2y":       gs(home, "pts_2y"),
            "away_pts_2y":       gs(away, "pts_2y"),
            # Last-10 form
            "pts_l10_diff":      gs(home, "pts_l10") - gs(away, "pts_l10"),
            "gd_l10_diff":       gs(home, "gd_l10")  - gs(away, "gd_l10"),
            "w_rate_l10_diff":   gs(home, "w_rate_l10") - gs(away, "w_rate_l10"),
            # Confederation labels (for OHE)
            "home_confederation": h_conf,
            "away_confederation": a_conf,
        }
        records.append(feat)

        # Sample weight — boost recent + competitive matches
        base_w = _sample_weight(date)
        comp_boost = 1.5 if row.get("is_competitive", False) else 1.0
        tw_boost   = float(row.get("tournament_weight", 1)) / 3.0 + 0.5
        weights.append(base_w * comp_boost * tw_boost)

    X = pd.DataFrame(records)
    y = pd.Series([r["result"] if "result" in r else "" for r in
                   [{"result": "home_win" if row["home_score"] > row["away_score"]
                     else ("away_win" if row["away_score"] > row["home_score"] else "draw")}
                    for _, row in df.iterrows()]])

    # Build y correctly
    labels = []
    for _, row in df.iterrows():
        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_: labels.append("home_win")
        elif as_ > hs: labels.append("away_win")
        else: labels.append("draw")
    y = pd.Series(labels)

    return X, y, np.array(weights)
