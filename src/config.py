"""Configuration — paths and global constants."""
from __future__ import annotations
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
MATCH_FEATURES_CSV    = DATA_DIR / "df_match_features.csv"
CONFEDERATIONS_CSV    = DATA_DIR / "FIFA_confederations.csv"
GROUP_STAGES_CSV      = DATA_DIR / "group_stages.csv"
WC_FIXTURES_CSV       = DATA_DIR / "wc2026_fixtures.csv"
KNOCKOUT_FIXTURES_CSV = DATA_DIR / "fixtures_knockout_wc2026.csv"
FIFA_RANK_CSV         = DATA_DIR / "wc_2026_48_teams_fifa_rank_change_corrected.csv"
WC_GROUPS_CSV         = DATA_DIR / "wc_2026_groups.csv"
TEAMS_SNAPSHOT_CSV    = DATA_DIR / "wc_2026_teams_snapshot.csv"

# ── Model artifacts ───────────────────────────────────────────────────────────
BEST_MODEL_PKL        = MODEL_DIR / "best_model.pkl"
ALL_MODELS_PKL        = MODEL_DIR / "all_models.pkl"
FEATURE_COLS_PKL      = MODEL_DIR / "feature_columns.pkl"
TEAM_STATS_PKL        = MODEL_DIR / "team_stats_cache.pkl"
METADATA_JSON         = MODEL_DIR / "model_metadata.json"
PERM_IMP_CSV          = MODEL_DIR / "permutation_importance.csv"

# ── Temporal constants ────────────────────────────────────────────────────────
import pandas as pd
CUTOFF          = pd.Timestamp("2026-06-01")   # no data after this
MODERN_ERA_FROM = pd.Timestamp("1993-01-01")   # only use matches from this date
RECENT_2Y       = pd.Timestamp("2024-06-01")   # last-2-year threshold
RECENT_5Y       = pd.Timestamp("2021-06-01")   # last-5-year threshold

# ── Sample weights ────────────────────────────────────────────────────────────
W_RECENT_2Y  = 4.0   # matches in last 2 years
W_RECENT_5Y  = 2.0   # matches 2–5 years ago
W_OLDER      = 1.0   # everything older

# ── Confederation strength (calibration multipliers) ─────────────────────────
CONFEDERATION_STRENGTH: dict[str, float] = {
    "UEFA":     1.22,   # Europe — highest quality, most depth
    "CONMEBOL": 1.10,   # South America — strong but smaller pool
    "CONCACAF": 0.93,   # North/Central America
    "CAF":      0.91,   # Africa
    "AFC":      0.88,   # Asia
    "OFC":      0.80,   # Oceania
    "Unknown":  0.88,
}

# ── Calibration weights ───────────────────────────────────────────────────────
CONF_CALIBRATION   = 0.35   # how strongly confederation edge shifts probs
RANK_CALIBRATION   = 0.28   # FIFA rank edge
FORM_CALIBRATION   = 0.22   # recent 2-year form edge
H2H_CALIBRATION    = 0.10   # head-to-head edge
DRAW_TEMPERATURE   = 1.08   # slight boost to draw probability

# ── Static confederation membership (~210 teams) ─────────────────────────────
STATIC_CONFEDERATION_MAP: dict[str, str] = {
    # UEFA
    "Albania":"UEFA","Andorra":"UEFA","Armenia":"UEFA","Austria":"UEFA",
    "Azerbaijan":"UEFA","Belarus":"UEFA","Belgium":"UEFA",
    "Bosnia and Herzegovina":"UEFA","Bulgaria":"UEFA","Croatia":"UEFA",
    "Cyprus":"UEFA","Czech Republic":"UEFA","Czechoslovakia":"UEFA",
    "Denmark":"UEFA","England":"UEFA","Estonia":"UEFA",
    "Faroe Islands":"UEFA","Finland":"UEFA","France":"UEFA",
    "Georgia":"UEFA","German DR":"UEFA","Germany":"UEFA",
    "Gibraltar":"UEFA","Greece":"UEFA","Hungary":"UEFA","Iceland":"UEFA",
    "Israel":"UEFA","Italy":"UEFA","Kazakhstan":"UEFA","Kosovo":"UEFA",
    "Latvia":"UEFA","Liechtenstein":"UEFA","Lithuania":"UEFA",
    "Luxembourg":"UEFA","Malta":"UEFA","Moldova":"UEFA","Monaco":"UEFA",
    "Montenegro":"UEFA","Netherlands":"UEFA","North Macedonia":"UEFA",
    "Northern Ireland":"UEFA","Norway":"UEFA","Poland":"UEFA",
    "Portugal":"UEFA","Republic of Ireland":"UEFA","Romania":"UEFA",
    "Russia":"UEFA","San Marino":"UEFA","Scotland":"UEFA",
    "Serbia":"UEFA","Slovakia":"UEFA","Slovenia":"UEFA","Spain":"UEFA",
    "Sweden":"UEFA","Switzerland":"UEFA","Turkey":"UEFA","Ukraine":"UEFA",
    "Wales":"UEFA","Yugoslavia":"UEFA",
    # CAF
    "Algeria":"CAF","Angola":"CAF","Benin":"CAF","Botswana":"CAF",
    "Burkina Faso":"CAF","Burundi":"CAF","Cameroon":"CAF",
    "Cape Verde":"CAF","Central African Republic":"CAF","Chad":"CAF",
    "Comoros":"CAF","Congo":"CAF","DR Congo":"CAF","Djibouti":"CAF",
    "Egypt":"CAF","Equatorial Guinea":"CAF","Eritrea":"CAF",
    "Eswatini":"CAF","Ethiopia":"CAF","Gabon":"CAF","Gambia":"CAF",
    "Ghana":"CAF","Guinea":"CAF","Guinea-Bissau":"CAF",
    "Ivory Coast":"CAF","Kenya":"CAF","Lesotho":"CAF","Liberia":"CAF",
    "Libya":"CAF","Madagascar":"CAF","Malawi":"CAF","Mali":"CAF",
    "Mauritania":"CAF","Mauritius":"CAF","Morocco":"CAF",
    "Mozambique":"CAF","Namibia":"CAF","Niger":"CAF","Nigeria":"CAF",
    "Rwanda":"CAF","Senegal":"CAF","Seychelles":"CAF","Sierra Leone":"CAF",
    "Somalia":"CAF","South Africa":"CAF","South Sudan":"CAF","Sudan":"CAF",
    "Tanzania":"CAF","Togo":"CAF","Tunisia":"CAF","Uganda":"CAF",
    "Zambia":"CAF","Zimbabwe":"CAF","Zanzibar":"CAF",
    # AFC
    "Afghanistan":"AFC","Australia":"AFC","Bahrain":"AFC",
    "Bangladesh":"AFC","Bhutan":"AFC","Brunei":"AFC","Cambodia":"AFC",
    "China PR":"AFC","Guam":"AFC","Hong Kong":"AFC","India":"AFC",
    "Indonesia":"AFC","Iran":"AFC","Iraq":"AFC","Japan":"AFC",
    "Jordan":"AFC","Kuwait":"AFC","Kyrgyzstan":"AFC","Laos":"AFC",
    "Lebanon":"AFC","Macau":"AFC","Malaysia":"AFC","Maldives":"AFC",
    "Mongolia":"AFC","Myanmar":"AFC","Nepal":"AFC","North Korea":"AFC",
    "Oman":"AFC","Pakistan":"AFC","Palestine":"AFC","Philippines":"AFC",
    "Qatar":"AFC","Saudi Arabia":"AFC","Singapore":"AFC",
    "South Korea":"AFC","Sri Lanka":"AFC","Syria":"AFC","Taiwan":"AFC",
    "Tajikistan":"AFC","Thailand":"AFC","Timor-Leste":"AFC",
    "Turkmenistan":"AFC","United Arab Emirates":"AFC","Uzbekistan":"AFC",
    "Vietnam":"AFC","Yemen":"AFC",
    # CONCACAF
    "Antigua and Barbuda":"CONCACAF","Aruba":"CONCACAF",
    "Bahamas":"CONCACAF","Barbados":"CONCACAF","Belize":"CONCACAF",
    "Bermuda":"CONCACAF","Canada":"CONCACAF","Cayman Islands":"CONCACAF",
    "Costa Rica":"CONCACAF","Cuba":"CONCACAF","Curacao":"CONCACAF",
    "Curaçao":"CONCACAF","Dominica":"CONCACAF","Dominican Republic":"CONCACAF",
    "El Salvador":"CONCACAF","Grenada":"CONCACAF","Guatemala":"CONCACAF",
    "Guyana":"CONCACAF","Haiti":"CONCACAF","Honduras":"CONCACAF",
    "Jamaica":"CONCACAF","Martinique":"CONCACAF","Mexico":"CONCACAF",
    "Montserrat":"CONCACAF","Nicaragua":"CONCACAF","Panama":"CONCACAF",
    "Puerto Rico":"CONCACAF","Saint Kitts and Nevis":"CONCACAF",
    "Saint Lucia":"CONCACAF","Saint Vincent and the Grenadines":"CONCACAF",
    "Suriname":"CONCACAF","Trinidad and Tobago":"CONCACAF",
    "United States":"CONCACAF",
    # CONMEBOL
    "Argentina":"CONMEBOL","Bolivia":"CONMEBOL","Brazil":"CONMEBOL",
    "Chile":"CONMEBOL","Colombia":"CONMEBOL","Ecuador":"CONMEBOL",
    "Paraguay":"CONMEBOL","Peru":"CONMEBOL","Uruguay":"CONMEBOL",
    "Venezuela":"CONMEBOL",
    # OFC
    "American Samoa":"OFC","Cook Islands":"OFC","Fiji":"OFC",
    "New Caledonia":"OFC","New Zealand":"OFC","Papua New Guinea":"OFC",
    "Samoa":"OFC","Solomon Islands":"OFC","Tahiti":"OFC","Tonga":"OFC",
    "Vanuatu":"OFC",
}

CLASS_NAMES   = ["away_win", "draw", "home_win"]
CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASS_NAMES)}
