from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_START_SEASON = 2009
MODEL_START_SEASON = 2010
PRODUCTION_TRAIN_END_SEASON = 2025

TEAM_ABBR_MAP = {
    'SD': 'LAC',
    'OAK': 'LV',
    'STL': 'LA',
}

EWM_SPAN = 5
PRIOR_TEAM_WEIGHT = 0.70
PRIOR_GAME_WEIGHT = 4

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ELO_BONUS = 50.0

PBP_COLUMNS = [
    'game_id',
    'posteam',
    'defteam',
    'epa',
    'success',
    'pass',
    'rush',
    'qb_kneel',
    'qb_spike',
    'two_point_attempt',
    'interception',
    'fumble_lost',
    'sack',
    'qb_dropback',
    'pass_attempt',
    'qb_hit',
    'passer_player_id',
    'passer_player_name',
]

SCHEDULE_COLUMNS = [
    'game_id',
    'season',
    'week',
    'gameday',
    'game_type',
    'home_team',
    'away_team',
    'home_score',
    'away_score',
    'home_rest',
    'away_rest',
    'location',
    'home_qb_id',
    'away_qb_id',
    'home_qb_name',
    'away_qb_name',
]

TEAM_METRICS = [
    'off_epa',
    'off_success_rate',
    'pass_epa',
    'rush_epa',
    'def_epa_allowed',
    'def_success_rate_allowed',
]

TEAM_FEATURE_COLUMNS = [
    "pregame_off_epa_blended",
    "pregame_off_success_rate_blended",
    "pregame_pass_epa_blended",
    "pregame_rush_epa_blended",
    "pregame_def_epa_allowed_blended",
    "pregame_def_success_rate_allowed_blended",
    "off_epa_last3",
    "off_success_rate_last3",
    "pass_epa_last3",
    "rush_epa_last3",
    "def_epa_allowed_last3",
    "def_success_rate_allowed_last3",
    "off_epa_last5",
    "off_success_rate_last5",
    "pass_epa_last5",
    "rush_epa_last5",
    "def_epa_allowed_last5",
    "def_success_rate_allowed_last5",
    "off_epa_ewm",
    "off_success_rate_ewm",
    "pass_epa_ewm",
    "rush_epa_ewm",
    "def_epa_allowed_ewm",
    "def_success_rate_allowed_ewm",
    "pregame_elo",
    "pregame_sos",
    "rest_days",
    "short_rest",
    "prior_games",
    "pregame_turnovers",
    "pregame_takeaways",
    "pregame_off_sack_rate",
    "pregame_def_sack_rate",
    "pregame_qb_epa",
    "pregame_qb_epa_last5",
    "pregame_qb_success_rate",
]

FINAL_FEATURES = [
    f'{side}_{feature}'
    for side in ('home', 'away')
    for feature in TEAM_FEATURE_COLUMNS
] + ['neutral_site']

REQUIRED_NON_NULL_FEATURES = [
    'neutral_site',
    'home_pregame_off_epa_blended',
    'away_pregame_off_epa_blended',
    'home_pregame_elo',
    'away_pregame_elo',
    'home_rest_days',
    'away_rest_days',
    'home_prior_games',
    'away_prior_games',
]

EVALUATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "final_nfl_win_probability_model.joblib"
)
PRODUCTION_MODEL_PATH = PROJECT_ROOT / 'models' / 'production_model.joblib'
MODEL_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'model_dataset_v2.csv'
PREDICTIONS_DIR = PROJECT_ROOT / 'outputs' / 'predictions'
RESULTS_DIR = PROJECT_ROOT / 'outputs' / 'results'