import joblib

DATA_START_SEASON = 2009
MODEL_START_SEASON = 2010

TEAM_ABBR_MAP = {
    'SD': 'LAC',
    'OAK': 'LV',
    'STL': 'LA',
}

EWM_SPAN = 5
PRIOR_TEAM_WEIGHT = 0.70
PRIOR_GAME_WEIGHT = 4

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ELO_BONUS = 50

artifact = joblib.load(
    './models/final_nfl_win_probability_model.joblib')

FINAL_FEATURES = artifact['features']