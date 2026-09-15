import joblib
import numpy as np
import pandas as pd

from src.build_features import (
    build_completed_team_games,
    create_target_team_rows,
    add_recent_features,
    add_previous_season_priors,
    add_blended_features,
    add_elo_features,
    add_sos_features,
    add_qb_features,
    convert_team_rows_to_matchups,
)

def build_week_features(season, week):
    pbp, schedule = load_current_data(season)