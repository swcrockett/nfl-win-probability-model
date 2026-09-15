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

    completed_games, target_games = split_schedule(schedule, season, week)

    history = build_completed_team_games(pbp, completed_games)
    target_rows = create_target_team_rows(target_games)

    team_games = pd.concat([history, target_rows], ignore_index=True)

    team_games = add_recent_features(team_games)
    team_games = add_previous_season_priors(team_games)
    team_games = add_blended_features(team_games)
    team_games = add_elo_features(team_games, completed_games, target_games)
    team_games = add_sos_features(team_games)
    team_games = add_qb_features(team_games, pbp, completed_games, target_games)

    return convert_team_rows_to_matchups(team_games, target_games)

def predict_week(season, week):
    artifact = joblib.load('models/production_model.joblib')

    model = artifact['model']
    features = artifact['features']

    games = build_week_features(season, week)
    X_week = games.reindex(columns=features)

    games['home_win_probability'] = model.predict_proba(X_week)[:, 1]

    games['away_win_probability'] = 1 - games['home_win_probability']

    games['predicted_winner'] = np.where(
        games['home_win_probability'] >= 0.5,
        games['home_team'],
        games['away_team'],
    )
    
    return games