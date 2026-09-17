import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .build_features import (
    add_blended_features,
    add_elo_features,
    add_previous_season_priors,
    add_qb_features,
    add_recent_features,
    add_sos_features,
    build_completed_team_games,
    convert_team_rows_to_matchups,
    create_target_team_rows,
    validate_future_features,
)
from .config import (
    FINAL_FEATURES,
    MODEL_DATA_PATH,
    PREDICTIONS_DIR,
    PRODUCTION_MODEL_PATH,
    TEAM_ABBR_MAP,
)
from .load_data import load_current_data, split_schedule


def build_week_features(season, week, qb_overrides=None):
    '''Build the exact Notebook 4 pregame feature schema for one target week.'''
    pbp, schedule = load_current_data(season)
    completed_games, target_games = split_schedule(schedule, season, week)

    # This explicit filter is the central target-week leakage barrier.
    completed_pbp = pbp.loc[
        pbp['game_id'].isin(completed_games['game_id'])
    ].copy()

    history = build_completed_team_games(completed_pbp, completed_games)
    target_rows = create_target_team_rows(target_games)
    team_games = pd.concat(
        [history, target_rows],
        ignore_index=True,
        sort=False,
    )
    team_games = (
        team_games.sort_values(['gameday', 'game_id', 'team'])
        .reset_index(drop=True)
    )

    team_games = add_recent_features(team_games)
    team_games = add_previous_season_priors(team_games)
    team_games = add_blended_features(team_games)
    team_games = add_elo_features(team_games, completed_games, target_games)
    team_games = add_sos_features(team_games)
    team_games = add_qb_features(
        team_games,
        completed_pbp,
        completed_games,
        target_games,
        qb_overrides=qb_overrides,
    )

    future_games = convert_team_rows_to_matchups(team_games, target_games)
    return future_games


def load_production_artifact(model_path=PRODUCTION_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f'Production model not found at {model_path}. '
            f'Run: python -m src.train_production'
        )
    artifact = joblib.load(model_path)
    required_keys = {'model', 'features', 'target', 'decision_threshold'}
    missing_keys = required_keys - set(artifact)
    if missing_keys:
        raise ValueError(f"Production artifact is missing keys: {sorted(missing_keys)}")
    return artifact


def predict_week(season, week, qb_overrides=None, model_path=PRODUCTION_MODEL_PATH):
    '''Return one clean probability and winner prediction for every target game.'''
    artifact = load_production_artifact(model_path)
    games = build_week_features(season, week, qb_overrides=qb_overrides)
    model_features = list(artifact['features'])
    model_matrix = validate_future_features(
        games,
        model_features=model_features,
        expected_games=len(games),
    )

    model = artifact['model']
    if not hasattr(model, 'predict_proba'):
        raise TypeError('The saved production model does not implement predict_proba().')

    probabilities = model.predict_proba(model_matrix)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f'The model\'s classes do not include home_win=1: {classes}')
    home_class_index = classes.index(1)
    home_probabilities = probabilities[:, home_class_index]

    threshold = float(artifact.get('decision_threshold', 0.5))
    predictions = games[
        ['season', 'week', 'gameday', 'game_id', 'away_team', 'home_team']
    ].copy()
    predictions['away_win_probability'] = 1.0 - home_probabilities
    predictions['home_win_probability'] = home_probabilities
    predictions['predicted_winner'] = np.where(
        predictions['home_win_probability'].ge(threshold),
        predictions['home_team'],
        predictions['away_team'],
    )
    predictions['confidence'] = predictions[
        ['away_win_probability', 'home_win_probability']
    ].max(axis=1)
    return predictions.sort_values(['gameday', 'game_id']).reset_index(drop=True)


def save_predictions(predictions, output_directory=PREDICTIONS_DIR):
    season = int(predictions['season'].iloc[0])
    week = int(predictions['week'].iloc[0])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f'{season}_week_{week:02d}.csv'
    predictions.to_csv(output_path, index=False, encoding='utf-8')
    return output_path


def print_predictions(predictions):
    season = int(predictions['season'].iloc[0])
    week = int(predictions['week'].iloc[0])
    print('=' * 64)
    print(f'{season} NFL WEEK {week} PREDICTIONS'.center(64))
    print('=' * 64)

    for game in predictions.itertuples(index=False):
        print(f'{game.away_team} @ {game.home_team}')
        print(
            f'  {game.away_team}: {game.away_win_probability:.1%} | '
            f'{game.home_team}: {game.home_win_probability:.1%} | '
            f'PICK: {game.predicted_winner}'
        )
    print('=' * 64)


def compare_with_historical_dataset(
    generated_features,
    season,
    week,
    dataset_path=MODEL_DATA_PATH,
    tolerance=1e-8,
):
    '''Compare production-time features with Notebook 4's saved historical rows.'''
    historical = pd.read_csv(dataset_path)
    expected = historical.loc[
        historical['season'].eq(season)
        & historical['week'].eq(week)
    ].copy()

    if expected.empty:
        raise ValueError(f'No historical model rows found for {season} Week {week}.')

    expected_ids = set(expected['game_id'])
    generated_ids = set(generated_features['game_id'])
    if expected_ids != generated_ids:
        raise ValueError(
            'Historical and generated game IDs differ. '
            f'Only historical: {sorted(expected_ids - generated_ids)}; '
            f'only generated: {sorted(generated_ids - expected_ids)}'
        )

    expected = expected.set_index('game_id').sort_index()
    generated = generated_features.set_index('game_id').sort_index()
    report_rows = []

    for feature in FINAL_FEATURES:
        left = pd.to_numeric(expected[feature], errors='raise').to_numpy(dtype=float)
        right = pd.to_numeric(generated[feature], errors='raise').to_numpy(dtype=float)
        matches = np.isclose(left, right, atol=tolerance, rtol=0, equal_nan=True)
        finite_differences = np.abs(left - right)
        max_error = (
            float(np.nanmax(finite_differences))
            if np.isfinite(finite_differences).any()
            else 0.0
        )
        report_rows.append(
            {
                'feature': feature,
                'matches': bool(matches.all()),
                'mismatched_games': int((~matches).sum()),
                'max_absolute_error': max_error,
            }
        )

    return pd.DataFrame(report_rows)


def parse_qb_overrides(values):
    '''Parse repeated TEAM=QB_ID command-line arguments.'''
    overrides = {}
    for value in values or []:
        if '=' not in value:
            raise ValueError(f'Invalid QB override {value!r}; use TEAM=QB_ID.')
        team, qb_id = value.split('=', 1)
        team = TEAM_ABBR_MAP.get(team.strip().upper(), team.strip().upper())
        qb_id = qb_id.strip()
        if not team or not qb_id:
            raise ValueError(f'Invalid QB override {value!r}; use TEAM=QB_ID.')
        overrides[team] = qb_id
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description='Build pregame features and predict one NFL week.'
    )
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    parser.add_argument(
        '--qb-override',
        action='append',
        default=[],
        metavar='TEAM=QB_ID',
        help='Override an expected starter; repeat for multiple teams.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Compare generated features with model_dataset_v2 instead of predicting.',
    )
    args = parser.parse_args()
    overrides = parse_qb_overrides(args.qb_override)

    if args.dry_run:
        generated = build_week_features(args.season, args.week, overrides)
        report = compare_with_historical_dataset(
            generated,
            args.season,
            args.week,
        )
        failures = report.loc[~report['matches']]
        if failures.empty:
            print(f'PASS: all {len(FINAL_FEATURES)} features match for {args.season} Week {args.week}.')
        else:
            print(f'FAIL: {len(failures)} feature columns do not match:')
            print(failures.to_string(index=False))
            raise SystemExit(1)
        return

    predictions = predict_week(args.season, args.week, overrides)
    output_path = save_predictions(predictions)
    print_predictions(predictions)
    print(f'Saved predictions to: {output_path}')


if __name__ == '__main__':
    main()
