import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import PREDICTIONS_DIR, RESULTS_DIR
from .load_data import load_schedule_data


def _metrics_row(data, season, scope, week=pd.NA):
    y_true = data['home_win'].astype(int)
    probability = data['home_win_probability'].astype(float)
    predicted = data['predicted_home_win'].astype(int)

    roc_auc = (
        roc_auc_score(y_true, probability)
        if y_true.nunique() == 2
        else np.nan
    )
    return {
        'season': season,
        'scope': scope,
        'week': week,
        'games': len(data),
        'correct': int((predicted == y_true).sum()),
        'accuracy': accuracy_score(y_true, predicted),
        'log_loss': log_loss(y_true, probability, labels=[0, 1]),
        'brier_score': brier_score_loss(y_true, probability),
        'roc_auc': roc_auc,
    }


def score_prediction_file(
    season,
    week,
    prediction_path=None,
    allow_partial=False,
    results_directory=RESULTS_DIR,
):
    '''Merge saved probabilities with final scores and update live results files.'''
    if prediction_path is None:
        prediction_path = PREDICTIONS_DIR / f'{season}_week_{week:02d}.csv'
    prediction_path = pd.io.common.stringify_path(prediction_path)
    predictions = pd.read_csv(prediction_path)

    required_prediction_columns = {
        'game_id',
        'season',
        'week',
        'away_team',
        'home_team',
        'home_win_probability',
    }
    missing = required_prediction_columns - set(predictions.columns)
    if missing:
        raise ValueError(f'Prediction file is missing columns: {sorted(missing)}')
    if predictions['game_id'].duplicated().any():
        raise ValueError('Prediction file contains duplicate game IDs.')
    if not predictions['season'].eq(season).all() or not predictions['week'].eq(week).all():
        raise ValueError('Prediction file season/week does not match the requested season/week.')
    if not predictions['home_win_probability'].between(0, 1).all():
        raise ValueError("home_win_probability must be between 0 and 1.")

    schedule = load_schedule_data(season)
    actuals = schedule.loc[
        schedule['season'].eq(season)
        & schedule['week'].eq(week),
        ['game_id', 'home_score', 'away_score'],
    ].copy()

    scored = predictions.merge(
        actuals,
        on='game_id',
        how='left',
        validate='one_to_one',
    )
    incomplete = scored['home_score'].isna() | scored['away_score'].isna()
    if incomplete.any() and not allow_partial:
        incomplete_ids = scored.loc[incomplete, 'game_id'].tolist()
        raise ValueError(
            f'{len(incomplete_ids)} games are not final: {incomplete_ids}. '
            'Run again after the games finish or pass --allow-partial.'
        )
    scored = scored.loc[~incomplete].copy()
    if scored.empty:
        raise ValueError('No completed predicted games are available to score.')

    tied = scored['home_score'].eq(scored['away_score'])
    if tied.any():
        warnings.warn(
            f'Excluding {int(tied.sum())} tied game(s) because the model target is binary.',
            stacklevel=2,
        )
        scored = scored.loc[~tied].copy()
    if scored.empty:
        raise ValueError('No non-tied completed games remain to score.')

    scored['home_win'] = scored['home_score'].gt(scored['away_score']).astype(int)
    scored['predicted_home_win'] = scored['home_win_probability'].ge(0.5).astype(int)
    scored['actual_winner'] = np.where(
        scored['home_win'].eq(1),
        scored['home_team'],
        scored['away_team'],
    )
    scored['predicted_winner'] = np.where(
        scored['predicted_home_win'].eq(1),
        scored['home_team'],
        scored['away_team'],
    )
    scored['correct'] = scored['predicted_home_win'].eq(scored['home_win'])

    results_directory.mkdir(parents=True, exist_ok=True)
    scored_path = results_directory / f'{season}_scored_predictions.csv'
    summary_path = results_directory / f'{season}_results.csv'

    if scored_path.exists():
        prior_scored = pd.read_csv(scored_path)
        prior_scored = prior_scored.loc[
            ~prior_scored['game_id'].isin(scored['game_id'])
        ]
        all_scored = pd.concat([prior_scored, scored], ignore_index=True, sort=False)
    else:
        all_scored = scored.copy()
    all_scored = all_scored.sort_values(['week', 'game_id']).reset_index(drop=True)
    all_scored.to_csv(scored_path, index=False, encoding='utf-8')

    summary_rows = [
        _metrics_row(week_data, season, 'weekly', int(scored_week))
        for scored_week, week_data in all_scored.groupby('week', sort=True)
    ]
    summary_rows.append(_metrics_row(all_scored, season, 'overall'))
    summary = pd.DataFrame(summary_rows)
    summary['week'] = summary['week'].astype('Int64')
    summary.to_csv(summary_path, index=False, encoding='utf-8')

    current_week_metrics = _metrics_row(scored, season, 'weekly', week)
    return scored, pd.DataFrame([current_week_metrics]), summary, scored_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description='Score one saved NFL weekly prediction file.'
    )
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    parser.add_argument('--prediction-file', type=str)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()

    _, week_metrics, summary, scored_path, summary_path = score_prediction_file(
        season=args.season,
        week=args.week,
        prediction_path=args.prediction_file,
        allow_partial=args.allow_partial,
    )
    print('WEEK RESULTS')
    print(week_metrics.to_string(index=False))
    print('\nSEASON RESULTS')
    print(summary.to_string(index=False))
    print(f'\nSaved scored games to: {scored_path}')
    print(f'Saved summary to: {summary_path}')


if __name__ == '__main__':
    main()
