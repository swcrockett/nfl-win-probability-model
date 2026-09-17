import pandas as pd
import nflreadpy as nfl

from .config import (
    DATA_START_SEASON,
    PBP_COLUMNS,
    SCHEDULE_COLUMNS,
    TEAM_ABBR_MAP
)

def normalize_team_abbreviations(frame):
    result = frame.copy()
    team_columns = [
        column
        for column in ('posteam', 'defteam', 'home_team', 'away_team')
        if column in result.columns
    ]
    if team_columns:
        result[team_columns] = result[team_columns].replace(TEAM_ABBR_MAP)
    return result

def load_schedule_data(prediction_season):

    seasons = list(range(DATA_START_SEASON, prediction_season + 1))
    schedule = (
        nfl.load_schedules(seasons)
        .select(SCHEDULE_COLUMNS)
        .to_pandas()
    )
    schedule = load_schedule_data(prediction_season)
    pbp = normalize_team_abbreviations(pbp)
    return pbp, schedule

def split_schedule(schedule, season, week):
    target_games = schedule.loc[
        schedule['season'].eq(season)
        & schedule['week'].eq(week)
    ].copy()

    if target_games.empty:
        raise ValueError(f"No games found for {season} Week {week}")

    before_target_week = (
        schedule['season'].lt(season) 
        | (
            schedule['season'].eq(season)
            & schedule['week'].lt(week)
        )
    )

    completed_games = schedule.loc[
        before_target_week
        & schedule['home_score'].notna()
        & schedule['away_score'].notna()
    ].copy()

    if completed_games.empty:
        raise ValueError(f"No completed games were available before the target week.")

    completed_games = completed_games.sort_values(
        ['gameday', 'game_id']
    ).reset_index(drop=True)
    target_games = target_games.sort_values(
        ['gameday', 'game_id']
    ).reset_index(drop=True)

    return completed_games, target_games