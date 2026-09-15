import nflreadpy as nfl
import pandas as pd
from src.config import DATA_START_SEASON, TEAM_ABBR_MAP

def load_nfl_data(prediction_season, pbp_columns, schedule_columns):
    seasons = list(range(DATA_START_SEASON, prediction_season + 1))

    schedule = (
        nfl.load_schedules(seasons)
        .select(schedule_columns)
        .to_pandas()
    )
    pbp = (
        nfl.load_pbp_data(seasons)
        .select(pbp_columns)
        .to_pandas()
    )

    return pbp, schedule

pbp[['posteam', 'defteam']] = (
    pbp[['posteam', 'defteam']]
    .replace(TEAM_ABBR_MAP)
)
schedule[['home_team', 'away_team']] = (
    schedule[['home_team', 'away_team']]
    .replace(TEAM_ABBR_MAP)
)
schedule['gameday'] = pd.to_datetime(schedule['gameday'])