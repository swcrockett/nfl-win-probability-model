import warnings
import numpy as np
import pandas as pd

from .config import (
    EWM_SPAN,
    FINAL_FEATURES,
    HOME_ELO_BONUS,
    INITIAL_ELO,
    K_FACTOR,
    PRIOR_GAME_WEIGHT,
    PRIOR_TEAM_WEIGHT,
    REQUIRED_NON_NULL_FEATURES,
    TEAM_ABBR_MAP,
    TEAM_FEATURE_COLUMNS,
    TEAM_METRICS,
)

def create_target_team_rows(target_games):
    '''Convert each scheduled game to one home-team row and one away-team row.'''
    games = target_games.copy()
    games[['home_team', 'away_team']] = games[
        ['home_team', 'away_team']
    ].replace(TEAM_ABBR_MAP)
    games['gameday'] = pd.to_datetime(games['gameday'])
    games['neutral_site'] = (
        games['location'].fillna('').astype(str).str.lower().eq('neutral').astype(int)
    )

    base_columns = [
        'season',
        'week',
        'gameday',
        'game_id',
        'neutral_site',
    ]
    team_frames = []

    for side, opponent_side, is_home in (
        ('home', 'away', 1),
        ('away', 'home', 0),
    ):
        rows = games[base_columns].copy()
        rows['team'] = games[f'{side}_team']
        rows['opponent'] = games[f'{opponent_side}_team']
        rows['rest_days'] = games[f'{side}_rest']
        rows['is_home'] = is_home
        rows['short_rest'] = rows['rest_days'].lt(6).astype(int)
        team_frames.append(rows)

    return (
        pd.concat(team_frames, ignore_index=True)
        .sort_values(['gameday', 'game_id', 'team'])
        .reset_index(drop=True)
    )

def build_completed_team_games(pbp, completed_schedule):
    '''Reproduce Notebook 4's per-team, per-game raw statistics.'''
    schedule = completed_schedule.copy()
    schedule[['home_team', 'away_team']] = schedule[
        ['home_team', 'away_team']
    ].replace(TEAM_ABBR_MAP)

    if schedule[['home_score', 'away_score']].isna().any().any():
        raise ValueError('completed_schedule contains an unplayed game.')

    plays = pbp.loc[
        pbp['game_id'].isin(schedule['game_id'])
        & pbp['posteam'].notna()
        & pbp['defteam'].notna()
        & pbp['epa'].notna()
        & pbp['success'].notna()
        & (pbp['pass'].eq(1) | pbp['rush'].eq(1))
        & pbp['qb_kneel'].fillna(0).ne(1)
        & pbp['qb_spike'].fillna(0).ne(1)
        & pbp['two_point_attempt'].fillna(0).ne(1)
    ].copy()
    plays[['posteam', 'defteam']] = plays[
        ['posteam', 'defteam']
    ].replace(TEAM_ABBR_MAP)

    plays['turnover'] = (
        plays['interception'].fillna(0)
        + plays['fumble_lost'].fillna(0)
    ).clip(upper=1)

    offense = (
        plays.groupby(['game_id', 'posteam'], as_index=False)
        .agg(
            off_epa=('epa', 'mean'),
            off_success_rate=('success', 'mean'),
            turnovers=('turnover', 'sum'),
            sacks_allowed=('sack', 'sum'),
            dropbacks=('qb_dropback', 'sum'),
        )
        .rename(columns={'posteam': 'team'})
    )
    offense['off_sack_rate'] = (
        offense['sacks_allowed']
        / offense['dropbacks'].replace(0, np.nan)
    )
    offense = offense.drop(columns=['sacks_allowed', 'dropbacks'])

    passing = (
        plays.loc[plays['pass'].eq(1)]
        .groupby(['game_id', 'posteam'], as_index=False)
        .agg(pass_epa=('epa', 'mean'))
        .rename(columns={'posteam': 'team'})
    )

    rushing = (
        plays.loc[plays['rush'].eq(1)]
        .groupby(['game_id', 'posteam'], as_index=False)
        .agg(rush_epa=('epa', 'mean'))
        .rename(columns={'posteam': 'team'})
    )

    defense = (
        plays.groupby(['game_id', 'defteam'], as_index=False)
        .agg(
            def_epa_allowed=('epa', 'mean'),
            def_success_rate_allowed=('success', 'mean'),
            takeaways=('turnover', 'sum'),
            sacks=('sack', 'sum'),
            dropbacks=('qb_dropback', 'sum'),
        )
        .rename(columns={'defteam': 'team'})
    )
    defense['def_sack_rate'] = (
        defense['sacks']
        / defense['dropbacks'].replace(0, np.nan)
    )
    defense = defense.drop(columns=['sacks', 'dropbacks'])

    points = pd.concat(
        [
            schedule[
                ['game_id', 'home_team', 'home_score', 'away_score']
            ].rename(
                columns={
                    'home_team': 'team',
                    'home_score': 'points_scored',
                    'away_score': 'points_allowed',
                }
            ),
            schedule[
                ['game_id', 'away_team', 'away_score', 'home_score']
            ].rename(
                columns={
                    'away_team': 'team',
                    'away_score': 'points_scored',
                    'home_score': 'points_allowed',
                }
            ),
        ],
        ignore_index=True,
    )
    points["win"] = np.select(
        [
            points['points_scored'] > points['points_allowed'],
            points['points_scored'] < points['points_allowed'],
        ],
        [1.0, 0.0],
        default=0.5,
    )

    team_games = create_target_team_rows(schedule)
    for feature_frame in (offense, passing, rushing, defense, points):
        if feature_frame.duplicated(['game_id', 'team']).any():
            raise ValueError('A raw feature table contains duplicate team-game rows.')
        team_games = team_games.merge(
            feature_frame,
            on=['game_id', 'team'],
            how='left',
            validate='one_to_one',
        )

    return (
        team_games.sort_values(['gameday', 'game_id', 'team'])
        .reset_index(drop=True)
    )

def add_recent_features(team_games):
    '''Add shifted last-3, last-5, EWM, turnover, and sack features.'''
    result = (
        team_games.sort_values(['gameday', 'game_id', 'team'])
        .reset_index(drop=True)
        .copy()
    )

    for metric in TEAM_METRICS:
        grouped = result.groupby(['season', 'team'])[metric]
        result[f'{metric}_last3'] = grouped.transform(
            lambda values: values.shift(1).rolling(3, min_periods=1).mean()
        )
        result[f'{metric}_last5'] = grouped.transform(
            lambda values: values.shift(1).rolling(5, min_periods=1).mean()
        )
        result[f'{metric}_ewm'] = grouped.transform(
            lambda values: values.shift(1).ewm(
                span=EWM_SPAN,
                adjust=False,
                min_periods=1,
            ).mean()
        )

    for metric in (
        'turnovers',
        'takeaways',
        'off_sack_rate',
        'def_sack_rate',
    ):
        result[f'pregame_{metric}'] = (
            result.groupby(['season', 'team'])[metric]
            .transform(lambda values: values.shift(1).expanding().mean())
        )

    return result

def add_previous_season_priors(team_games):
    '''Add the 70% team / 30% league regressed prior from Notebook 4.'''
    result = team_games.copy()
    history = result.loc[result['points_scored'].notna()].copy()

    previous_season = (
        history.groupby(['season', 'team'], as_index=False)[TEAM_METRICS]
        .mean()
    )
    previous_season['season'] += 1
    previous_season = previous_season.rename(
        columns={
            metric: f'{metric}_previous_season'
            for metric in TEAM_METRICS
        }
    )

    league_previous = (
        history.groupby('season', as_index=False)[TEAM_METRICS]
        .mean()
    )
    league_previous['season'] += 1
    league_previous = league_previous.rename(
        columns={
            metric: f'league_{metric}_previous'
            for metric in TEAM_METRICS
        }
    )

    result = result.merge(
        previous_season,
        on=['season', 'team'],
        how='left',
        validate='many_to_one',
    ).merge(
        league_previous,
        on='season',
        how='left',
        validate='many_to_one',
    )

    for metric in TEAM_METRICS:
        league_value = result[f'league_{metric}_previous']
        team_value = result[f'{metric}_previous_season'].fillna(league_value)
        result[f'{metric}_prior'] = (
            PRIOR_TEAM_WEIGHT * team_value
            + (1.0 - PRIOR_TEAM_WEIGHT) * league_value
        )

    return result

def add_blended_features(team_games):
    '''Blend four pseudo-games of prior with current-season pregame form.'''
    result = (
        team_games.sort_values(['gameday', 'game_id', 'team'])
        .reset_index(drop=True)
        .copy()
    )
    result['prior_games'] = result.groupby(['season', 'team']).cumcount()

    for metric in TEAM_METRICS:
        current_column = f'pregame_{metric}_current'
        result[current_column] = (
            result.groupby(['season', 'team'])[metric]
            .transform(lambda values: values.shift(1).expanding().mean())
        )
        result[f'pregame_{metric}_blended'] = (
            PRIOR_GAME_WEIGHT * result[f'{metric}_prior']
            + result['prior_games'] * result[current_column].fillna(0)
        ) / (PRIOR_GAME_WEIGHT + result['prior_games'])

    return result

def add_elo_features(team_games, completed_schedule, target_games):
    '''Replay Elo chronologically, then read—but do not update—target ratings.'''
    completed = completed_schedule.copy()
    completed['completed_flag'] = True
    targets = target_games.copy()
    targets['completed_flag'] = False
    games = pd.concat([completed, targets], ignore_index=True)

    if games['game_id'].duplicated().any():
        raise ValueError('Completed games and target games overlap.')

    games[['home_team', 'away_team']] = games[
        ['home_team', 'away_team']
    ].replace(TEAM_ABBR_MAP)
    games['gameday'] = pd.to_datetime(games['gameday'])
    games['neutral_site'] = (
        games['location'].fillna('').astype(str).str.lower().eq('neutral').astype(int)
    )
    games = games.sort_values(['gameday', 'game_id']).reset_index(drop=True)

    ratings = {}
    elo_rows = []

    for game in games.itertuples(index=False):
        home_elo = ratings.get(game.home_team, INITIAL_ELO)
        away_elo = ratings.get(game.away_team, INITIAL_ELO)

        elo_rows.extend(
            [
                {
                    'game_id': game.game_id,
                    'team': game.home_team,
                    'pregame_elo': home_elo,
                    'opponent_pregame_elo': away_elo,
                },
                {
                    'game_id': game.game_id,
                    'team': game.away_team,
                    'pregame_elo': away_elo,
                    'opponent_pregame_elo': home_elo,
                },
            ]
        )

        if not game.completed_flag:
            continue

        effective_home_elo = home_elo + (
            0.0 if game.neutral_site else HOME_ELO_BONUS
        )
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_elo - effective_home_elo) / 400.0)
        )

        if game.home_score > game.away_score:
            home_result = 1.0
        elif game.home_score < game.away_score:
            home_result = 0.0
        else:
            home_result = 0.5

        elo_change = K_FACTOR * (home_result - expected_home)
        ratings[game.home_team] = home_elo + elo_change
        ratings[game.away_team] = away_elo - elo_change

    elo = pd.DataFrame(elo_rows)
    return team_games.merge(
        elo,
        on=['game_id', 'team'],
        how='left',
        validate='one_to_one',
    )

def add_sos_features(team_games):
    '''Add current season pregame strength of schedule.'''
    results = (
        team_games.sort_values(['gameday', 'game_id', 'team'])
        .reset_index(drop=True)
        .copy()
    )
    results['pregame_sos'] = (
        results.groupby(['season', 'team'])['opponent_pregame_elo']
        .transform(lambda values: values.shift(1).expanding().mean())
    )
    return results

def _make_starter_rows(schedule):
    frames = []
    for side in ('home', 'away'):
        frames.append(
            schedule[
                [
                    'game_id',
                    'gameday',
                    f'{side}_team',
                    f'{side}_qb_id',
                    f'{side}_qb_name',
                ]
            ].rename(
                columns={
                    f'{side}_team': 'team',
                    f'{side}_qb_id': 'starting_qb_id',
                    f'{side}_qb_name': 'starting_qb_name',
                }
            )
        )
    return pd.concat(frames, ignore_index=True)

def _assign_target_qbs(completed_starters, target_starters, qb_overrides):
    '''Use override, then listed QB, then the team's latest known starter.'''
    targets = target_starters.copy()
    targets['starting_qb_id'] = targets['starting_qb_id'].astype('object')
    targets['starting_qb_name'] = targets['starting_qb_name'].astype('object')

    known = completed_starters.loc[
        completed_starters['starting_qb_id'].notna()
    ].sort_values(['gameday', 'game_id'])
    latest_by_team = (
        known.drop_duplicates('team', keep='last')
        .set_index('team')[['starting_qb_id', 'starting_qb_name']]
    )

    for index, row in targets.iterrows():
        team = row['team']
        game_team_key = (row['game_id'], team)
        override = qb_overrides.get(game_team_key, qb_overrides.get(team))

        if override is not None:
            targets.at[index, 'starting_qb_id'] = override
            targets.at[index, 'starting_qb_name'] = pd.NA
        elif pd.isna(row['starting_qb_id']) and team in latest_by_team.index:
            targets.at[index, 'starting_qb_id'] = latest_by_team.at[
                team, 'starting_qb_id'
            ]
            targets.at[index, 'starting_qb_name'] = latest_by_team.at[
                team, 'starting_qb_name'
            ]

    unresolved = targets.loc[
        targets['starting_qb_id'].isna(), ['game_id', 'team']
    ]
    if not unresolved.empty:
        warnings.warn(
            'No starting QB could be assigned for: '
            + ', '.join(
                f'{row.team} ({row.game_id})'
                for row in unresolved.itertuples(index=False)
            )
            + '. QB features will be missing; pass a QB override if needed.',
            stacklevel=2,
        )

    return targets

def add_qb_features(
    team_games,
    pbp,
    completed_schedule,
    target_games,
    qb_overrides=None,
):
    '''Add career-to-date and last-five features for each scheduled starter.'''
    completed_ids = set(completed_schedule['game_id'])
    qb_games = (
        pbp.loc[
            pbp['game_id'].isin(completed_ids)
            & pbp['qb_dropback'].eq(1)
            & pbp['passer_player_id'].notna()
        ]
        .groupby(['game_id', 'passer_player_id'], as_index=False)
        .agg(
            qb_epa=('epa', 'mean'),
            qb_success_rate=('success', 'mean'),
        )
        .merge(
            completed_schedule[['game_id', 'gameday']],
            on='game_id',
            how='inner',
            validate='many_to_one',
        )
    )

    completed = completed_schedule.copy()
    targets = target_games.copy()
    for frame in (completed, targets):
        frame[['home_team', 'away_team']] = frame[
            ['home_team', 'away_team']
        ].replace(TEAM_ABBR_MAP)
        frame['gameday'] = pd.to_datetime(frame['gameday'])

    completed_starters = _make_starter_rows(completed)
    target_starters = _make_starter_rows(targets)
    target_starters = _assign_target_qbs(
        completed_starters,
        target_starters,
        qb_overrides or {},
    )
    starters = pd.concat(
        [completed_starters, target_starters],
        ignore_index=True,
    )

    target_qb_rows = (
        target_starters.loc[
            target_starters['starting_qb_id'].notna(),
            ['game_id', 'gameday', 'starting_qb_id'],
        ]
        .rename(columns={'starting_qb_id': 'passer_player_id'})
        .drop_duplicates(['game_id', 'passer_player_id'])
        .assign(qb_epa=np.nan, qb_success_rate=np.nan)
    )
    qb_games = pd.concat([qb_games, target_qb_rows], ignore_index=True)
    qb_games['gameday'] = pd.to_datetime(qb_games['gameday'])
    qb_games = qb_games.sort_values(
        ['passer_player_id', 'gameday', 'game_id']
    ).reset_index(drop=True)

    qb_group = qb_games.groupby('passer_player_id')
    qb_games['pregame_qb_epa'] = qb_group['qb_epa'].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    qb_games['pregame_qb_epa_last5'] = qb_group['qb_epa'].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    qb_games['pregame_qb_success_rate'] = qb_group[
        'qb_success_rate'
    ].transform(lambda values: values.shift(1).expanding().mean())

    starters = (
        starters.drop(columns='gameday')
        .merge(
            qb_games[
                [
                    'game_id',
                    'passer_player_id',
                    'pregame_qb_epa',
                    'pregame_qb_epa_last5',
                    'pregame_qb_success_rate',
                ]
            ],
            left_on=['game_id', 'starting_qb_id'],
            right_on=['game_id', 'passer_player_id'],
            how='left',
            validate='many_to_one',
        )
        .drop(columns='passer_player_id')
    )

    return team_games.merge(
        starters,
        on=['game_id', 'team'],
        how='left',
        validate='one_to_one',
    )

def convert_team_rows_to_matchups(team_games, target_games):
    '''Convert the two target team rows into the model's 73-column game row.'''
    games = target_games.copy()
    games[['home_team', 'away_team']] = games[
        ['home_team', 'away_team']
    ].replace(TEAM_ABBR_MAP)
    games['gameday'] = pd.to_datetime(games['gameday'])
    games['neutral_site'] = (
        games['location'].fillna("").astype(str).str.lower().eq("neutral").astype(int)
    )
    games = games[
        [
            'season',
            'week',
            'gameday',
            'game_id',
            'home_team',
            'away_team',
            'neutral_site',
        ]
    ].copy()

    target_ids = set(games['game_id'])
    for side, is_home in (("home", 1), ("away", 0)):
        side_features = team_games.loc[
            team_games['game_id'].isin(target_ids)
            & team_games['is_home'].eq(is_home),
            ['game_id', 'team', *TEAM_FEATURE_COLUMNS],
        ].copy()
        side_features = side_features.rename(
            columns={
                'team': f'{side}_team',
                **{
                    feature: f'{side}_{feature}'
                    for feature in TEAM_FEATURE_COLUMNS
                },
            }
        )
        games = games.merge(
            side_features,
            on=['game_id', f'{side}_team'],
            how='left',
            validate='one_to_one',
        )

    metadata = [
        'season',
        'week',
        'gameday',
        'game_id',
        'away_team',
        'home_team',
    ]
    return games[metadata + FINAL_FEATURES]

def validate_future_features(future_games, model_features, expected_games):
    '''Validate row count, model schema, core values, and numeric finiteness.'''
    if len(future_games) != expected_games:
        raise ValueError(
            f'Built {len(future_games)} rows for {expected_games} scheduled games.'
        )
    if future_games['game_id'].duplicated().any():
        raise ValueError('Duplicate game IDs were produced.')

    configured = list(FINAL_FEATURES)
    artifact_features = list(model_features)
    if artifact_features != configured:
        raise ValueError(
            'The production artifact\'s feature order does not match config.FINAL_FEATURES.'
        )

    missing_columns = [
        feature for feature in artifact_features if feature not in future_games.columns
    ]
    if missing_columns:
        raise ValueError(f'Missing model features: {missing_columns}')

    core_missing = future_games[REQUIRED_NON_NULL_FEATURES].isna().sum()
    core_missing = core_missing[core_missing.gt(0)]
    if not core_missing.empty:
        raise ValueError(f'Required features contain missing values:\n{core_missing}')

    model_matrix = future_games.reindex(columns=artifact_features).apply(
        pd.to_numeric,
        errors='raise',
    )
    if np.isinf(model_matrix.to_numpy(dtype=float)).any():
        raise ValueError('At least one model feature is infinite.')

    return model_matrix
