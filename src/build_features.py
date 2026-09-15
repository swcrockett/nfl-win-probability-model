import numpy as np
import pandas as pd

from .config import (
    TEAM_ABBR_MAP,
    EWM_SPAN,
    PRIOR_TEAM_WEIGHT,
    PRIOR_GAME_WEIGHT,
    INITIAL_ELO,
    K_FACTOR,
    HOME_ELO_BONUS,
    FINAL_FEATURES,
)

TEAM_METRICS = [
    "off_epa",
    "off_success_rate",
    "pass_epa",
    "rush_epa",
    "def_epa_allowed",
    "def_success_rate_allowed",
]

def build_completed_team_games(pbp, completed_schedule):
    schedule = completed_schedule.copy()
    schedule[["home_team", "away_team"]] = (
        schedule[["home_team", "away_team"]].replace(TEAM_ABBR_MAP)
    )

    if schedule[["home_score", "away_score"]].isna().any().any():
        raise ValueError(
            "completed_schedule must contain only completed games."
        )

    plays = pbp.loc[
        pbp["game_id"].isin(schedule["game_id"])
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & pbp["epa"].notna()
        & pbp["success"].notna()
        & (pbp["pass"].eq(1) | pbp["rush"].eq(1))
        & pbp["qb_kneel"].fillna(0).ne(1)
        & pbp["qb_spike"].fillna(0).ne(1)
        & pbp["two_point_attempt"].fillna(0).ne(1)
    ].copy()

    plays[["posteam", "defteam"]] = (
        plays[["posteam", "defteam"]].replace(TEAM_ABBR_MAP)
    )

    plays["turnover"] = (
        plays["interception"].fillna(0)
        + plays["fumble_lost"].fillna(0)
    ).clip(upper=1)

    offense = (
        plays.groupby(["game_id", "posteam"], as_index=False)
        .agg(
            off_epa=("epa", "mean"),
            off_success_rate=("success", "mean"),
            turnovers=("turnover", "sum"),
            sacks_allowed=("sack", "sum"),
            dropbacks=("qb_dropback", "sum"),
        )
        .rename(columns={"posteam": "team"})
    )

    offense["off_sack_rate"] = (
        offense["sacks_allowed"]
        / offense["dropbacks"].replace(0, np.nan)
    )
    offense = offense.drop(columns=["sacks_allowed", "dropbacks"])

    defense = (
        plays.groupby(["game_id", "defteam"], as_index=False)
        .agg(
            def_epa_allowed=("epa", "mean"),
            def_success_rate_allowed=("success", "mean"),
            takeaways=("turnover", "sum"),
            sacks=("sack", "sum"),
            dropbacks=("qb_dropback", "sum"),
        )
        .rename(columns={"defteam": "team"})
    )

    defense["def_sack_rate"] = (
        defense["sacks"]
        / defense["dropbacks"].replace(0, np.nan)
    )
    defense = defense.drop(columns=["sacks", "dropbacks"])

    passing = (
        plays.loc[plays["pass"].eq(1)]
        .groupby(["game_id", "posteam"], as_index=False)
        .agg(pass_epa=("epa", "mean"))
        .rename(columns={"posteam": "team"})
    )

    rushing = (
        plays.loc[plays["rush"].eq(1)]
        .groupby(["game_id", "posteam"], as_index=False)
        .agg(rush_epa=("epa", "mean"))
        .rename(columns={"posteam": "team"})
    )

    points = pd.concat(
        [
            schedule[
                ["game_id", "home_team", "home_score", "away_score"]
            ].rename(
                columns={
                    "home_team": "team",
                    "home_score": "points_scored",
                    "away_score": "points_allowed",
                }
            ),
            schedule[
                ["game_id", "away_team", "away_score", "home_score"]
            ].rename(
                columns={
                    "away_team": "team",
                    "away_score": "points_scored",
                    "home_score": "points_allowed",
                }
            ),
        ],
        ignore_index=True,
    )

    points["win"] = np.select(
        [
            points["points_scored"] > points["points_allowed"],
            points["points_scored"] < points["points_allowed"],
        ],
        [1.0, 0.0],
        default=0.5,
    )

    team_games = create_target_team_rows(schedule)

    for frame in [offense, passing, rushing, defense, points]:
        team_games = team_games.merge(
            frame,
            on=["game_id", "team"],
            how="left",
            validate="one_to_one",
        )

    return team_games

def add_recent_features(team_games):
    team_games = (
        team_games.sort_values(["gameday", "game_id", "team"])
        .reset_index(drop=True)
        .copy()
    )

    for metric in TEAM_METRICS:
        grouped = team_games.groupby(["season", "team"])[metric]

        for window in (3, 5):
            team_games[f"{metric}_last{window}"] = grouped.transform(
                lambda x: (
                    x.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
            )

        team_games[f"{metric}_ewm"] = grouped.transform(
            lambda x: (
                x.shift(1)
                .ewm(span=EWM_SPAN, adjust=False, min_periods=1)
                .mean()
            )
        )

    for metric in [
        "turnovers",
        "takeaways",
        "off_sack_rate",
        "def_sack_rate",
    ]:
        team_games[f"pregame_{metric}"] = (
            team_games.groupby(["season", "team"])[metric]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

    return team_games

def add_previous_season_priors(team_games):
    team_games = team_games.copy()

    history = team_games.loc[
        team_games["points_scored"].notna()
        & team_games["points_allowed"].notna()
    ]

    previous_season = (
        history.groupby(["season", "team"], as_index=False)[TEAM_METRICS]
        .mean()
    )

    previous_season["season"] += 1

    previous_season = previous_season.rename(
        columns={
            metric: f"{metric}_previous_season"
            for metric in TEAM_METRICS
        }
    )

    league_previous = (
        history.groupby("season", as_index=False)[TEAM_METRICS]
        .mean()
    )

    league_previous["season"] += 1

    league_previous = league_previous.rename(
        columns={
            metric: f"league_{metric}_previous"
            for metric in TEAM_METRICS
        }
    )

    team_games = team_games.merge(
        previous_season,
        on=["season", "team"],
        how="left",
        validate="many_to_one",
    ).merge(
        league_previous,
        on="season",
        how="left",
        validate="many_to_one",
    )

    for metric in TEAM_METRICS:
        league_previous_value = team_games[
            f"league_{metric}_previous"
        ]

        team_previous_value = team_games[
            f"{metric}_previous_season"
        ].fillna(league_previous_value)

        team_games[f"{metric}_prior"] = (
            PRIOR_TEAM_WEIGHT * team_previous_value
            + (1 - PRIOR_TEAM_WEIGHT) * league_previous_value
        )

    return team_games

def add_blended_features(team_games):
    team_games = (
        team_games.sort_values(["gameday", "game_id", "team"])
        .reset_index(drop=True)
        .copy()
    )

    team_games["prior_games"] = (
        team_games.groupby(["season", "team"]).cumcount()
    )

    for metric in TEAM_METRICS:
        current_column = f"pregame_{metric}_current"

        team_games[current_column] = (
            team_games.groupby(["season", "team"])[metric]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

        team_games[f"pregame_{metric}_blended"] = (
            PRIOR_GAME_WEIGHT * team_games[f"{metric}_prior"]
            + team_games["prior_games"]
            * team_games[current_column].fillna(0)
        ) / (
            PRIOR_GAME_WEIGHT + team_games["prior_games"]
        )

    return team_games

def add_elo_features(team_games, completed_schedule, target_games):
    completed = completed_schedule.copy()
    completed["_completed"] = True

    targets = target_games.copy()
    targets["_completed"] = False

    games = pd.concat([completed, targets], ignore_index=True)

    if games["game_id"].duplicated().any():
        raise ValueError(
            "Completed and target games must have distinct game IDs."
        )

    games[["home_team", "away_team"]] = (
        games[["home_team", "away_team"]].replace(TEAM_ABBR_MAP)
    )

    games["gameday"] = pd.to_datetime(games["gameday"])

    games["neutral_site"] = (
        games["location"]
        .fillna("")
        .str.lower()
        .eq("neutral")
        .astype(int)
    )

    games = games.sort_values(["gameday", "game_id"])

    ratings = {}
    rows = []

    for game in games.to_dict("records"):
        home_team = game["home_team"]
        away_team = game["away_team"]

        home_elo = ratings.get(home_team, INITIAL_ELO)
        away_elo = ratings.get(away_team, INITIAL_ELO)

        rows.extend(
            [
                {
                    "game_id": game["game_id"],
                    "team": home_team,
                    "pregame_elo": home_elo,
                    "opponent_pregame_elo": away_elo,
                },
                {
                    "game_id": game["game_id"],
                    "team": away_team,
                    "pregame_elo": away_elo,
                    "opponent_pregame_elo": home_elo,
                },
            ]
        )

        if not game["_completed"]:
            continue

        effective_home_elo = home_elo + (
            0 if game["neutral_site"] else HOME_ELO_BONUS
        )

        expected_home = 1 / (
            1 + 10 ** ((away_elo - effective_home_elo) / 400)
        )

        if game["home_score"] > game["away_score"]:
            result = 1.0
        elif game["home_score"] < game["away_score"]:
            result = 0.0
        else:
            result = 0.5

        change = K_FACTOR * (result - expected_home)

        ratings[home_team] = home_elo + change
        ratings[away_team] = away_elo - change

    elo = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "team",
            "pregame_elo",
            "opponent_pregame_elo",
        ],
    )

    return team_games.merge(
        elo,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

def add_sos_features(team_games):
    team_games = (
        team_games.sort_values(["gameday", "game_id", "team"])
        .reset_index(drop=True)
        .copy()
    )

    team_games["pregame_sos"] = (
        team_games.groupby(["season", "team"])["opponent_pregame_elo"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return team_games

def add_qb_features(
    team_games,
    pbp,
    completed_schedule,
    target_games,
    qb_overrides=None,
):
    qb_games = (
        pbp.loc[
            pbp["qb_dropback"].eq(1)
            & pbp["passer_player_id"].notna()
        ]
        .groupby(["game_id", "passer_player_id"], as_index=False)
        .agg(
            qb_epa=("epa", "mean"),
            qb_success_rate=("success", "mean"),
        )
        .merge(
            completed_schedule[["game_id", "gameday"]],
            on="game_id",
            how="inner",
            validate="many_to_one",
        )
    )

    schedule = pd.concat(
        [completed_schedule, target_games],
        ignore_index=True,
    )

    schedule[["home_team", "away_team"]] = (
        schedule[["home_team", "away_team"]].replace(TEAM_ABBR_MAP)
    )

    starter_frames = []

    for side in ("home", "away"):
        starter_frames.append(
            schedule.reindex(
                columns=[
                    "game_id",
                    "gameday",
                    f"{side}_team",
                    f"{side}_qb_id",
                    f"{side}_qb_name",
                ]
            ).rename(
                columns={
                    f"{side}_team": "team",
                    f"{side}_qb_id": "starting_qb_id",
                    f"{side}_qb_name": "starting_qb_name",
                }
            )
        )

    starters = pd.concat(starter_frames, ignore_index=True)
    target_ids = set(target_games["game_id"])

    for (game_id, team), qb_id in (qb_overrides or {}).items():
        team = TEAM_ABBR_MAP.get(team, team)

        mask = (
            starters["game_id"].eq(game_id)
            & starters["team"].eq(team)
        )

        if game_id not in target_ids or mask.sum() != 1:
            raise ValueError(
                "QB override does not match one target team: "
                f"{(game_id, team)}"
            )

        starters.loc[mask, "starting_qb_id"] = qb_id
        starters.loc[mask, "starting_qb_name"] = pd.NA

    # Upcoming games have no QB statistics yet. Their placeholder rows
    # let the shifted calculations include the latest completed game.
    target_qbs = (
        starters.loc[
            starters["game_id"].isin(target_ids)
            & starters["starting_qb_id"].notna(),
            ["game_id", "gameday", "starting_qb_id"],
        ]
        .rename(columns={"starting_qb_id": "passer_player_id"})
        .drop_duplicates(["game_id", "passer_player_id"])
        .assign(qb_epa=np.nan, qb_success_rate=np.nan)
    )

    qb_games = pd.concat(
        [qb_games, target_qbs],
        ignore_index=True,
    )

    qb_games["gameday"] = pd.to_datetime(qb_games["gameday"])

    qb_games = qb_games.sort_values(
        ["passer_player_id", "gameday", "game_id"]
    ).reset_index(drop=True)

    grouped = qb_games.groupby("passer_player_id")

    qb_games["pregame_qb_epa"] = grouped["qb_epa"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    qb_games["pregame_qb_epa_last5"] = grouped["qb_epa"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    qb_games["pregame_qb_success_rate"] = (
        grouped["qb_success_rate"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    starters = (
        starters.drop(columns="gameday")
        .merge(
            qb_games[
                [
                    "game_id",
                    "passer_player_id",
                    "pregame_qb_epa",
                    "pregame_qb_epa_last5",
                    "pregame_qb_success_rate",
                ]
            ],
            left_on=["game_id", "starting_qb_id"],
            right_on=["game_id", "passer_player_id"],
            how="left",
            validate="many_to_one",
        )
        .drop(columns="passer_player_id")
    )

    return team_games.merge(
        starters,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

def create_target_team_rows(target_games):
    games = target_games.copy()

    games[["home_team", "away_team"]] = (
        games[["home_team", "away_team"]].replace(TEAM_ABBR_MAP)
    )

    games["gameday"] = pd.to_datetime(games["gameday"])

    games["neutral_site"] = (
        games["location"]
        .fillna("")
        .str.lower()
        .eq("neutral")
        .astype(int)
    )

    base = [
        "season",
        "week",
        "gameday",
        "game_id",
        "neutral_site",
    ]

    frames = []

    for side, opponent_side, is_home in [
        ("home", "away", 1),
        ("away", "home", 0),
    ]:
        rows = games[base].copy()

        rows["team"] = games[f"{side}_team"]
        rows["opponent"] = games[f"{opponent_side}_team"]
        rows["rest_days"] = games[f"{side}_rest"]
        rows["is_home"] = is_home
        rows["short_rest"] = rows["rest_days"].lt(6).astype(int)

        frames.append(rows)

    target_team_rows = pd.concat(frames, ignore_index=True)

    return (
        target_team_rows
        .sort_values(["gameday", "game_id", "team"])
        .reset_index(drop=True)
    )

def convert_team_rows_to_matchups(team_games, target_games):
    feature_columns = [
        name.removeprefix("home_")
        for name in FINAL_FEATURES
        if name.startswith("home_")
    ]

    game_rows = target_games.copy()

    game_rows[["home_team", "away_team"]] = (
        game_rows[["home_team", "away_team"]].replace(TEAM_ABBR_MAP)
    )

    game_rows["gameday"] = pd.to_datetime(game_rows["gameday"])

    game_rows["neutral_site"] = (
        game_rows["location"]
        .fillna("")
        .str.lower()
        .eq("neutral")
        .astype(int)
    )

    game_rows = game_rows[
        [
            "season",
            "week",
            "gameday",
            "game_id",
            "home_team",
            "away_team",
            "neutral_site",
        ]
    ]

    for side, is_home in [("home", 1), ("away", 0)]:
        features = team_games.loc[
            team_games["game_id"].isin(game_rows["game_id"])
            & team_games["is_home"].eq(is_home),
            ["game_id", "team", *feature_columns],
        ].rename(
            columns={
                "team": f"{side}_team",
                **{
                    name: f"{side}_{name}"
                    for name in feature_columns
                },
            }
        )

        game_rows = game_rows.merge(
            features,
            on=["game_id", f"{side}_team"],
            how="left",
            validate="one_to_one",
        )

    missing = set(FINAL_FEATURES) - set(game_rows.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    metadata = [
        "season",
        "week",
        "gameday",
        "game_id",
        "home_team",
        "away_team",
    ]

    return game_rows[metadata + list(FINAL_FEATURES)]