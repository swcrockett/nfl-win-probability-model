import argparse
import joblib
import pandas as pd
from sklearn.base import clone

from .config import (
    EVALUATION_MODEL_PATH,
    FINAL_FEATURES,
    MODEL_DATA_PATH,
    MODEL_START_SEASON,
    PRODUCTION_MODEL_PATH,
    PRODUCTION_TRAIN_END_SEASON,
)

def train_production_model(
    through_season=PRODUCTION_TRAIN_END_SEASON,
    evaluation_model_path=EVALUATION_MODEL_PATH,
    data_path=MODEL_DATA_PATH,
    output_path=PRODUCTION_MODEL_PATH,
):
    '''Clone the selected evaluation model and refit it on all available seasons.'''
    evaluation_artifact = joblib.load(evaluation_model_path)
    model_dataset = pd.read_csv(data_path)

    features = list(evaluation_artifact['features'])
    target = evaluation_artifact['target']

    if features != FINAL_FEATURES:
        raise ValueError(
            f"The evaluation artifact's feature order does not match config.FINAL_FEATURES."
        )

    missing_columns = [
        column
        for column in ['season', target, *features]
        if column not in model_dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            f"The model dataset is missing columns: {missing_columns}")

    production_data = model_dataset.loc[
        model_dataset['season'].between(MODEL_START_SEASON, through_season)     
    ].copy()
    if production_data.empty:
        raise ValueError('No rows matched the requested production training seasons.')
    if production_data[target].isna().any():
        raise ValueError('The production target contains missing values.')
    if production_data[target].nunique() != 2:
        raise ValueError('The production target must contain both binary classes.')

    production_model = clone(evaluation_artifact['model'])
    production_model.fit(
        production_data[features],
        production_data[target]
    )

    production_artifact = {
        'model': production_model,
        'model_name': evaluation_artifact.get(
            'model_name', type(production_model).__name__
        ),
        'model_parameters': production_model.get_params(),
        'features': features,
        'target': target,
        'positive_class': evaluation_artifact.get('positive_class', 'home_win'),
        'decision_threshold': evaluation_artifact.get('decision_threshold', 0.5),
        'training_seasons': f'{MODEL_START_SEASON}-{through_season}',
        'training_games': len(production_data),
        'dataset_version': evaluation_artifact.get('dataset_version', 'model_dataset_v2'),
        'artifact_role': 'production',
        'source_evaluation_artifact': evaluation_model_path.name,
        'probability_definition': 'predict_proba(X)[:, 1] is the probability that the home team wins',
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_artifact, output_path)
    return production_artifact, output_path

def main():
    parser = argparse.ArgumentParser(
        description='Retrain the frozen NFL model on all completed historical seasons.'
    )
    parser.add_argument(
        '--through-season',
        type=int,
        default=PRODUCTION_TRAIN_END_SEASON,
        help=f'Last season to include in production training (default: 2025).',     
    )
    args = parser.parse_args()

    artifact, output_path = train_production_model(args.through_season)
    print(
        f'Saved {artifact['model_name']} trained on '
        f'{artifact['training_games']:,} games ({artifact['training_seasons']}) to:\n',
        f'{output_path}'
    )
if __name__ == '__main__':
    main()