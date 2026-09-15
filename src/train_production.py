from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_nfl_win_probability_model.joblib"
)
PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production_model.joblib"
)
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset_v2.csv"
)
evaluation_artifact = joblib.load(
    EVALUATION_MODEL_PATH
)
model_dataset = pd.read_csv(DATA_PATH)

features = evaluation_artifact["features"]
target = evaluation_artifact["target"]

production_data = model_dataset.loc[
    model_dataset["season"].between(2010, 2025)
].copy()

production_model = clone(evaluation_artifact["model"])

production_model.fit(
    production_data[features],
    production_data[target],
)

production_artifact = {
    'model': production_model,
    'model_name': evaluation_artifact['model_name'],
    'model_parameters': production_model.get_params(),
    'features': features,
    'target': target,
    'positive_class': 'home_win',
    'decision_threshold': evaluation_artifact.get('decision_threshold', 0.5),
    'training_seasons': '2010-2025',
    'data_set_version': 'model_dataset_v2',
    'artifact_role': 'production',
    'probability_definition': ('Probability that the home team wins'),
}

joblib.dump(production_artifact, PRODUCTION_MODEL_PATH)

print(f'Saved production model to: {PRODUCTION_MODEL_PATH}')