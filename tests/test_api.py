import pandas as pd
import random

from pprint import pprint
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print("Prediction:", prediction[0])

def preprocess_tournament_data(base_path: str, year: int, player: int = None,
                                tournament: int = None, features: List[str] = None):
    
    csv_file_path = f"{base_path}/{year}.csv"
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        raise ValueError(f"Year {year} data not found at path: {csv_file_path}")
    # Remove rows with no feature data
    data = data.dropna(subset=features)

    # Random player or specific player
    if player is None:
        player = random.choice(data["player"].unique())

    # Filter data for the player
    player_data = data[data["player"] == player]
    if player_data.empty:
        raise ValueError(f"No data available for Player {player} in {year}.")

    # Random tournament or specific tournament
    if tournament is None:
        tournament = random.choice(player_data["tournament name"].unique())

    # Filter data for the tournament
    tournament_data = player_data[player_data["tournament name"] == tournament]
    if tournament_data.empty:
        raise ValueError(f"No data available for Player {player} in Tournament {tournament}.")

    # Sort data by date
    tournament_data = tournament_data.sort_values(by="date")

    actual_finish = tournament_data["pos"].values
    feature_data = tournament_data[features].values

    print(f"Selected Player ID: {player}")
    print(f"Selected Tournament ID: {tournament}")
    print(f"Year: {year}")
    print("Strokes Gained (higher = better)")
    print(f"  putting:   {feature_data[0][0]:7.2f}")
    print(f"  chipping:  {feature_data[0][1]:7.2f}")
    print(f"  driving:   {feature_data[0][2]:7.2f}")
    print(f"  approach:  {feature_data[0][3]:7.2f}")
    print(f"Finish: {actual_finish[0]}")

    return feature_data.tolist()


if __name__ == "__main__":
    FEATURES = ['sg_putt', 'sg_arg', 'sg_ott', 'sg_t2g']

    try:
        base_path = "seasons"
        result = preprocess_tournament_data(base_path, year=2022, features=FEATURES)

        predict_custom_trained_model_sample(
            project="925092514777", 
            endpoint_id="8293142318802796544",
            location="us-central1",
            instances=result)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    
    