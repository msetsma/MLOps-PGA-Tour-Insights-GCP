import pandas as pd
import numpy as np

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
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", prediction)

def preprocess_data_for_player(csv_file_path: str, player_id: int, features: List[str]):
    """
    Load data from a CSV file, filter by player_id, and prepare it for inference.
    """

    data = pd.read_csv(csv_file_path)
    player_data = data[data["player id"] == player_id].sort_values(by="date")
    if player_data.empty:
        raise ValueError(f"No data available for Player {player_id}.")

    feature_data = player_data[features].values

    # Convert to a JSON-serializable format (list of lists)
    return feature_data.tolist()


if __name__ == "__main__":
    # Configuration
    CSV_FILE_PATH = "season_2016_eval_data.csv"  
    PLAYER_ID = 4848  # Justin Thomas
    FEATURES = ['sg_putt', 'sg_arg', 'sg_ott', 'sg_t2g']

    try:
        # builds the expected values for an actual player!
        preprocessed_data = preprocess_data_for_player(
            csv_file_path=CSV_FILE_PATH,
            player_id=PLAYER_ID,
            features=FEATURES,
        )

        print("Prepared data for inference:")
        pprint(preprocessed_data)
        
        predict_custom_trained_model_sample(
            project="925092514777",
            endpoint_id="304319529800957952",
            location="us-central1",
            instances=preprocessed_data
        )
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    
    