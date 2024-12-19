import kfp
from kfp import dsl
from kfp.dsl import component, Dataset, Model, Output, Input
from google.cloud import aiplatform


BASE_IMAGE = "gcr.io/mitchell-setsma-gcp-project/pipeline-image:latest"

# Preprocessing Component
@component(base_image=BASE_IMAGE)
def preprocess_data(gcs_bucket: str, output_data_path: Output[Dataset], eval_data_path: Output[Dataset]):
    """
    Preprocess input data directly and return the processed DataFrame.
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from google.cloud import storage
    import random

    gcs_bucket = "mlops-data-ingestion"

    def list_csv_files(bucket_name: str):
        """List all CSV files in the specified GCS bucket."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs())
        csv_files = [blob.name for blob in blobs if blob.name.endswith(".csv")]
        return csv_files

    def download_csv(bucket_name: str, blob_name: str):
        """Download a CSV file from GCS to a local temporary path."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        local_path = f"/tmp/{blob_name.split('/')[-1]}"
        blob.download_to_filename(local_path)
        return local_path
    
    csv_files = list_csv_files(gcs_bucket)

    # Combine all CSVs into a single DataFrame
    data = pd.DataFrame()
    for csv_file in csv_files:
        local_file_path = download_csv(gcs_bucket, csv_file)
        temp_data = pd.read_csv(local_file_path)
        data = pd.concat([data, temp_data], ignore_index=True)

    # for those that dont make the cut we are giving them a value of the field size plus 1
    data['field_size'] = data.groupby('tournament id')['pos'].transform('count')
    data.loc[data['pos'].isnull(), 'pos'] = data['field_size'] + 1

    # Normalize the data for position
    data['pos_normalized'] = data.groupby('tournament id')['pos'].transform(lambda x: x / x.max())

    # Convert the date column to datetime format if not already
    data['date'] = pd.to_datetime(data['date'])

    # Drop rows where n_rounds or strokes are NaN or 0
    data = data.dropna(subset=['n_rounds', 'strokes'])  
    data = data[(data['n_rounds'] > 0) & (data['strokes'] > 0)]

    # New feature of avg_strokes_per_round
    data['avg_strokes_per_round'] = data['strokes'] / data['n_rounds']

    # Build rolling strokes gained data
    rolling_strokes_gained = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']
    for metric in rolling_strokes_gained:
        data[f'{metric}_rolling_mean'] = data.groupby('player id')[metric].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Normalize and scale the data
    scaler = StandardScaler()
    scaled_columns = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g',
                    'sg_total', 'sg_putt_rolling_mean', 'sg_arg_rolling_mean',
                    'sg_app_rolling_mean', 'sg_ott_rolling_mean', 'sg_t2g_rolling_mean',
                    'sg_total_rolling_mean']

    # Ensure only existing columns are scaled to avoid KeyError
    existing_scaled_columns = [col for col in scaled_columns if col in data.columns]
    if not existing_scaled_columns:
        raise ValueError("No columns to scale in the scaled_columns list.")
    data[existing_scaled_columns] = scaler.fit_transform(data[existing_scaled_columns])

    # Drop unnecessary features
    features_to_drop = ['tournament id', 'hole_par', 'player', 'tournament name', 'course', 'purse', 'no_cut']
    data = data.drop(columns=features_to_drop, errors='ignore')

    # Split off evaluation data by randomly selecting players
    unique_players = data['player id'].unique()
    random.seed(42)
    # save 10% of players for later
    eval_players = random.sample(list(unique_players), int(len(unique_players) * 0.1))  
    eval_data = data[data['player id'].isin(eval_players)]
    train_data = data[~data['player id'].isin(eval_players)]

    # Save train and evaluation datasets
    train_data.to_csv(output_data_path.path, index=False)
    eval_data.to_csv(eval_data_path.path, index=False)


@component(base_image=BASE_IMAGE)
def save_eval_data_to_gcs(eval_data_path: Input[Dataset], eval_data_out: Output[Dataset]):
    """
    Save preprocessed evaluation data to GCS as CSV files.
    """
    import pandas as pd
    from google.cloud import storage

    def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str):
        """Upload a file to GCS."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

    # Load the evaluation data
    eval_data = pd.read_csv(eval_data_path.path)
    season = eval_data["season"].iloc[0]

    # Define the GCS path based on the season
    gcs_season_path = f"season_{season}_eval_data.csv"

    # Save the evaluation data to a temporary file
    local_temp_path = "/tmp/eval_data.csv"
    eval_data.to_csv(local_temp_path, index=False)

    # Upload the file to GCS
    upload_to_gcs(local_temp_path, "pga-tour-pipeline-artifacts", gcs_season_path)
    eval_data.to_csv(eval_data_out.path, index=False)


# model training portion
@component(base_image=BASE_IMAGE)
def train_model(processed_data_path: Input[Dataset], model_output: Output[Model], metrics_output: Output[Dataset]):
    """
    Train a model using the processed data and save the model and metrics.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pandas as pd
    import numpy as np
    import json

    # Sequence creation
    def create_sequences(data, sequence_length, features, target):
        sequences = []
        targets = []
        for _, group in data.groupby('player id'):
            group = group.sort_values(by='date')
            player_data = group[features].values
            player_target = group[target].values
            for i in range(len(player_data) - sequence_length + 1):
                sequences.append(player_data[i:i + sequence_length])
                targets.append(player_target[i + sequence_length - 1])
        return np.array(sequences), np.array(targets)
    
    # Load and prepare data
    data = pd.read_csv(processed_data_path.path)
    features = [
        'made_cut', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 
        'sg_t2g', 'sg_total', 'avg_strokes_per_round', 
        'sg_putt_rolling_mean', 'sg_arg_rolling_mean', 
        'sg_app_rolling_mean', 'sg_ott_rolling_mean', 
        'sg_t2g_rolling_mean', 'sg_total_rolling_mean', 
        'pos_normalized'
    ]
    target = 'pos'
    sequence_length = 5

    # Create sequences
    X, y = create_sequences(data, sequence_length, features, target)
    X_padded = pad_sequences(X, maxlen=sequence_length, padding='pre', dtype='float32', value=0.0)

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # Define model
    model = Sequential([
        Input(shape=(sequence_length, len(features))),
        Masking(mask_value=0.0),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1,  # Adjust for quicker testing
        batch_size=32
    )

    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)

    # Save performance metrics
    metrics = {
        "test_loss": test_loss,
        "test_mae": test_mae,
        "history": history.history,
    }
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics, f)

    model.export(model_output.path)
    print(f"Model saved to: {model_output.path}")


@component(base_image=BASE_IMAGE)
def register_model(
    model_path: Input[Model],
    metrics_path: Input[Dataset],
    model_display_name: str,
    gcs_bucket: str,
    model_resource_name: Output[Dataset],
):
    """
    Registers a trained model with Vertex AI Model Registry and saves metadata to GCS.

    Args:
        model_path (Model): Path to the trained model.
        metrics_path (Dataset): Path to the metrics JSON file.
        model_display_name (str): Display name for the model in Vertex AI.
        gcs_bucket (str): GCS bucket to save metadata.
        model_resource_name (Output[Dataset]): Output path to write the model's resource name.
    """
    from google.cloud import aiplatform, storage
    import json
    import datetime

    aiplatform.init()

    # Load metrics
    with open(metrics_path.path, 'r') as f:
        metrics = json.load(f)

    artifact_uri = model_path.path
    # Convert `/gcs/` to `gs://` if necessary
    if artifact_uri.startswith('/gcs/'):
        artifact_uri = artifact_uri.replace('/gcs/', 'gs://', 1)

    # Generate a unique version identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    versioned_display_name = f"{model_display_name}_v{timestamp}"

    # Register model in Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=versioned_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
    )

    # Write the model resource name to the output
    with open(model_resource_name.path, 'w') as f:
        f.write(model.resource_name)

    #Prepare metadata to save to GCS
    rows_to_insert = {
        "timestamp": timestamp,
        "model_display_name": versioned_display_name,
        "model_resource_name": model.resource_name,
        "test_loss": metrics.get("test_loss", None),
        "test_mae": metrics.get("test_mae", None),
    }

    # Save metadata to GCS
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(f"model-metrics/{versioned_display_name}_metrics.json")
    blob.upload_from_string(
        data=json.dumps(rows_to_insert, indent=4), content_type="application/json"
    )

    # Logs for debugging
    print(f"Model registered with ID: {model.resource_name}")
    print(f"Saved metrics to: gs://{gcs_bucket}/model-metrics/{versioned_display_name}_metrics.json")


@component(base_image=BASE_IMAGE)
def deploy_model_to_endpoint(
    model_resource_name: Input[Dataset],
    endpoint_display_name: str
):
    from google.cloud import aiplatform

    aiplatform.init()

    # Read the model resource name from the input
    with open(model_resource_name.path, 'r') as f:
        model_name = f.read().strip()

    # Check if the endpoint exists
    endpoints = aiplatform.Endpoint.list(
        filter=f"display_name={endpoint_display_name}"
    )

    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.resource_name}")
    else:
        # Create a new endpoint if it doesn't exist
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name
        )
        print(f"Created new endpoint: {endpoint.resource_name}")

    # Undeploy existing models if any
    if endpoint.traffic_split:
        for deployed_model_id in endpoint.traffic_split.keys():
            endpoint.undeploy(deployed_model_id=deployed_model_id)
            print(f"Undeployed model: {deployed_model_id}")

    # Deploy the new model
    model = aiplatform.Model(model_name)
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{endpoint_display_name}_model",
        machine_type="n1-standard-4",
    )
    print(f"Deployed model to endpoint: {endpoint.resource_name}")


@component(base_image=BASE_IMAGE)
def evaluate_deployed_model(eval_data_path: Input[Dataset], endpoint_path: Input[Dataset]):
    """
    evaluate newly deployed pipeline
    """
    print(f"Testing model: {eval_data_path.path} & {endpoint_path.path}")

    # add evaluation logic here


# Pipeline Definition
def vertex_pipeline():
    @dsl.pipeline(name="pga-tour-pipeline")
    def pipeline(gcs_bucket: str):
        preprocess_task = preprocess_data(
            gcs_bucket=gcs_bucket
        )

        eval_data = save_eval_data_to_gcs(
            eval_data_path=preprocess_task.outputs["eval_data_path"]
        )

        train_task = train_model(
            processed_data_path=preprocess_task.outputs["output_data_path"]
        )

        register_task = register_model(
            model_path=train_task.outputs["model_output"],
            metrics_path=train_task.outputs["metrics_output"],
            model_display_name="pga-tour-model",
            gcs_bucket="pga-tour-pipeline-artifacts"
        )

        deploy_task = deploy_model_to_endpoint(
            model_resource_name=register_task.outputs["model_resource_name"],
            endpoint_display_name="pga-tour-endpoint"
        )

        evaluate_model = evaluate_deployed_model(
            eval_data_path=eval_data.outputs["eval_data_out"],
            endpoint_path=register_task.outputs["model_resource_name"]
        )

    return pipeline


def compile_pipeline():
    pipeline = vertex_pipeline()  # Reference your pipeline definition
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pga_tour_pipeline.json"  # The compiled pipeline file
    )


if __name__ == "__main__":
    import argparse
    from google.cloud import aiplatform

    # Parse arguments
    parser = argparse.ArgumentParser(description="Compile and optionally submit Vertex AI pipeline.")
    parser.add_argument("--submit", action="store_true", help="Submit the pipeline to Vertex AI Pipeline after compilation.")
    parser.add_argument("--project", type=str, required=True, help="GCP Project ID.")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region for the pipeline.")
    parser.add_argument("--bucket", type=str, required=True, help="GCS bucket for pipeline artifacts.")
    parser.add_argument("--gcs_csv_location", type=str, required=True, help="Location of CSV files.")
    parser.add_argument("--service-account", type=str, help="Service account to run the pipeline.")

    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.bucket,
    )

    if args.submit:
        # Compile pipeline
        compile_pipeline()

        # Submit the pipeline to Vertex AI
        pipeline_job = aiplatform.PipelineJob(
            display_name="pga-tour-pipeline",
            template_path="pga_tour_pipeline.json",
            pipeline_root=f"gs://{args.bucket}/pipeline-root",
            parameter_values={
                "gcs_bucket": args.gcs_csv_location,
            },
        )
        # Pass the service account to the run method
        pipeline_job.run(
            service_account=args.service_account if args.service_account else None,
            sync=True  # Set `sync=False` to run asynchronously
        )
