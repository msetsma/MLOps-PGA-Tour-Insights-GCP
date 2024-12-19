from google.cloud import bigquery
from google.cloud import aiplatform

# Initialize BigQuery client
bigquery_client = bigquery.Client()

# BigQuery table details
DATASET_ID = "mitchell-setsma-gcp-project.pga_tour_data"
TABLE_ID = "raw_data"

# Vertex AI details
PROJECT_ID = "mitchell-setsma-gcp-project"
REGION = "us-central1"
PIPELINE_NAME = "pga-tour-template"  
PIPELINE_TEMPLATE_PATH = "projects/mitchell-setsma-gcp-project/locations/us-central1/repositories/pipeline-templates/packages/pga-tour-template/versions/sha256:30d5243a21881bed0d2b014b14083fc6e92f97f235ab5e718d488f79b3beb1be"
BUCKET = "pga-tour-pipeline-artifacts"
CSV_LOCATION = "mlops-data-ingestion"

def gcs_to_bigquery(event, context):
    """
    Triggered by a change to a Cloud Storage bucket.
    Loads the file into a BigQuery table and triggers a Vertex AI pipeline.
    """
    try:
        gcs_bucket = event["bucket"]
        gcs_file_name = event["name"]
        gcs_uri = f"gs://{gcs_bucket}/{gcs_file_name}"

        print(f"Processing file: {gcs_file_name} from bucket: {gcs_bucket}")

        # Configure BigQuery load job
        table_ref = f"{DATASET_ID}.{TABLE_ID}"
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            autodetect=True,  # Automatically detect schema
            write_disposition="WRITE_APPEND",  # Append to existing table
        )

        # Load data from GCS to BigQuery
        load_job = bigquery_client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
        load_job.result()  # Wait for the job to complete

        # Initialize Vertex AI client
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

        # Submit the pipeline to Vertex AI
        pipeline = aiplatform.PipelineJob(
            display_name="pga-tour-pipeline",
            template_path=PIPELINE_TEMPLATE_PATH,
            pipeline_root=f"gs://{BUCKET}/pipeline-root",
            parameter_values={
                "gcs_bucket": CSV_LOCATION,
            },
        )
        pipeline.run()  # Start the pipeline execution

        print(f"Vertex AI pipeline {PIPELINE_NAME} triggered successfully.")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise
