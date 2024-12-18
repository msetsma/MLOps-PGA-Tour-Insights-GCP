from google.cloud import bigquery

# Initialize BigQuery clienty
bigquery_client = bigquery.Client()

# BigQuery table details
DATASET_ID = "mitchell-setsma-gcp-project.pga_tour_data"
TABLE_ID = "raw_data"

def gcs_to_bigquery(event, context):
    """
    Triggered by a change to a Cloud Storage bucket.
    Loads the file into a BigQuery table.
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

        print(f"File {gcs_file_name} loaded to BigQuery table {table_ref} successfully.")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise
