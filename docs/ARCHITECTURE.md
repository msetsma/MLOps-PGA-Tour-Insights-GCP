# Inital Architecture

1. **Data Ingestion**
    - **Source**: New tournament data arrives as CSV files in a **Google Cloud Storage** bucket.
    - **Dataflow Pipeline**: A **Dataflow** job automatically processes the incoming CSV files:
        - Creates copy of raw data in a seprate RAW BigQuery Table.
        - Validates the data format and schema.
        - Performs any feature engineering or preprocessing.
        - Writes the processed data to **BigQuery** for storage and analysis.

2. **Model Retraining**
    - **Trigger**: The **Cloud Storage** bucket triggers a Cloud Function whenever new data is uploaded.
    - **Pipeline Execution**:
        - The Cloud Function triggers a **Vertex AI Pipeline**.
        - The pipeline:
        - Loads the new data from BigQuery.
        - Prepares the data for training (splits into training/validation sets).
        - Trains a new model version using Vertex AI Training.
        - Evaluates the model (e.g., using RMSE or MAE for a regression task).
        - Deploys the new model to a **Vertex AI Endpoint** if performance is acceptable.

3. **Deployment and Inference**
    - The trained model is deployed as an API using **Vertex AI Endpoints** or **Cloud Run** (for custom deployment needs).
    - The API accepts player details and returns predictions for future tournaments.

4. **Monitoring**
    - **Model Monitoring**: Use **Vertex AI Model Monitoring** to track model performance, input feature drift, and prediction quality.
    - **API Monitoring**: Use **Cloud Logging** and **Cloud Monitoring** to observe API health, latency, and usage patterns.

5. **Versioning and Rollbacks**
    - Keep track of model versions in Vertex AI.
    - Enable rollback to a previous model if the newly deployed model performs poorly.

6. **Automation**
    - Automate retraining by setting up vertex AI pipelines to retain model when new data is added to the bigquery table.
    - Test new model and version it.