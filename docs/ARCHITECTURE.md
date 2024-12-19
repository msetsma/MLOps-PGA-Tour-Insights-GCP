# Architecture

1. **Data Ingestion**
    - **Source**: New tournament data arrives as CSV files in a **Google Cloud Storage** bucket.
    - **Cloud Function**: A cloud function move the new csv into the expected table

2. **Model Retraining**
    - **Trigger**: New data in big query then kicks of vertex ai pipelines to retrain the model.
    - **Pipeline Execution**:
        - Loads the new data from BigQuery.
        - Prepares the data for training (splits into training/validation sets).
        - Trains a new model version using Vertex AI Training.
        - Evaluates the model & versions it. Saves to model regestry.
        - Deploys the new model to a **Vertex AI Endpoint** if performance is acceptable.

3. **Deployment and Inference**
    - The trained model is deployed as an API using **Vertex AI Endpoints**.
    - The API accepts player details and returns predictions for future tournaments.

4. **Monitoring**
    - **Model Monitoring**: Use **Vertex AI Model Monitoring** to track model performance, input feature drift, and prediction quality.
    - **API Monitoring**: Use **Cloud Logging** and **Cloud Monitoring** to observe API health, latency, and usage patterns.

5. **Versioning and Rollbacks**
    - Keep track of model versions in Vertex AI.
    - Enable rollback to a previous model if the newly deployed model performs poorly.

# Tools & Services
- Code - GitHub
- CI/CD - GitHub Actions (not yet implemented)
- Data Storage - BigQuery & Cloud Storage
- Feature Engineering - Vertex AI Pipelines
- Model Training - Vertex AI
- Model Deployment - Vertex AI Endpoints
- Containers - Docker saved to Artifact Registry
- Monitoring - Vertex AI Monitoring, Cloud Logging
- Automation - Vertex AI Pipelines & Cloud Functions

# My Workflow
1. Local development with Jupyterlab notebook.
2. Develop pipelines and automation baised off of Juypterlab testing.
3. Deploy service to GCP.
4. Set up CI CD using GitHub Actions