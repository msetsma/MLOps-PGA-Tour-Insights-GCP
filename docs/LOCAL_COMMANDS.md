# Command to run locally to set up my notebook

### Create a Virtual Environment
Run this command in your terminal (Command Prompt or PowerShell):

```bash
python -m venv .venv
```

### Activate the Virtual Environment
- **In PowerShell**:
   ```bash
   .\.venv\Scripts\Activate
   ```

- **In Command Prompt**:
   ```cmd
   .\.venv\Scripts\activate.bat
   ```

### Upgrade `pip` and Install Dependencies

```bash
pip install -r requirements.txt
```

### Register the Virtual Environment as a Jupyter Kernel

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (venv)"
```

- `--name=venv`: Internal name for the kernel.
- `--display-name "Python (venv)"`: The name you'll see in Jupyter Notebook.


### Launch Jupyter Notebook:

```bash
jupyter notebook
```
- This opens a new browser window with the Jupyter Notebook interface.
- When creating a new notebook, you should see **"Python (venv)"** as an available kernel.
# Use the gcloud CLI to upload the model to GCS
gcloud storage cp -r model_dir gs://ga-tour-pipeline-artifacts/model
python vertex_ai_pipeline.py --submit --project mitchell-setsma-gcp-project --bucket pga-tour-pipeline-artifacts --region us-central1 --gcs_csv_location mlops-data-ingestion --service-account=925092514777-compute@developer.gserviceaccount.com


gcloud ai models upload --region=us-central1 --display-name=pga-tour-model --artifact-uri=gs://pga-tour-pipeline-artifacts/model --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest  