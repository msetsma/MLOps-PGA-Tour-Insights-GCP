# base image for the vertex ai pipelines
FROM python:3.12-slim

# Install required Python packages
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    tensorflow \
    kfp \
    google-cloud-storage \
    google-cloud-aiplatform \
    google-cloud-bigquery \
    pyarrow \
    google-auth \
    google-auth-oauthlib \
    google-api-python-client

# Set working directory
WORKDIR /app