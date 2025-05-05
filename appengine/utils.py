from google.cloud import storage
from io import StringIO, BytesIO
import joblib
import pandas as pd


GCLOUD_PROJECT = "cs163-project-452620"
GCLOUD_BUCKET = "cs163-project-452620.appspot.com"

# --------- Load JSON files ---------
def get_json_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project=GCLOUD_PROJECT)
    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    return pd.read_json(StringIO(data), lines=True)

def load_pickle_from_gcs(bucket_name, source_blob_name):
    """Downloads a pickle file from the GCS bucket and loads it."""
    storage_client = storage.Client(project=GCLOUD_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    pickle_bytes = blob.download_as_bytes()
    obj = joblib.load(BytesIO(pickle_bytes))
    
    return obj