import os
import pandas as pd
from google.cloud import bigquery

# Construct a BigQuery client object.
# Follow the instruction:
# https://cloud.google.com/bigquery/docs/reference/libraries?authuser=1#client-libraries-usage-python


def to_df(query, credentials_path) -> pd.DataFrame:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = bigquery.Client()
    return client.query(query).to_dataframe()
