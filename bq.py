import pandas as pd
from google.cloud import bigquery

# Construct a BigQuery client object.
# Follow the instruction:
# https://cloud.google.com/bigquery/docs/reference/libraries?authuser=1#client-libraries-usage-python


def to_df(query) -> pd.DataFrame:
    client = bigquery.Client()
    return client.query(query).to_dataframe()
