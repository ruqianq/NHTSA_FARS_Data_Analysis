from google.cloud import bigquery
import pandas as pd

# Construct a BigQuery client object.
# Follow the instruction: https://cloud.google.com/bigquery/docs/reference/libraries?authuser=1#client-libraries-usage-python


def to_df() -> pd.DataFrame:
    client = bigquery.Client()
    query = """
    SELECT *
    FROM (
    SELECT *
    FROM
    `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` a
    LEFT JOIN
    `bigquery-public-data.nhtsa_traffic_fatalities.person_2015` p
    ON a.consecutive_number = p.consecutive_number 
    WHERE p.person_number = 1
    ) ap
    JOIN
    `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v
    ON ap.vehicle_number = v.vehicle_number
    WHERE number_of_forms_submitted_for_persons_not_in_motor_vehicles > 0
    LIMIT 20
    """
    return client.query(query).to_dataframe()
