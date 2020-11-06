import pandas as pd
from google.cloud import bigquery

# Construct a BigQuery client object.
# Follow the instruction:
# https://cloud.google.com/bigquery/docs/reference/libraries?authuser=1#client-libraries-usage-python


def to_df() -> pd.DataFrame:
    client = bigquery.Client()
    query = """
    SELECT
    ap.*, vehicle_make, vehicle_model, body_type, vehicle_model_year, vehicle_identification_number_vin, 
    vehicle_trailing, gross_vehicle_weight_rating, hazardous_material_involvement, travel_speed, 
    previous_recorded_crashes, previous_recorded_suspensions_and_revocations, previous_dwi_convictions, 
    previous_speeding_convictions, speeding_related, related_factors_driver_level_1, related_factors_driver_level_2, 
    related_factors_driver_level_3, related_factors_driver_level_4, roadway_alignment, roadway_grade, 
    roadway_surface_type, roadway_surface_condition, crash_type, fatalities_in_vehicle
    FROM (
    SELECT a.state_number, state_name, a.consecutive_number, 
    number_of_forms_submitted_for_persons_not_in_motor_vehicles, a.county, city, a.day_of_crash, a.month_of_crash, 
    year_of_crash, day_of_week, a.hour_of_crash, national_highway_system, a.land_use, a.functional_system, ownership 
    route_signing, a.first_harmful_event, a.manner_of_collision, relation_to_junction_within_interchange_area, 
    relation_to_junction_specific_location, type_of_intersection, work_zone, relation_to_trafficway, light_condition, 
    atmospheric_conditions, related_factors_crash_level_1, related_factors_crash_level_2, related_factors_crash_level_3, 
    number_of_fatalities, p.vehicle_number, p.person_number, p.rollover, p.age, p.sex, p.person_type, p.injury_severity, 
    p.police_reported_alcohol_involvement, p.police_reported_drug_involvement, related_factors_person_level1, 
    related_factors_person_level2, related_factors_person_level3, race
    FROM
    `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` a
    LEFT JOIN
    `bigquery-public-data.nhtsa_traffic_fatalities.person_2015` p
    ON a.consecutive_number = p.consecutive_number 
    WHERE p.person_type = 1
    ) ap
    JOIN
    `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v
    ON ap.vehicle_number = v.vehicle_number
    WHERE number_of_forms_submitted_for_persons_not_in_motor_vehicles	> 0
    LIMIT 20
    """
    return client.query(query).to_dataframe()
