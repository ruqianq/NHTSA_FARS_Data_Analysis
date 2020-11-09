from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import bq

# query = """
#     SELECT
#     ap.*, body_type,
#     vehicle_trailing, gross_vehicle_weight_rating, hazardous_material_involvement, travel_speed,
#     previous_recorded_crashes, previous_recorded_suspensions_and_revocations, previous_dwi_convictions,
#     previous_speeding_convictions, speeding_related, related_factors_driver_level_1, related_factors_driver_level_2,
#     related_factors_driver_level_3, related_factors_driver_level_4, roadway_alignment, roadway_grade,
#     roadway_surface_condition, crash_type, fatalities_in_vehicle
#     FROM (
#     SELECT a.state_number, state_name, a.consecutive_number,
#     number_of_forms_submitted_for_persons_not_in_motor_vehicles, a.county, city, a.day_of_crash, a.month_of_crash,
#     day_of_week, a.hour_of_crash, national_highway_system, a.land_use, a.functional_system, ownership
#     route_signing, a.first_harmful_event, a.manner_of_collision, relation_to_junction_within_interchange_area,
#     relation_to_junction_specific_location, type_of_intersection, work_zone, relation_to_trafficway, light_condition,
#     atmospheric_conditions, related_factors_crash_level_1, related_factors_crash_level_2, related_factors_crash_level_3,
#     number_of_fatalities, p.vehicle_number, p.person_number, p.rollover, p.age, p.sex, p.person_type, p.injury_severity,
#     p.police_reported_alcohol_involvement, p.police_reported_drug_involvement, related_factors_person_level1,
#     related_factors_person_level2, related_factors_person_level3, race
#     FROM
#     `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` a
#     LEFT JOIN
#     `bigquery-public-data.nhtsa_traffic_fatalities.person_2015` p
#     ON a.consecutive_number = p.consecutive_number
#     WHERE p.person_type = 1
#     ) ap
#     JOIN
#     `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v
#     ON ap.vehicle_number = v.vehicle_number
#     LIMIT 100
#     """


# Data cleanse

df = pd.read_csv('sample_data.csv')
df_cln = df[df.columns[df.isnull().mean() < 0.5]]
df_cln["ped_death"] = df_cln.apply(lambda row: 1 if (row.number_of_fatalities - row.fatalities_in_vehicle > 0) else 0,
                                   axis=1)
print(df_cln.select_dtypes('object'))
# X = df_cln.drop(["ped_death"], axis=1)
# y = df_cln["ped_death"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# model = RandomForestClassifier().fit(X, y)
# print(model)
