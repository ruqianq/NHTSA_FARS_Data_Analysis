import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# query = """
#     SELECT
#     ap.*, body_type,
#     vehicle_trailing, gross_vehicle_weight_rating, travel_speed,
#     previous_recorded_crashes, previous_recorded_suspensions_and_revocations, previous_dwi_convictions,
#     previous_speeding_convictions, related_factors_driver_level_1, related_factors_driver_level_2,
#     related_factors_driver_level_3, related_factors_driver_level_4, roadway_alignment, roadway_grade,
#     roadway_surface_type, roadway_surface_condition, crash_type, fatalities_in_vehicle
#     FROM (
#     SELECT a.state_number, a.consecutive_number,number_of_forms_submitted_for_persons_not_in_motor_vehicles,
#     a.county, city, a.day_of_crash, a.month_of_crash, day_of_week, a.hour_of_crash, national_highway_system,
#     a.land_use, a.functional_system, route_signing, a.first_harmful_event, a.manner_of_collision,
#     relation_to_junction_within_interchange_area, relation_to_junction_specific_location, type_of_intersection,
#     relation_to_trafficway, light_condition, atmospheric_conditions, related_factors_crash_level_1,
#     related_factors_crash_level_2, related_factors_crash_level_3, number_of_fatalities, p.vehicle_number,
#     p.person_number, p.rollover, p.age, p.sex, p.person_type, p.injury_severity,
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
#     WHERE speeding_related LIKE '%Yes%'
#     """
import bq

query = """
SELECT * FROM `nhtsa-daisy.fars_2015.fars_apv_2015` 
"""

# Data cleanse
df = bq.to_df(query)
df_cln = df[df.columns[df.isnull().mean() < 0.5]]

# Create Target
df_cln["ped_death"] = df_cln.apply(lambda row: 1 if (row.number_of_fatalities - row.fatalities_in_vehicle > 0) else 0,
                                   axis=1)

# Encoding
df_cln = df_cln.drop(['consecutive_number', 'number_of_fatalities', 'fatalities_in_vehicle'], axis=1)

df_cln["relation_to_junction_within_interchange_area"] = df_cln.apply(
    lambda row: 1 if (row.relation_to_junction_within_interchange_area == 'Yes') else 0, axis=1)

df_cln['vehicle_trailing'] = df_cln['vehicle_trailing'].str[0:3]
df_cln['vehicle_trailing'] = df_cln.apply(
    lambda row: 0 if (row.vehicle_trailing == 'No ') else 1, axis=1)

labels = df_cln['gross_vehicle_weight_rating'].astype('category').cat.categories.tolist()
replace_map_comp = {'gross_vehicle_weight_rating': {k: v for k, v in zip(labels, list(range(0, len(labels) + 1)))}}
df_cln.replace(replace_map_comp, inplace=True)

labels2 = df_cln['roadway_alignment'].astype('category').cat.categories.tolist()
replace_map_comp2 = {'roadway_alignment': {k: v for k, v in zip(labels2, list(range(1, len(labels2) + 1)))}}
df_cln.replace(replace_map_comp2, inplace=True)

labels3 = df_cln['roadway_grade'].astype('category').cat.categories.tolist()
replace_map_comp3 = {'roadway_grade': {k: v for k, v in zip(labels3, list(range(1, len(labels3) + 1)))}}
df_cln.replace(replace_map_comp3, inplace=True)

labels4 = df_cln['roadway_surface_type'].astype('category').cat.categories.tolist()
replace_map_comp4 = {'roadway_surface_type': {k: v for k, v in zip(labels4, list(range(1, len(labels4) + 1)))}}
df_cln.replace(replace_map_comp4, inplace=True)

labels5 = df_cln['type_of_intersection'].astype('category').cat.categories.tolist()
replace_map_comp5 = {'type_of_intersection': {k: v for k, v in zip(labels5, list(range(1, len(labels5) + 1)))}}
df_cln.replace(replace_map_comp5, inplace=True)

df_cln['rollover'] = df_cln['rollover'].str[0:2]
df_cln['rollover'] = df_cln.apply(
    lambda row: 0 if (row.rollover == 'No') else 1, axis=1)

df_cln['police_reported_alcohol_involvement'] = df_cln['police_reported_alcohol_involvement'].str[0:3]
df_cln['police_reported_alcohol_involvement'] = df_cln.apply(
    lambda row: 1 if (row.police_reported_alcohol_involvement == 'Yes') else 0, axis=1)

df_cln['police_reported_drug_involvement'] = df_cln['police_reported_drug_involvement'].str[0:3]
df_cln['police_reported_drug_involvement'] = df_cln.apply(
    lambda row: 1 if (row.police_reported_drug_involvement == 'Yes') else 0, axis=1)

df_cln['sex'] = df_cln.apply(
    lambda row: 0 if (row.sex == 'Female') else 1, axis=1)

# Split training set and testing set
X = df_cln.drop(["ped_death"], axis=1)
y = df_cln["ped_death"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Training model
classifiers = [LogisticRegression(random_state=1234),
               GaussianNB(),
               KNeighborsClassifier(),
               DecisionTreeClassifier(random_state=1234),
               RandomForestClassifier(random_state=1234)]

result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::, 1]

    fpr, tpr, _ = roc_curve(y_test, yproba)
    auc = roc_auc_score(y_test, yproba)

    result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)

result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8, 6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')

plt.show()
