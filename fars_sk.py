from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import bq

df = bq.to_df()

# Data cleanse

df_cln = df[df.columns[df.isnull().mean() < 0.5]]
df_cln["ped_death"] = df_cln.apply(lambda row: 1 if (row.number_of_fatalities - row.fatalities_in_vehicle > 0) else 0,
                                   axis=1)

X = df_cln.drop(["ped_death"], axis=1)
y = df_cln["ped_death"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier().fit(X, y)
print(model)
