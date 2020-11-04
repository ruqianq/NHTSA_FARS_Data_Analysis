from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data_sample.csv')

# Data cleanse

df_cln = df[df.columns[df.isnull().mean() < 0.5]]

df_cln["ped_death"] = df.apply(lambda row: 1 if (row.fatals - row.deaths > 0) else 0, axis=1)
print(df_cln["ped_death"])

X = df_cln.drop(["ped_death"], axis=1)
y = df_cln["ped_death"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Viya Open Source Code Node
# # Training data
# X = dm_traindf.loc[:, dm_input]
#
# # Labels
# y = dm_traindf[dm_dec_target]
#
# Fit RandomForest model w/ training data
dm_model = RandomForestClassifier(n_estimators=10)
dm_model.fit(X_train, y_train)

print(pd.DataFrame(dm_model.predict(X_test), columns=['P_Injury_0']))
