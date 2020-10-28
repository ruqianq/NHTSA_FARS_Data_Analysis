from sklearn.ensemble import RandomForestClassifier

# Training data
X = dm_traindf.loc[:, dm_input]

# Labels
y = dm_traindf[dm_dec_target]

# Fit RandomForest model w/ training data
dm_model = RandomForestClassifier(n_estimators=10)
dm_model.fit(X, y)
print(dm_model)
