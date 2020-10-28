from sklearn import ensemble

# Training data
X = dm_traindf.loc[:, dm_input]

# Labels
y = dm_traindf[dm_dec_target]

# Fit RandomForest model w/ training data
params = {'n_estimators': 100}
dm_model = ensemble.RandomForestClassifier(**params)
dm_model.fit(X, y)
print(dm_model)