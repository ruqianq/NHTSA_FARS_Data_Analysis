from sklearn.linear_model import SGDClassifier

# Training data
X = dm_traindf.loc[:, dm_input]

# Labels
y = dm_traindf[dm_dec_target]

dm_model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

dm_model.fit(X, y)

print(dm_model)
