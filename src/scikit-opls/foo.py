from sklearn.cross_decomposition import PLSRegression

X = [[0.0, 1.0, 2.0], [2.0, 4.0, 5.0]]
Y = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]

pls2 = PLSRegression(n_components=1)

# Fit the model
pls2.fit(X, Y)

# The predicted values of Y
print(pls2.predict(X))
