import numpy as np
import matplotlib.pyplot as plt
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import shap

X, Xsc, X_pocket, Xsc_pocket, Y, Ysc, Y_pocket, Ysc_pocket = preprocessing.main()

total_X = np.concatenate((Xsc, Xsc_pocket), axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(total_X, Ysc, test_size = 0.2, random_state = 0)

rf = RandomForestRegressor(criterion = 'squared_error', random_state = 0)
hparams = {'max_depth': [x for x in range(23, 24, 1)],
           'n_estimators': [x for x in range(250, 251, 10)],
           'min_samples_leaf': [x for x in range(2, 3, 1)]}

gscv = GridSearchCV(rf, param_grid = hparams, cv = 5, scoring = 'r2')
gscv.fit(X_train, np.ravel(Y_train))

best_param = gscv.best_params_
print("Best Parameters")
print(best_param)

optimized_rf = RandomForestRegressor(**best_param, random_state = 0)

optimized_rf.fit(X_train, np.ravel(Y_train))
y_pred = optimized_rf.predict(X_test)

print(r2_score(np.ravel(Y_test), np.ravel(y_pred)))

plt.scatter(y_pred, Y_test, marker = '.', alpha = 0.5, )
x = np.linspace(-5, 5)
plt.plot(x, x, color = 'r', alpha = 0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Predicted Binding Affinities vs. Actual Binding Affinities")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()

# random forest regressor gets 0.3859 with 23 max_depth, min_samples_leaf 2, 250 n_estimators


shap.initjs()

explainer = shap.Explainer(optimized_rf)
shap_values = explainer(X_train)
shap_values = shap.Explanation(shap_values)
shap.plots.beeswarm(shap_values)
plt.show()
shap.plots.bar(shap_values)
plt.show()
