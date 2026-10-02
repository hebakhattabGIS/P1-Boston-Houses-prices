"""Train the Boston housing decision tree and save it as model.pkl.

Run once (and again whenever you want to retrain):
    python train_model.py
Needs housing.csv in the same folder.
"""
import joblib
import pandas as pd
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor


def performance_metric(y_true, y_predict):
    return r2_score(y_true, y_predict)


def fit_model(X, y):
    """Grid search over max_depth for a decision tree regressor."""
    # NOTE: the first argument of ShuffleSplit is n_splits in current scikit-learn
    # (in the old API it was the number of samples), so it is named explicitly here.
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    regressor = DecisionTreeRegressor(random_state=0)
    params = {"max_depth": list(range(1, 11))}
    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_


if __name__ == "__main__":
    data = pd.read_csv("housing.csv")
    prices = data["MEDV"]
    features = data.drop("MEDV", axis=1)  # columns: RM, LSTAT, PTRATIO

    X_train, X_test, y_train, y_test = train_test_split(
        features, prices, test_size=0.2, random_state=1
    )

    reg = fit_model(X_train, y_train)
    print("Feature order:", list(features.columns))
    print("Optimal max_depth:", reg.get_params()["max_depth"])
    print("R2 on test set: {:.3f}".format(performance_metric(y_test, reg.predict(X_test))))

    joblib.dump(reg, "model.pkl")
    print("Saved model.pkl")
