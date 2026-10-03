"""Train the Boston housing decision tree and save it as model.pkl.

Run once (and again whenever you change the settings below):
    python train_model.py
"""
import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = "housing.csv"   # columns: RM, LSTAT, PTRATIO, MEDV
MODEL_PATH = "model.pkl"
TARGET = "MEDV"
MAX_DEPTH = 4               # tree depth used for training
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ----------------------------------------------------------------------

data = pd.read_csv(DATA_PATH)
y = data[TARGET]
X = data.drop(TARGET, axis=1)  # feature order here must match FEATURES in app.py

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print("Features:", list(X.columns))
print("max_depth:", MAX_DEPTH)
print(f"Training R2 score: {r2_score(y_train, model.predict(X_train)):.3f}")
print(f"Testing R2 score:  {r2_score(y_test, model.predict(X_test)):.3f}")

joblib.dump(model, MODEL_PATH)
print(f"Saved {MODEL_PATH}")