import pandas as pd
import os
import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv('housing.csv')

X = data.drop("MEDV", axis=1) 
y = data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

optimal_model = DecisionTreeRegressor(max_depth=4,random_state=42)
optimal_model.fit(X_train, y_train)

train_predicts = optimal_model.predict(X_train)
test_predicts = optimal_model.predict(X_test)

train_score = r2_score(y_train,train_predicts)
test_score = r2_score(y_test,test_predicts)

print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

joblib.dump(optimal_model,'model.pkl')

print("Model trained and saved as model.pkl")