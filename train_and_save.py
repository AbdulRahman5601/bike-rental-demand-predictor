"""Backup: rebuilds the champion XGBoost model exactly as in your notebook
(same cleaning, same LabelEncoder logic, same params) and saves the .pkl.
Only needed if you can't find champion_bike_rental_model.pkl from Jupyter."""
import pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

df = pd.read_csv("Dataset.csv")
cat_cols = ['season', 'yr', 'mnth', 'holiday', 'workingday', 'weathersit']
num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
for c in cat_cols:
    df[c] = df[c].replace('?', pd.NA)
    df[c] = df[c].fillna(df[c].mode()[0])
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

df_model = df.drop(columns=['instant', 'dteday', 'registered', 'casual', 'atemp'])
le = LabelEncoder()
for c in cat_cols:
    df_model[c] = le.fit_transform(df_model[c].astype(str))

X, y = df_model.drop('cnt', axis=1), df_model['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(random_state=42, objective='reg:squarederror',
                     n_estimators=200, max_depth=7, learning_rate=0.2)
model.fit(X_train, y_train)
print("Feature order:", list(X.columns))
print("Test R2:", round(r2_score(y_test, model.predict(X_test)), 4))
joblib.dump(model, "champion_bike_rental_model.pkl")
# Version-proof format: loads on any xgboost version (use this one for deployment)
model.save_model("champion_bike_rental_model.json")
print("Saved champion_bike_rental_model.pkl and champion_bike_rental_model.json")
