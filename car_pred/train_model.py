# train_model.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("car_data.csv")

# 2. Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# 3. Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Define features (X) and target (y)
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price'].astype(float)

# 5. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train multiple models & find best one
models = {
    "Lasso Regression": Lasso(alpha=0.1),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

best_model = None
best_score = -1
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"{name} R² Score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print(f"\n✅ Best Model: {best_name} with R² Score: {best_score:.4f}")

# 7. Save best model & label encoders
with open("car_price_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "label_encoders": label_encoders,
        "features": list(X.columns)
    }, f)

print("💾 Model saved as car_price_model.pkl")
