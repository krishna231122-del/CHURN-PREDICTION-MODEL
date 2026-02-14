import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from PREPROCESS import preprocess_data


# 📁 Create model folder
os.makedirs("model", exist_ok=True)

# 📊 Load data
df = pd.read_csv("/Users/krishnasoni/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")

# ⚙️ Preprocess
df = preprocess_data(df, training=True)

# 🎯 Split features + target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🔤 One hot encoding
X = pd.get_dummies(X)

# ✂️ Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🤖 Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 📈 Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 💾 Save model + columns
joblib.dump(
    {
        "model": model,
        "columns": X.columns.tolist()
    },
    "model/churn_model.pkl"
)

print("✅ Model saved successfully")
