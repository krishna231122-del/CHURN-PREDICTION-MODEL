import pandas as pd
import joblib

from PREPROCESS import preprocess_data

# 📦 Load saved model
data = joblib.load("model/churn_model.pkl")

model = data["model"]
train_columns = data["columns"]

# 📊 Load new data
df = pd.read_csv("/Users/krishnasoni/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")

# ⚙️ Preprocess (NO churn column here)
df = preprocess_data(df, training=False)

# 🔤 One hot encoding
df = pd.get_dummies(df)

# 🧠 Match training columns
df = df.reindex(columns=train_columns, fill_value=0)

# 🤖 Predict
predictions = model.predict(df)

print("Predictions:", predictions)
