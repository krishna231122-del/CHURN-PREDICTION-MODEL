import pandas as pd

def preprocess_data(df, training=True):

    if training and "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df