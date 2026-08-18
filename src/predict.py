import joblib
import pandas as pd

MODEL_PATH = "models/fraud_detection_pipeline.joblib"
model = joblib.load(MODEL_PATH)

def predict_transaction(transaction: dict):
    row = pd.DataFrame([transaction])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": prediction,
        "label": "Fraudulent" if prediction else "Legitimate",
        "fraud_probability": probability
    }

def predict_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    expected = list(model.feature_names_in_)
    result = df.copy()
    result["Fraud_Prediction"] = model.predict(df[expected])
    result["Fraud_Probability"] = model.predict_proba(df[expected])[:, 1]
    result.to_csv(output_csv, index=False)
    return result
