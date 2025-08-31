import argparse
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------
# Argument Parser
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Make predictions using Loan Default Model")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV or JSON file")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with model + preprocessor")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file for predictions")
    return parser.parse_args()

# -------------------------------
# Prediction Function
# -------------------------------
def main():
    args = parse_args()

    # Load preprocessing pipeline and model
    preprocess_path = os.path.join(args.model_dir, "preprocess.joblib")
    model_path = os.path.join(args.model_dir, "loan_default_model.h5")

    if not os.path.exists(preprocess_path) or not os.path.exists(model_path):
        raise FileNotFoundError("❌ Preprocessor or model not found! Run main.py first to train and save them.")

    preprocessor = joblib.load(preprocess_path)
    model = load_model(model_path)

    # Load input data
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    elif args.input.endswith(".json"):
        df = pd.read_json(args.input)
    else:
        raise ValueError("❌ Input must be a CSV or JSON file")

    print(f"✅ Loaded input data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Preprocess input (ignore missing target if present)
    if "loan_status" in df.columns:
        df = df.drop(columns=["loan_status"])

    X_processed = preprocessor.transform(df)

    # Predict probabilities
    predictions = model.predict(X_processed)
    df["default_probability"] = predictions
    df["prediction"] = (predictions > 0.5).astype(int)

    # Save results if CSV
    if args.input.endswith(".csv"):
        df.to_csv(args.output, index=False)
        print(f"✅ Predictions saved to {args.output}")
    else:
        print("✅ Predictions:")
        print(df[["default_probability", "prediction"]])

if __name__ == "__main__":
    main()
