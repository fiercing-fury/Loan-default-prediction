import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# Argument Parser
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Loan Default Prediction Model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model")
    return parser.parse_args()

# -------------------------------
# Build Model
# -------------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# -------------------------------
# Main Function
# -------------------------------
def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.data)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Define target & features
    target = "loan_status"   # assuming column name
    y = df[target].apply(lambda x: 1 if x == "Charged Off" else 0)
    X = df.drop(columns=[target])

    # Identify categorical & numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # Preprocessing pipeline
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # Preprocess features
    X_processed = preprocessor.fit_transform(X)
    input_dim = X_processed.shape[1]

    # Save preprocessing pipeline
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(args.output_dir, "preprocess.joblib"))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=args.test_size, random_state=42
    )

    # Build & Train Model
    model = build_model(input_dim)
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("🔹 ROC-AUC:", roc_auc_score(y_test, y_pred))

    # Save model
    model.save(os.path.join(args.output_dir, "loan_default_model.h5"))
    print(f"✅ Model saved in {args.output_dir}")

if __name__ == "__main__":
    main()
