import pandas as pd
import joblib
import os

# Load trained models
rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

def extract_importance(model, model_name):
    # Get preprocessing pipeline
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Get importance values
    importance = classifier.feature_importances_

    # Create dataframe
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    # Make names cleaner
    df["Feature"] = (
        df["Feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("numerical__", "", regex=False)
    )

    # Sort
    df = df.sort_values("Importance", ascending=False)

    # Save
    filename = f"models/{model_name.lower().replace(' ', '_')}_feature_importance.csv"
    df.to_csv(filename, index=False)

    print(f"\n{model_name} Feature Importance")
    print("-" * 50)
    print(df.head(15).to_string(index=False))
    print(f"\nSaved to: {filename}")

    return df


# Extract importance
rf_importance = extract_importance(
    rf_model,
    "Random Forest"
)

xgb_importance = extract_importance(
    xgb_model,
    "XGBoost"
)