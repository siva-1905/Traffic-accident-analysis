import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("traffic_accidents.csv")

print("=" * 60)
print("TRAFFIC ACCIDENT ML PIPELINE")
print("=" * 60)

print("\nDataset shape:", df.shape)


# =========================================================
# 2. FEATURES
# =========================================================

features = [
    "City",
    "Weather",
    "Road_Condition",
    "Light_Condition",
    "Speed_Limit",
    "Traffic_Density",
    "Vehicle_Count",
    "Hour",
    "Day_of_Week"
]

target = "Accident_Severity"

X = df[features]
y = df[target]


# =========================================================
# 3. ENCODE TARGET
# =========================================================

target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y = y.map(target_mapping)


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 5. PREPROCESSING
# =========================================================

categorical_features = [
    "City",
    "Weather",
    "Road_Condition",
    "Light_Condition",
    "Traffic_Density",
    "Day_of_Week"
]

numerical_features = [
    "Speed_Limit",
    "Vehicle_Count",
    "Hour"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# =========================================================
# 6. RANDOM FOREST
# =========================================================

rf_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


print("\nTraining Random Forest...")

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)


# =========================================================
# 7. XGBOOST
# =========================================================

xgb_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


print("Training XGBoost...")

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_test
)


# =========================================================
# 8. EVALUATION FUNCTION
# =========================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Low",
                "Medium",
                "High"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(cm)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }


# =========================================================
# 9. EVALUATE BOTH MODELS
# =========================================================

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    xgb_predictions
)


# =========================================================
# 10. MODEL COMPARISON
# =========================================================

comparison = pd.DataFrame([
    rf_results,
    xgb_results
])

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False
    )
)


# =========================================================
# 11. SELECT BEST MODEL
# =========================================================

if (
    xgb_results["F1_Score"]
    >=
    rf_results["F1_Score"]
):

    best_model = xgb_model
    best_model_name = "XGBoost"

else:

    best_model = rf_model
    best_model_name = "Random Forest"


print(
    f"\nBest Model: {best_model_name}"
)


# =========================================================
# 12. CREATE MODELS DIRECTORY
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)


# =========================================================
# 13. SAVE MODELS
# =========================================================

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

joblib.dump(
    xgb_model,
    "models/xgboost.pkl"
)

joblib.dump(
    target_mapping,
    "models/target_mapping.pkl"
)

joblib.dump(
    features,
    "models/features.pkl"
)


# =========================================================
# 14. SAVE EVALUATION RESULTS
# =========================================================

comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)


# =========================================================
# 15. SAVE CONFUSION MATRICES
# =========================================================

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

xgb_cm = confusion_matrix(
    y_test,
    xgb_predictions
)

pd.DataFrame(
    rf_cm,
    index=["Low", "Medium", "High"],
    columns=["Low", "Medium", "High"]
).to_csv(
    "models/random_forest_confusion_matrix.csv"
)

pd.DataFrame(
    xgb_cm,
    index=["Low", "Medium", "High"],
    columns=["Low", "Medium", "High"]
).to_csv(
    "models/xgboost_confusion_matrix.csv"
)


# =========================================================
# 16. SAVE TEST DATA
# =========================================================

test_results = X_test.copy()

test_results["Actual"] = y_test.values
test_results["RF_Prediction"] = rf_predictions
test_results["XGB_Prediction"] = xgb_predictions

test_results.to_csv(
    "models/test_predictions.csv",
    index=False
)


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nSaved files:")

print("✓ best_model.pkl")
print("✓ random_forest.pkl")
print("✓ xgboost.pkl")
print("✓ target_mapping.pkl")
print("✓ features.pkl")
print("✓ model_comparison.csv")
print("✓ confusion matrices")
print("✓ test_predictions.csv")