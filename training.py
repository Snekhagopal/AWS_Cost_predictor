import pandas as pd
import numpy as np


# ==========================================
# 1. DATASET PATH
# ==========================================

ec2 = "data/AmazonEC2.csv"


# ==========================================
# 2. READ LARGE CSV IN CHUNKS + FILTER
# ==========================================

filtered_chunks = []

for chunk in pd.read_csv(
    ec2,
    chunksize=100_000,
    low_memory=False
):

    filtered = chunk[
        (chunk["Unit"] == "Hrs") &
        (chunk["Product Family"] == "Compute Instance") &
        (chunk["PricePerUnit"] > 0)
    ].copy()

    filtered_chunks.append(filtered)


model_df = pd.concat(
    filtered_chunks,
    ignore_index=True
)

print("Filtered rows:", len(model_df))
print("Columns:", model_df.shape[1])


# ==========================================
# 3. TAKE SAMPLE FOR TRAINING
# ==========================================

model_df = model_df.sample(
    n=50_000,
    random_state=42
).copy()

print("Training sample rows:", len(model_df))


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================


# --------------------------
# Memory
# Example:
# "64 GiB" -> 64
# --------------------------

model_df["Memory_GiB"] = (
    model_df["Memory"]
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)


# --------------------------
# Clock Speed
# Example:
# "3.5 GHz" -> 3.5
# "Up to 3.7 GHz" -> 3.7
# --------------------------

model_df["ClockSpeed_GHz"] = (
    model_df["Clock Speed"]
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)


# --------------------------
# Storage
# Example:
# "2 x 900 NVMe SSD"
# --------------------------

storage_parts = model_df["Storage"].str.extract(
    r"(\d+)\s*x\s*(\d+(?:\.\d+)?)"
)


model_df["Storage_Count"] = pd.to_numeric(
    storage_parts[0],
    errors="coerce"
).fillna(0)


model_df["Storage_Size_GB"] = pd.to_numeric(
    storage_parts[1],
    errors="coerce"
).fillna(0)


model_df["Total_Storage_GB"] = (
    model_df["Storage_Count"] *
    model_df["Storage_Size_GB"]
)


# ==========================================
# 5. STORAGE TYPE
# ==========================================

model_df["Storage_Type"] = "Other"


model_df.loc[
    model_df["Storage"].str.contains(
        "NVMe SSD",
        case=False,
        na=False
    ),
    "Storage_Type"
] = "NVMe SSD"


model_df.loc[
    model_df["Storage"].str.contains(
        "SSD",
        case=False,
        na=False
    )
    &
    ~model_df["Storage"].str.contains(
        "NVMe",
        case=False,
        na=False
    ),
    "Storage_Type"
] = "SSD"


model_df.loc[
    model_df["Storage"].str.contains(
        "HDD",
        case=False,
        na=False
    ),
    "Storage_Type"
] = "HDD"


model_df.loc[
    model_df["Storage"].str.contains(
        "EBS only",
        case=False,
        na=False
    ),
    "Storage_Type"
] = "EBS only"


# ==========================================
# 6. GPU
# ==========================================

model_df["GPU"] = (
    pd.to_numeric(
        model_df["GPU"],
        errors="coerce"
    )
    .fillna(0)
)


model_df["GPU_Memory_GB"] = (
    model_df["GPU Memory"]
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
    .fillna(0)
)


# ==========================================
# 7. EBS THROUGHPUT
# ==========================================

def convert_ebs_to_mbps(value):

    if pd.isna(value):
        return np.nan

    value = str(value)

    match = pd.Series([value]).str.extract(
        r"(\d+(?:\.\d+)?)"
    )[0].iloc[0]

    if pd.isna(match):
        return np.nan

    number = float(match)

    if "Gbps" in value:
        number = number * 1000

    return number


model_df["EBS_Throughput_Mbps"] = (
    model_df["Dedicated EBS Throughput"]
    .apply(convert_ebs_to_mbps)
)


# ==========================================
# 8. FEATURES + TARGET
# ==========================================

features = [

    # Hardware
    "vCPU",
    "Memory_GiB",
    "ClockSpeed_GHz",
    "GPU",
    "GPU_Memory_GB",
    "Normalization Size Factor",

    # Storage
    "Storage_Count",
    "Total_Storage_GB",
    "Storage_Type",

    # EBS
    "EBS_Throughput_Mbps",

    # Instance
    "Instance Type",
    "Instance Family",
    "Processor Architecture",
    "Current Generation",

    # Configuration
    "Region Code",
    "Operating System",
    "Tenancy",
    "License Model",
    "CapacityStatus",
    "Pre Installed S/W",

    # Pricing
    "TermType",
    "PurchaseOption",
    "LeaseContractLength",
    "OfferingClass"
]


target = "PricePerUnit"


X = model_df[features].copy()

y = model_df[target].copy()


print("\nX shape:", X.shape)

print("y shape:", y.shape)


print("\nMissing values:")

print(
    X.isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
)


# ==========================================
# 9. NUMERIC / CATEGORICAL FEATURES
# ==========================================

numeric_features = [

    "vCPU",
    "Memory_GiB",
    "ClockSpeed_GHz",
    "GPU",
    "GPU_Memory_GB",
    "Normalization Size Factor",
    "Storage_Count",
    "Total_Storage_GB",
    "EBS_Throughput_Mbps"

]


categorical_features = [

    "Storage_Type",
    "Instance Type",
    "Instance Family",
    "Processor Architecture",
    "Current Generation",
    "Region Code",
    "Operating System",
    "Tenancy",
    "License Model",
    "CapacityStatus",
    "Pre Installed S/W",
    "TermType",
    "PurchaseOption",
    "LeaseContractLength",
    "OfferingClass"

]


# ==========================================
# 10. PREPROCESSING
# ==========================================

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Not Applicable"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )

    ]
)


# ==========================================
# 11. TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)


print("\nTrain shape:", X_train.shape)

print("Test shape:", X_test.shape)


# ==========================================
# 12. RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestRegressor


rf_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(

                n_estimators=200,

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


print("Training completed.")


# ==========================================
# 13. PREDICTION
# ==========================================

rf_pred = rf_model.predict(
    X_test
)


# ==========================================
# 14. MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


mae = mean_absolute_error(
    y_test,
    rf_pred
)


mse = mean_squared_error(
    y_test,
    rf_pred
)


rmse = np.sqrt(mse)


r2 = r2_score(
    y_test,
    rf_pred
)


print("\n==============================")

print("RANDOM FOREST RESULTS")

print("==============================")


print("MAE :", mae)

print("RMSE:", rmse)

print("R2  :", r2)


# ==========================================
# 15. TRAIN R2
# ==========================================

train_pred = rf_model.predict(
    X_train
)


train_r2 = r2_score(
    y_train,
    train_pred
)


print("\nTrain R2:", train_r2)

print("Test R2 :", r2)


# ==========================================
# 16. SAVE MODEL
# ==========================================

import joblib


joblib.dump(
    rf_model,
    "cloudcost_model.pkl"
)


print(
    "\nModel saved as cloudcost_model.pkl"
)


# ==========================================
# 17. SAVE DROPDOWN OPTIONS
# ==========================================

input_options = {}


for col in categorical_features:

    input_options[col] = sorted(
        X[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


joblib.dump(
    input_options,
    "input_options.pkl"
)


print(
    "Input options saved as input_options.pkl"
)