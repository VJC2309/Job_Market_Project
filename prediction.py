import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("data/job_market.csv")

print("=" * 60)
print("       JOB SALARY PREDICTION USING MACHINE LEARNING")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Total records:", len(df))


# ---------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ---------------------------------------------------

features = [
    "Job Title",
    "Experience Level",
    "Employment Type",
    "Work Model",
    "Work Year",
    "Employee Residence",
    "Company Location",
    "Company Size"
]

target = "Salary USD"

X = df[features]
y = df[target]


# ---------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nData splitting completed!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ---------------------------------------------------
# 4. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ---------------------------------------------------

categorical_features = [
    "Job Title",
    "Experience Level",
    "Employment Type",
    "Work Model",
    "Employee Residence",
    "Company Location",
    "Company Size"
]

numerical_features = [
    "Work Year"
]


# ---------------------------------------------------
# 5. ENCODE CATEGORICAL DATA
# ---------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ---------------------------------------------------
# 6. CREATE MACHINE LEARNING MODEL
# ---------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------
# 7. CREATE COMPLETE PIPELINE
# ---------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ---------------------------------------------------
# 8. TRAIN MODEL
# ---------------------------------------------------

print("\nTraining machine learning model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ---------------------------------------------------
# 9. MAKE PREDICTIONS
# ---------------------------------------------------

y_pred = pipeline.predict(X_test)


# ---------------------------------------------------
# 10. EVALUATE MODEL
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("             MODEL PERFORMANCE")
print("=" * 60)

print("Mean Absolute Error :", round(mae, 2))
print("Mean Squared Error  :", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R² Score            :", round(r2, 4))


# ---------------------------------------------------
# 11. EXAMPLE SALARY PREDICTION
# ---------------------------------------------------

sample_job = pd.DataFrame({
    "Job Title": ["Data Scientist"],
    "Experience Level": ["Senior-level"],
    "Employment Type": ["Full-time"],
    "Work Model": ["Remote"],
    "Work Year": [2025],
    "Employee Residence": ["United States"],
    "Company Location": ["United States"],
    "Company Size": ["Medium"]
})

predicted_salary = pipeline.predict(sample_job)[0]


print("\n" + "=" * 60)
print("             SAMPLE SALARY PREDICTION")
print("=" * 60)

print("Job Title          : Data Scientist")
print("Experience Level   : Senior-level")
print("Employment Type    : Full-time")
print("Work Model         : Remote")
print("Company Size       : Medium")

print("\nPredicted Salary   : $", round(predicted_salary, 2))

print("\n" + "=" * 60)
print("          MACHINE LEARNING COMPLETED")
print("=" * 60)