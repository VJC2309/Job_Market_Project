import pandas as pd

# Public salary dataset
url = "https://gist.githubusercontent.com/SeshadriKamini7/14ed8d61ff06a071338bf120f64dfa78/raw/data_science_salaries.csv"

print("Loading salary dataset...")

# Read dataset
df = pd.read_csv(url)

print("Dataset loaded successfully!")

# Display original columns
print("\nOriginal columns:")
print(df.columns.tolist())

# Keep the columns useful for our project
columns_to_keep = [
    "job_title",
    "experience_level",
    "employment_type",
    "work_models",
    "work_year",
    "employee_residence",
    "salary_in_usd",
    "company_location",
    "company_size"
]

df = df[columns_to_keep]

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing important values
df = df.dropna()

# Rename columns to simpler names
df = df.rename(columns={
    "job_title": "Job Title",
    "experience_level": "Experience Level",
    "employment_type": "Employment Type",
    "work_models": "Work Model",
    "work_year": "Work Year",
    "employee_residence": "Employee Residence",
    "salary_in_usd": "Salary USD",
    "company_location": "Company Location",
    "company_size": "Company Size"
})

# Create data folder if it does not exist
import os
os.makedirs("data", exist_ok=True)

# Save cleaned dataset
df.to_csv("data/job_market.csv", index=False)

print("\nData cleaning completed!")
print("Number of records:", len(df))

print("\nCleaned dataset:")
print(df.head(10))

print("\nDataset saved as:")
print("data/job_market.csv")