import pandas as pd

# ---------------------------------------------------
# JOB MARKET ANALYSIS
# ---------------------------------------------------

# Load the cleaned dataset
df = pd.read_csv("data/job_market.csv")

print("=" * 60)
print("        JOB MARKET ANALYSIS")
print("=" * 60)

# 1. Basic information
print("\n1. BASIC DATASET INFORMATION")
print("-" * 40)

print("Total number of jobs:", len(df))
print("Total number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# 2. First five records
print("\n2. FIRST 5 RECORDS")
print("-" * 40)

print(df.head())


# 3. Check missing values
print("\n3. MISSING VALUES")
print("-" * 40)

print(df.isnull().sum())


# 4. Most common job titles
print("\n4. TOP 10 MOST COMMON JOB TITLES")
print("-" * 40)

top_jobs = df["Job Title"].value_counts().head(10)

print(top_jobs)


# 5. Jobs by experience level
print("\n5. JOBS BY EXPERIENCE LEVEL")
print("-" * 40)

experience_count = df["Experience Level"].value_counts()

print(experience_count)


# 6. Average salary by experience level
print("\n6. AVERAGE SALARY BY EXPERIENCE LEVEL")
print("-" * 40)

average_salary_experience = (
    df.groupby("Experience Level")["Salary USD"]
    .mean()
    .sort_values(ascending=False)
)

print(average_salary_experience)


# 7. Jobs by employment type
print("\n7. JOBS BY EMPLOYMENT TYPE")
print("-" * 40)

employment_count = df["Employment Type"].value_counts()

print(employment_count)


# 8. Jobs by company size
print("\n8. JOBS BY COMPANY SIZE")
print("-" * 40)

company_size_count = df["Company Size"].value_counts()

print(company_size_count)


# 9. Top company locations
print("\n9. TOP 10 COMPANY LOCATIONS")
print("-" * 40)

top_locations = df["Company Location"].value_counts().head(10)

print(top_locations)


# 10. Highest-paying job titles
print("\n10. TOP 10 HIGHEST-PAYING JOB ROLES")
print("-" * 40)

highest_paying_jobs = (
    df.groupby("Job Title")["Salary USD"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print(highest_paying_jobs)


# 11. Overall average salary
print("\n11. OVERALL SALARY STATISTICS")
print("-" * 40)

print("Average Salary: $", round(df["Salary USD"].mean(), 2))
print("Minimum Salary: $", round(df["Salary USD"].min(), 2))
print("Maximum Salary: $", round(df["Salary USD"].max(), 2))


# 12. Save analysis results
print("\n12. SAVING ANALYSIS RESULTS")
print("-" * 40)

top_jobs.to_csv("data/top_job_titles.csv")
experience_count.to_csv("data/jobs_by_experience.csv")
average_salary_experience.to_csv("data/average_salary_by_experience.csv")
top_locations.to_csv("data/top_locations.csv")
highest_paying_jobs.to_csv("data/highest_paying_jobs.csv")

print("Analysis results saved successfully!")

print("\n" + "=" * 60)
print("        ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)