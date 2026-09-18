import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("data/job_market.csv")

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)

# Use seaborn style
sns.set_theme(style="whitegrid")

# -----------------------------
# GRAPH 1 - Top Job Titles
# -----------------------------
top_jobs = df["Job Title"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_jobs.values, y=top_jobs.index)
plt.title("Top 10 Most Common Job Titles")
plt.xlabel("Number of Jobs")
plt.ylabel("Job Title")
plt.tight_layout()
plt.savefig("graphs/top_job_titles.png")
plt.close()

# -----------------------------
# GRAPH 2 - Experience Level Pie Chart
# -----------------------------
experience = df["Experience Level"].value_counts()

plt.figure(figsize=(7,7))
plt.pie(experience.values,
        labels=experience.index,
        autopct="%1.1f%%",
        startangle=90)
plt.title("Jobs by Experience Level")
plt.tight_layout()
plt.savefig("graphs/experience_level_pie.png")
plt.close()

# -----------------------------
# GRAPH 3 - Average Salary
# -----------------------------
salary_exp = df.groupby("Experience Level")["Salary USD"].mean()

plt.figure(figsize=(8,6))
sns.barplot(x=salary_exp.index, y=salary_exp.values)
plt.title("Average Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Average Salary (USD)")
plt.tight_layout()
plt.savefig("graphs/average_salary_experience.png")
plt.close()

# -----------------------------
# GRAPH 4 - Company Size
# -----------------------------
company_size = df["Company Size"].value_counts()

plt.figure(figsize=(7,5))
sns.barplot(x=company_size.index, y=company_size.values)
plt.title("Jobs by Company Size")
plt.xlabel("Company Size")
plt.ylabel("Number of Jobs")
plt.tight_layout()
plt.savefig("graphs/company_size.png")
plt.close()

# -----------------------------
# GRAPH 5 - Top Locations
# -----------------------------
locations = df["Company Location"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=locations.values, y=locations.index)
plt.title("Top 10 Company Locations")
plt.xlabel("Number of Jobs")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("graphs/top_locations.png")
plt.close()

# -----------------------------
# GRAPH 6 - Highest Paying Jobs
# -----------------------------
highest_jobs = (
    df.groupby("Job Title")["Salary USD"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,6))
sns.barplot(x=highest_jobs.values, y=highest_jobs.index)
plt.title("Top 10 Highest Paying Job Roles")
plt.xlabel("Average Salary (USD)")
plt.ylabel("Job Title")
plt.tight_layout()
plt.savefig("graphs/highest_paying_jobs.png")
plt.close()

print("="*50)
print("All graphs created successfully!")
print("Graphs saved inside 'graphs' folder.")
print("="*50)