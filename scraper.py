import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Website URL
URL = "https://realpython.github.io/fake-jobs/"

# Send request to website
response = requests.get(URL)

# Check if website opened successfully
if response.status_code == 200:
    print("Website connected successfully!")
else:
    print("Unable to connect to website.")
    exit()

# Parse HTML using Beautiful Soup
soup = BeautifulSoup(response.content, "html.parser")

# Find all job cards
job_cards = soup.find_all("div", class_="card-content")

jobs = []

# Extract information from each job
for job in job_cards:

    title = job.find("h2", class_="title")
    company = job.find("h3", class_="company")
    location = job.find("p", class_="location")

    if title and company and location:

        job_title = title.get_text(strip=True)
        company_name = company.get_text(strip=True)
        job_location = location.get_text(strip=True)

        jobs.append({
            "Job Title": job_title,
            "Company": company_name,
            "Location": job_location
        })

# Convert collected data into Pandas DataFrame
df = pd.DataFrame(jobs)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save data to CSV
df.to_csv("data/jobs.csv", index=False)

# Display results
print("\nJob data collected successfully!")
print("Total jobs collected:", len(df))

print("\nFirst 10 jobs:")
print(df.head(10))

print("\nData saved to:")
print("data/jobs.csv")