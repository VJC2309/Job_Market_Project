import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="Job Market Analysis",
    page_icon="💼",
    layout="wide"
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("💼 Job Market Analysis & Salary Prediction")

st.write(
    "Analyze job market data and predict salary using Machine Learning."
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("data/job_market.csv")


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "💰 Salary Prediction"]
)


# =================================================
# DASHBOARD
# =================================================

if page == "📊 Dashboard":

    st.header("📊 Job Market Dashboard")

    # ---------------------------------------------
    # KEY STATISTICS
    # ---------------------------------------------

    total_jobs = len(df)
    average_salary = df["Salary USD"].mean()
    maximum_salary = df["Salary USD"].max()
    minimum_salary = df["Salary USD"].min()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Jobs",
        f"{total_jobs:,}"
    )

    col2.metric(
        "Average Salary",
        f"${average_salary:,.0f}"
    )

    col3.metric(
        "Maximum Salary",
        f"${maximum_salary:,.0f}"
    )

    col4.metric(
        "Minimum Salary",
        f"${minimum_salary:,.0f}"
    )


    st.divider()


    # ---------------------------------------------
    # JOBS BY EXPERIENCE LEVEL
    # ---------------------------------------------

    st.subheader("👨‍💼 Jobs by Experience Level")

    experience = df["Experience Level"].value_counts()

    fig1, ax1 = plt.subplots()

    ax1.bar(
        experience.index,
        experience.values
    )

    ax1.set_xlabel("Experience Level")
    ax1.set_ylabel("Number of Jobs")
    ax1.set_title("Jobs by Experience Level")

    plt.xticks(rotation=20)

    st.pyplot(fig1)


    # ---------------------------------------------
    # TOP JOB TITLES
    # ---------------------------------------------

    st.subheader("💼 Top 10 Most Common Job Titles")

    top_jobs = (
        df["Job Title"]
        .value_counts()
        .head(10)
    )

    fig2, ax2 = plt.subplots()

    ax2.barh(
        top_jobs.index[::-1],
        top_jobs.values[::-1]
    )

    ax2.set_xlabel("Number of Jobs")
    ax2.set_ylabel("Job Title")
    ax2.set_title("Top 10 Job Titles")

    st.pyplot(fig2)


    # ---------------------------------------------
    # TOP COMPANY LOCATIONS
    # ---------------------------------------------

    st.subheader("🌎 Top 10 Company Locations")

    locations = (
        df["Company Location"]
        .value_counts()
        .head(10)
    )

    fig3, ax3 = plt.subplots()

    ax3.bar(
        locations.index,
        locations.values
    )

    ax3.set_xlabel("Country")
    ax3.set_ylabel("Number of Jobs")
    ax3.set_title("Top Company Locations")

    plt.xticks(rotation=45)

    st.pyplot(fig3)


    # ---------------------------------------------
    # SALARY BY EXPERIENCE
    # ---------------------------------------------

    st.subheader("💰 Average Salary by Experience Level")

    salary_experience = (
        df.groupby("Experience Level")["Salary USD"]
        .mean()
        .sort_values(ascending=False)
    )

    fig4, ax4 = plt.subplots()

    ax4.bar(
        salary_experience.index,
        salary_experience.values
    )

    ax4.set_xlabel("Experience Level")
    ax4.set_ylabel("Average Salary (USD)")
    ax4.set_title("Average Salary by Experience Level")

    plt.xticks(rotation=20)

    st.pyplot(fig4)


# =================================================
# SALARY PREDICTION
# =================================================

else:

    st.header("💰 Salary Prediction")

    st.write(
        "Enter job details below to estimate the expected salary."
    )


    # ---------------------------------------------
    # FEATURES
    # ---------------------------------------------

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


    # ---------------------------------------------
    # MACHINE LEARNING MODEL
    # ---------------------------------------------

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


    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )


    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


    pipeline.fit(X, y)


    # ---------------------------------------------
    # USER INPUT
    # ---------------------------------------------

    st.subheader("Enter Job Details")


    col1, col2 = st.columns(2)


    with col1:

        job_title = st.selectbox(
            "Job Title",
            sorted(df["Job Title"].unique())
        )

        experience = st.selectbox(
            "Experience Level",
            sorted(df["Experience Level"].unique())
        )

        employment = st.selectbox(
            "Employment Type",
            sorted(df["Employment Type"].unique())
        )

        work_model = st.selectbox(
            "Work Model",
            sorted(df["Work Model"].unique())
        )


    with col2:

        work_year = st.number_input(
            "Work Year",
            min_value=2020,
            max_value=2030,
            value=2025,
            step=1
        )

        residence = st.selectbox(
            "Employee Residence",
            sorted(df["Employee Residence"].unique())
        )

        company_location = st.selectbox(
            "Company Location",
            sorted(df["Company Location"].unique())
        )

        company_size = st.selectbox(
            "Company Size",
            sorted(df["Company Size"].unique())
        )


    # ---------------------------------------------
    # PREDICTION
    # ---------------------------------------------

    if st.button("🔮 Predict Salary"):

        input_data = pd.DataFrame({

            "Job Title": [job_title],

            "Experience Level": [experience],

            "Employment Type": [employment],

            "Work Model": [work_model],

            "Work Year": [work_year],

            "Employee Residence": [residence],

            "Company Location": [company_location],

            "Company Size": [company_size]

        })


        prediction = pipeline.predict(
            input_data
        )[0]


        st.success(
            f"💰 Predicted Salary: ${prediction:,.2f} per year"
        )


        st.info(
            "This prediction is an estimate generated using "
            "the project's dataset and Random Forest machine-learning model."
        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Job Market Analysis and Salary Prediction Using Python"
)