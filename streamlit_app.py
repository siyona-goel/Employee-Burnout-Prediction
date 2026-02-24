import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

st.set_page_config(
    page_title="Employee Burnout Predictive Dashboard 🫂 ",
    layout="centered",
    page_icon="🫂",
)

## Step 01 - Setup
st.sidebar.title("Employee Burnout Predictive Dashboard")
page = st.sidebar.selectbox("Select Page",["Introduction","Visualization", "Linear Regression Model","Prediction and Solution"])

st.image("burnout.jpg")

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("work_from_home_burnout_dataset.csv")

## Step 02 - Load dataset
if page == "Introduction":
    st.subheader("Welcome to the Employee Burnout Predictive Dashboard!")

    st.markdown("#### 🎯 Objectives")
    st.text("Although remote work increases flexibility for its employees, " \
    "it also increases the hidden burnout risk. A rise in burnout means a loss in the company's productivity," \
    " as well as worse mental health for its employees. In order to keep their employees happy and get " \
    "optimal performace, the company needs to be able to predict burnout so they can encourage its prevention.")

    st.write("   ")

    st.markdown("#### 🎯 Dataset")
    st.text("This dataset captures daily work-from-home behavioral patterns and their relationship " \
    "with employee burnout and productivity. It is designed to help analyze how factors such as " \
    "working hours, screen exposure, meetings, breaks, sleep, and after-hours work collectively " \
    "influence task completion efficiency and burnout risk.")

    st.write("   ")

    st.markdown("#### 📊 Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.write("   ")

    st.markdown("#### 🔴 Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.write("   ")

    st.markdown("#### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

if page == "Visualization 📊":

    st.title("Data Visualization & Insights 📊")
    st.write("In this section, we explore how work habits influence employee burnout.")


    # Burnout Distribution
    st.subheader("1️⃣ Burnout Score Distribution")

    fig1, ax1 = plt.subplots()
    sns.histplot(df["burnout_score"], kde=True, ax=ax1)
    ax1.set_xlabel("Burnout Score")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    st.markdown("""
    **Insight:**  
    This graph shows how burnout scores are distributed among employees.
    It helps us understand whether burnout levels are generally low, moderate, or extreme.
    """)

    # Burnout by Day Type
    st.subheader("2️⃣ Burnout by Day Type (Weekday vs Weekend)")

    fig2, ax2 = plt.subplots()
    sns.boxplot(x="day_type", y="burnout_score", data=df, ax=ax2)
    st.pyplot(fig2)

    st.markdown("""
    **Insight:**  
    The boxplot shows that median burnout levels are slightly higher on weekdays (just above 40) compared to weekends (just below 40). While the difference is not dramatic, it suggests that structured workdays may contribute modestly to increased stress levels.
    """)

    # Work Hours vs Burnout
    st.subheader("3️⃣ Average Burnout by Work Hours")

    work_hours_avg = df.groupby("work_hours")["burnout_score"].mean().reset_index()

    fig3, ax3 = plt.subplots()
    sns.lineplot(x="work_hours", y="burnout_score", data=work_hours_avg, marker="o", ax=ax3)
    ax3.set_ylabel("Average Burnout Score")
    st.pyplot(fig3)

    st.markdown("""
    **Insight:**  
    This line chart shows the average burnout score at each work-hour level. 
    We observe a slight upward trend, indicating that burnout tends to increase gradually as work hours increase.
    """)

    # Screen Time vs Burnout
    st.subheader("4️⃣ Average Burnout by Screen Time")

    screen_avg = df.groupby("screen_time_hours")["burnout_score"].mean().reset_index()

    fig4, ax4 = plt.subplots()
    sns.lineplot(x="screen_time_hours", y="burnout_score", data=screen_avg, marker="o", ax=ax4)
    ax4.set_ylabel("Average Burnout Score")
    st.pyplot(fig4)

    st.markdown("""
    **Insight:**  
    The upward trend suggests that increased screen exposure may contribute to higher burnout levels.
    This aggregated view makes the relationship clearer than individual data points.
    """)

    # Sleep Hours vs Burnout
    st.subheader("5️⃣ Sleep Hours vs Burnout (Trend Line)")

    fig5, ax5 = plt.subplots()
    sns.regplot(x="sleep_hours", y="burnout_score", data=df, ax=ax5)
    st.pyplot(fig5)

    st.markdown("""
    **Insight:**  
    The regression line appears nearly flat, confirming that sleep hours have little to no linear relationship with burnout in this dataset.
    """)

    # ------------------------------
    # Correlation Heatmap
    # ------------------------------
    st.subheader("6️⃣ Correlation Matrix")

    fig6, ax6 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    st.pyplot(fig6)

    st.markdown("""
    **Insight:**  
    The correlation matrix reveals that task completion rate has an extremely strong negative correlation with burnout score (-0.96), suggesting that lower productivity is strongly associated with higher burnout.
In contrast, work hours and screen time show only weak positive correlations with burnout (around 0.12), indicating that these factors alone do not strongly predict burnout levels.
Additionally, work hours and screen time are highly correlated with each other (0.95), which may introduce multicollinearity in the regression model and should be considered during model development.
    """)

    st.success("These insights will guide our Linear Regression model on the next page.")
