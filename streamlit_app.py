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

