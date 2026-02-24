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
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Linear Regression Model 📑","Prediction and Solution"])


st.image("burnout.jpg")

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("work_from_home_burnout_dataset.csv")
