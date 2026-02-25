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

## Step 02 - Load dataset
df = pd.read_csv("work_from_home_burnout_dataset.csv")

# BUILD MODEL
# Clean data
df2 = df.dropna()

# Label non-number features 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# le.fit()
df2["day_type"] = le.fit_transform(df2["day_type"])

# Divide train - test set
from sklearn.model_selection import train_test_split
X = df2.drop(["user_id", "burnout_score", "burnout_risk"], axis=1)
y = df2["burnout_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=42)

# Build and train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)


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

elif page == "Visualization":

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

# Page 3: 
elif page == "Linear Regression Model": 

    # Model information
    st.title("Burnout Prediction Model")
    st.subheader("What Drives Employee Exhaustion?")
    st.markdown("---")

    st.text("""
        We built a Linear Regression model to predict employee burnout score based on work behavior and lifestyle variables.

        Our goal is not only to predict burnout, but to identify which factors increase or decrease it the most.

        The target variable is burnout_score, a continuous measure of employee exhaustion.        
    """)

    st.subheader("How Our Model Works")
    st.markdown("##### Input Features")
    st.markdown("""
        - work_hours
        - screen_time_hours
        - meetings_count
        - breaks_taken
        - after_hours_work
        - sleep_hours
        - task_completion_rate
        - day_type (encoded)
    """)

    st.markdown("##### Train - Test Size")
    st.markdown("""
        - Train set: 80%
        - Test set: 20%
    """)

    st.markdown("##### Model Performance")
    st.text("R² score indicates how well our model explains variation in burnout. A higher R² means better predictive power.")
    st.markdown(f"""
        - R² Score: {r2}
        - MAE (Mean Absolute Error): {mae}
        - MSE: {rmse}
    """)


    # Linear Regression Chart

# Page 4:
elif page == "Prediction and Solution":
    
    # Input arguments
    ''' work_hours = st.slider()
    screen_time_hours = st.slider()
    meetings_count = st.slider()
    breaks_taken = st.slider()
    after_hours_work = st.slider()
    sleep_hours = st.slider()
    task_completion_rate = st.slider()'''

    work_hours = st.slider("Work Hours", 0.0, 16.0, 8.0)
    screen_time_hours = st.slider("Screen Time Hours", 0.0, 16.0, 8.0)
    meetings_count = st.slider("Meetings Count", 0, 10, 2)
    breaks_taken = st.slider("Breaks Taken", 0, 10, 2)
    after_hours_work = st.slider("After Hours Work (0=No, 1=Yes)", 0, 1, 0)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    task_completion_rate = st.slider("Task Completion Rate (%)", 0.0, 100.0, 80.0)

    # day_type = st.selectbox(["Weekday", "Weekend"]) 
    day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
    day_type =  0  if (day_type == "Weekday") else 1

    # Compose argument into an array
    
    # user_inputs = np.array(["day_type","work_hours","screen_time_hours","meetings_count","breaks_taken","after_hours_work","sleep_hours","task_completion_rate"])
    user_inputs = np.array([[ 
    work_hours,
    screen_time_hours,
    meetings_count,
    breaks_taken,
    after_hours_work,
    sleep_hours,
    task_completion_rate,
    day_type
    ]])

    # Predict
    # user_prediction = model.predict(user_inputs)

    if st.button("Predict Burnout"):
        user_prediction = model.predict(user_inputs)
        st.success(f"Predicted Burnout Score: {round(user_prediction[0], 2)}")
    
