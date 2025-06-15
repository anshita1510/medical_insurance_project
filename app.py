
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Insurance Charges Analysis", layout="wide")

# --- Sidebar ---
st.sidebar.title("Insurance EDA & Prediction")
uploaded_file = st.sidebar.file_uploader("C:/Users/Acer/Downloads/Medical_charges-main/insurance.csv", type=["csv"])

# --- Load Data ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_data(uploaded_file)
    st.title("📊 Insurance Charges EDA Dashboard")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### 📌 Dataset Summary")
    st.write(df.describe())

    # --- EDA Visuals ---
    st.subheader("🔍 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='age', marginal='box', nbins=47, title='Age Distribution')
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig)

        fig = px.histogram(df, x='charges', marginal='box', title='Charges Distribution', color='smoker')
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig)

    with col2:
        fig = px.histogram(df, x='bmi', marginal='box', title='BMI Distribution')
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig)

        fig = px.histogram(df, x='smoker', color='sex', title='Smoker by Gender')
        st.plotly_chart(fig)

    # --- Scatter Plots ---
    st.subheader("📈 Relationship Analysis")
    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(df, x='age', y='charges', color='smoker',
                         marginal_x='box', marginal_y='box', title='Age vs Charges')
        st.plotly_chart(fig)

    with col4:
        fig = px.scatter(df, x='bmi', y='charges', color='smoker',
                         marginal_x='box', marginal_y='box', title='BMI vs Charges')
        st.plotly_chart(fig)

    # --- Correlation Check ---
    st.subheader("📊 Correlation Insights")
    df['smoker_num'] = df['smoker'].map({'no': 0, 'yes': 1})
    st.write("**Age vs Charges correlation:**", round(df['charges'].corr(df['age']), 2))
    st.write("**BMI vs Charges correlation:**", round(df['charges'].corr(df['bmi']), 2))
    st.write("**Smoker vs Charges correlation:**", round(df['charges'].corr(df['smoker_num']), 2))

    # --- Simple Regression ---
    st.subheader("🤖 Predict Charges Using Age + Smoker + BMI")

    # Prepare data
    df_encoded = pd.get_dummies(df, drop_first=True)
    features = ['age', 'bmi', 'smoker_yes']
    X = df_encoded[features]
    y = df_encoded['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
    st.markdown(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")

    # --- Prediction Interface ---
    st.sidebar.subheader("Try Your Own Prediction")
    input_age = st.sidebar.slider("Age", 18, 65, 30)
    input_bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
    input_smoker = st.sidebar.radio("Smoker?", ("yes", "no"))

    input_data = pd.DataFrame({
        'age': [input_age],
        'bmi': [input_bmi],
        'smoker_yes': [1 if input_smoker == 'yes' else 0]
    })

    predicted_charge = model.predict(input_data)[0]
    st.sidebar.markdown(f"### 💰 Estimated Charges: **${predicted_charge:,.2f}**")
else:
    st.title("📊 Insurance Charges EDA Dashboard")
    st.info("Please upload the insurance dataset to get started.")


