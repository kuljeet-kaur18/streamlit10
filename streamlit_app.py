import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load the data into a DataFrame
df = pd.read_csv("C://Users//San//Desktop//streamlit app//bank-additional-full.csv", delimiter=";")

# Set page title and favicon
st.set_page_config(page_title="Bank Marketing Predictor", page_icon=":bank:")

# Title of our app
st.title("Bank Marketing Dataset - Predicting Term Deposit Subscriptions using SVM")

# Add a checkbox to show the dataset
if st.checkbox("Show Dataset"):
    st.write(df)

# Creating columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Select Age", int(df['age'].min()), int(df['age'].max()), int(df['age'].median()))
    job = st.selectbox("Select Job", df['job'].unique())
    marital = st.selectbox("Select Marital Status", df['marital'].unique())
    education = st.selectbox("Select Education", df['education'].unique())
    default = st.selectbox("Select Default", df['default'].unique())
    housing = st.selectbox("Select Housing", df['housing'].unique())
    loan = st.selectbox("Select Loan", df['loan'].unique())

with col2:
    contact = st.selectbox("Select Contact", df['contact'].unique())
    month = st.selectbox("Select Month", df['month'].unique())
    day_of_week = st.selectbox("Select Day of Week", df['day_of_week'].unique())
    duration = st.slider("Select Duration", int(df['duration'].min()), int(df['duration'].max()), int(df['duration'].median()))
    campaign = st.slider("Select Campaign", int(df['campaign'].min()), int(df['campaign'].max()), int(df['campaign'].median()))
    pdays = st.slider("Select pdays", int(df['pdays'].min()), int(df['pdays'].max()), int(df['pdays'].median()))
    previous = st.slider("Select Previous", int(df['previous'].min()), int(df['previous'].max()), int(df['previous'].median()))

with st.expander("Advanced Settings"):
    poutcome = st.selectbox("Select Poutcome", df['poutcome'].unique())
    emp_var_rate = st.slider("Select emp.var.rate", float(df["emp.var.rate"].min()), float(df["emp.var.rate"].max()), float(df["emp.var.rate"].median()))
    cons_price_idx = st.slider("Select cons.price.idx", float(df["cons.price.idx"].min()), float(df["cons.price.idx"].max()), float(df["cons.price.idx"].median()))
    cons_conf_idx = st.slider("Select cons.conf.idx", float(df["cons.conf.idx"].min()), float(df["cons.conf.idx"].max()), float(df["cons.conf.idx"].median()))
    euribor3m = st.slider("Select euribor3m", float(df['euribor3m'].min()), float(df['euribor3m'].max()), float(df['euribor3m'].median()))
    nr_employed = st.slider("Select nr.employed", float(df['nr.employed'].min()), float(df['nr.employed'].max()), float(df['nr.employed'].median()), float(df['nr.employed'].median()))
    
    
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
le_dict = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

feature_columns = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
X = df[feature_columns]
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = SVC(random_state=0, kernel='rbf')
clf.fit(X_train, y_train)

if st.button("Predict"):
    user_input = pd.DataFrame({"age": [age], "job": [job], "marital": [marital], "education": [education], "default": [default], "housing": [housing], "loan": [loan], "contact": [contact], "month": [month], "day_of_week": [day_of_week], "duration": [duration], "campaign": [campaign], "pdays": [pdays], "previous": [previous], "poutcome": [poutcome], "emp.var.rate": [emp_var_rate], "cons.price.idx": [cons_price_idx], "cons.conf.idx": [cons_conf_idx], "euribor3m": [euribor3m], "nr.employed": [nr_employed]})
    # Transform user input categorical columns
    for col in categorical_columns:
        user_input[col] = le_dict[col].transform(user_input[col])

    prediction = clf.predict(user_input)
    st.markdown("---")
    if prediction[0] == "no":
        st.error("This customer is not likely to subscribe for a term deposit.")
    else:
        st.success("This customer is likely to subscribe for a term deposit.")
