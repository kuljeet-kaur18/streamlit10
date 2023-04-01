import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import accuracy_score

#load the data into dataframe
df= pd.read.csv("C:\Capstone\bank-additional-full.csv",delimiter=";")
# title of or app
st.title("Bank Marketing Dataset - Predicting Term Deposit Subcsriptions using SVM")

# Add a checkbox to show the dataset
if st.checkbox("show Dataset"):
    st.write (df)

# Define input fields for user input
age = st.slider("Select Age",56,37,24)
job = st.slider("Select Job","admin.","blue-collar","technician")
previous = st.slider("Select Previous",0,1)
if st.button("Predict"):user_input = pd.dataFrame({"age":[age],"job":[job],"previous":[previous]})

X = df[["age","job","previous"]]
y= df["y"]
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,randomstate=0)

# Train a SVM classifier Model 
clf = SVC(random_state=0,kernel='rbf')
clf.fit(X_train, y_train)

prediction = clf.predict(user_input)
if prediction[0] == "no":
    st.write("This customer is not likely to subscribe for a term deposit")
else:
    st.write(" this customer is likely to subscibe for a term deposit")

    

