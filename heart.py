import pickle
from pyarrow import create_library_symlinks
from xgboost import XGBClassifier
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

print(pickle.format_version)
with open('xgb_model', "rb") as model:
    clf = pickle.load(model)

st.set_page_config(layout='wide')
st.title("Heart Failure Prediction")

st.sidebar.header("Input Features")
age = st.sidebar.number_input("Age",step=1,value=61)
creatinine_phosphokinase = st.sidebar.number_input("Creatinine phosphokinase",step=1,value=580)
ejection_fraction = st.sidebar.number_input("Ejection fraction",step=1,value=38)
platelets = st.sidebar.number_input("Platelets",value=260000)
serum_creatine = st.sidebar.number_input("Serum creatinine",step=0.1,value=1.4)
serum_sodium = st.sidebar.number_input("Serum sodium",step=1,value=137)
time = st.sidebar.number_input("Time",step=1,value=130)
diabetes = st.sidebar.selectbox('Diabetes',("Yes","No"))
anaemia = st.sidebar.selectbox('Anaemia',("Yes","No"))
high_blood = st.sidebar.selectbox('High blood pressure',("Yes","No"))
sex = st.sidebar.selectbox('Sex',("Male","Female"))
smoking = st.sidebar.selectbox('Smoking',("Yes","No"))

diabetes = 1 if diabetes=='yes' else 0
anaemia = 1 if anaemia=='yes' else 0
high_blood = 1 if high_blood=='yes' else 0
smoking = 1 if smoking=='yes' else 0
sex  = 1 if sex=='Male' else 0

predict_df = pd.DataFrame({'age':age, 'anaemia':anaemia, "creatinine_phosphokinase":creatinine_phosphokinase, "diabetes":diabetes, "ejection_fraction":ejection_fraction, 
                            'high_blood_pressure':high_blood, 'platelets':platelets, "serum_creatinine":serum_creatine, "serum_sodium":serum_sodium,'sex':sex, 'smoking':smoking,
                            'time':time},index=[1])


st.markdown("*press predict after inserting input*")
if st.button("predict"):
    res = clf.predict(predict_df.iloc[[-1]])[0]
    prob = clf.predict_proba(predict_df.iloc[[-1]])
    if res==1:
        st.markdown("Person is predicted to have heart failure " + "with "+ str(prob[0][1]) +" probability")
    else:
        st.write("Person is predicted to be healthy "+ "with "+ str(prob[0][0]) +" probability")

st.markdown("""---""")
# create view distributions
st.subheader("View distributions")

# read in dataframe
data = pd.read_csv("heart_data.csv")
num_var = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
selected_var = st.multiselect('Numerical variables', num_var)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

for var in selected_var:
    fig = plt.figure(figsize=(10, 4))
    ax = sns.kdeplot(data[var])
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    target_var = 0
    if var=='age':
        target_var = age
    elif var=='creatinine_phosphokinase':
        target_var = creatinine_phosphokinase
    elif var=='ejection_fraction':
        target_var = ejection_fraction
    elif var=='platelets':
        target_var = platelets
    elif var=='serum_creatinine':
        target_var = serum_creatine
    elif var=='serum_sodium':
        target_var = serum_sodium
    else:
        target_var = time
    point_id = find_nearest(x,target_var)
    plt.plot(x[point_id],y[point_id],'bo',ms=10)
    st.pyplot(fig)
