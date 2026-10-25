import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


st.title('Titanic Survival Chance')
Pclass = st.slider('Enter the passenger class for the user',1,3)
Sex = st.selectbox('Enter the gender of the passenger',['male',"female"])
SibSp = st.slider('Enter the total number of siblings and spouse of the passenger',1,8)
Parch = st.slider('Enter the total number of Parents and children of the passenger',0,6)
Fare = st.number_input('Enter the fare ')
Embarked = st.selectbox('Enter the station from where passenger started their journey',['Chebourg','Queenstown','Southampton'])

data = pd.DataFrame([{'Pclass':Pclass, 'Sex':Sex, 'SibSp':SibSp, 'Parch':Parch, 'Fare':Fare, 'Embarked':Embarked}])

model = load_model('model.h5')
with open('label_encoder.pkl','rb') as file:
    label = pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    onehot = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])
embarked = onehot.transform(data[['Embarked']])
embarked = pd.DataFrame(embarked, columns=onehot.get_feature_names_out())
data = pd.concat([data.drop(columns=['Embarked']),embarked], axis=1)
data[['Pclass','SibSp','Parch','Fare']] = scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y = model.predict(data)
y=y[0][0]

def Chance(y):
    if y>0.5 :
        return 'Passenger will survive the journey'
    else:
        return 'Passenger will not survive the journey'

if st.button('Predict survival chance'):
    st.write('Probability of survival chance of passenger', y)
    st.write(Chance(y))