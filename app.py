import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title('Analysis and Prediction of Penguins')
st.header('Data Exploration')

data = sns.load_dataset('penguins')

st.header('Penguins Species Distribution')
st.bar_chart(data['species'].value_counts())

st.header('Penguis Model')
data.dropna(inplace=True)

X = data.drop('species', axis=1)
X = pd.get_dummies(X)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

st.header('Models Metrics')
st.write(f'Model Accuracy: {accuracy}')
st.write('Classification Report')
st.write(classification_report)

island = st.selectbox('Island', data['island'].unique())
bill_length = st.slider('Bill length', data['bill_length_mm'].min(),  data['bill_length_mm'].max())
bill_depth = st.slider('Bill depth', data['bill_depth_mm'].min(),  data['bill_depth_mm'].max())
flipper_length = st.slider('Flipper length', data['flipper_length_mm'].min(),  data['flipper_length_mm'].max())
body_mass = st.slider('Body Mass', data['body_mass_g'].min(),  data['body_mass_g'].max())
sex = st.selectbox('Sex', data['sex'].unique())

user_data = {
    'island': island,
    'bill_length':bill_length,
    'bill_depth':bill_depth,
    'flipper_length':flipper_length,
    'body_mass':body_mass,
    'sex':sex
}

user_data = pd.DataFrame(user_data, index=[0])
user_data = pd.get_dummies(user_data)
user_data = user_data.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(user_data)
st.subheader('Penguin Species Prediction')
st.write(f'Predicted Species: {prediction[0]}')