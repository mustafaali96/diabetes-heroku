import streamlit as st
import pickle


model = pickle.load(open("dt_model.sav", 'rb'))

st.title("Would you have diabetes?")
st.subheader("This model will predict if you have diabetes or not")

# 'pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree
pregnant = st.slider("Pregnant", 0, 17, 0)
insulin = st.slider("insulin", 0, 846, 0)
bmi = st.slider("BMI", 0, 70, 0)
age = st.slider("Age", 20, 85, 25)
glucose = st.slider("Glucose", 0, 200, 130)
bp = st.slider("BP", 0, 140, 80)
pedigree = st.slider("Pedigree", 0, 3, 1)
# pedigree = st.slider("Pedigree", 0.078, 2.42, 1)


# input_data = scaler.transform([[sex , age, f_class , s_class, t_class]])
prediction = model.predict([[pregnant, insulin, bmi, age, glucose, bp, pedigree]])
predict_probability = model.predict_proba([[pregnant, insulin, bmi, age, glucose, bp, pedigree]])

if prediction[0] == 1:
	st.subheader('You have diabetes with a probability of {}%'.format(round(predict_probability[0][1]*100 , 3)))
else:
	st.subheader("You don't not have diabetes with a probability of {}%".format(round(predict_probability[0][0]*100 , 3)))

# streamlit run diabetes.py