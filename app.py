import streamlit as st  
import joblib
import numpy as np 
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("mymodel.joblib")

st.title("sales Predicton App")

tv  = st.slider("TV Advertising Budget",max_value=500)
radio  = st.slider("Radio Advertising Budget",max_value=500)
newspaper  = st.slider("Newspaper Advertising Budget",max_value=500)


input_data = np.array([[tv,radio,newspaper]])

prediction = model.predict(input_data)

st.subheader(f"Predicted sales {prediction[0]:.2f}")