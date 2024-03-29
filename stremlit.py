import numpy as np
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

heart_disease = pickle.load(open('mymodel.sav','rb'))

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Liver Disease Prediction',
                           'Heart Disease Prediction',
                           ],
                          icons=['health','heart','person'],
                          default_index=0)
    
    # Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Gender')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')


    # Liver Disease Prediction Page
if (selected == 'Liver Disease Prediction'):
    
    # page title
    st.title('Liver Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Gender')
        
    with col3:
        cp = st.text_input('Total Bilirubin')
        
    with col1:
        trestbps = st.text_input('Total Protiens')
        
    with col2:
        chol = st.text_input('ALB Albumin')
        


# Create a button and store the click status in a variable
clicked = st.button("Predict")

# If the button is clicked, perform some action (replace with your logic)
if clicked:
    # Your prediction logic here
    st.write("Prediction in progress...")
    # Replace the following with your actual prediction code
    prediction = "This is your predicted outcome"
    st.success(f"Prediction: {prediction}")
