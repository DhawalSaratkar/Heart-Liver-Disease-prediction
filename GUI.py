import numpy as np
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

liver_disease = pickle.load(open('liverModel.sav','rb'))

heart_disease = pickle.load(open('heartModel.sav','rb'))

liver_data = pd.read_csv("Liver Patient Dataset (LPD)_train.csv", encoding='unicode_escape')
heart_data = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

data1 = liver_data.head(10)
data2 = heart_data.head(10)

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          [ 'Home',
                            'Liver Disease Prediction',
                           'Heart Disease Prediction',
                           ],
                          icons=['house','person','heart'],
                          default_index=0)


if (selected == 'Home'):

   st.title("Heart & Liver Disease Prediction Using Machine Learning Model")

   st.image("medicalProfessional.png")
   
   st.write("- Heart and liver diseases are two of the leading causes of death worldwide. Early diagnosis and intervention can improve the chances of survival for patients with these diseases.")
   st.write("- The aim of this research is to develop a novel machine learning algorithm that can accurately predict the likelihood of both heart disease and liver disease in individuals.")
   st.write("- The primary challenge is to create an effective hybrid model that can efficiently process medical data and provide reliable predictions for these two distinct health conditions.")
   
   
   st.write("You can see the dataset used to train the model for liver disease.")
   if st.checkbox("Show Table"):
        st.write("We are showing you the first ten entries of our dataset. we have total 11 attributes and 27158 entries in the dataset.")
        st.table(data1)
        
   st.write("You can see the dataset used to train the model for heart disease.")     
   if st.checkbox("Show Data"):
        st.write("We are showing you the first ten entries of our dataset. we have total 22 attributes and 253680 entries in the dataset.")
        st.table(data2)    
        
   st.write("To know more about our research you see and download the research paper we have published.")   
     
   st.write("check out this [link](https://www.jetir.org/view?paper=JETIR2402572)")  
           
    
    # Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    st.write("We highly reccomend you to first get to know about the insights of the information asked here.")     
    if st.checkbox("Insights"):
           
        st.write("The chosen database includes the following structure, with 1 for yes and no.")
        st.write("- Heart DiseaseorAttack: Indicates whether the person has had a heart disease or heart attack.")
        st.write("- HighBP: Indicates whether the person has been informed by a healthcare professional that they have High Blood Pressure.")
        st.write("- High Chol: Indicates whether the person has been told by a healthcare professional that they have high cholesterol.")
        st.write("- CholCheck: Indicates if the person has had their cholesterol levels checked in the last 5 years.")
        st.write("- BMI: Body Mass Index, calculated by dividing a person's weight (in kilograms) by the square of their height (in meters).  generally, a BMI of: less than 18.5 is below the normal BMI range. 18.5-24.9 is a normal BMI range.")
        st.write("- Smoker: indicates whether the person has smoked at least 100 cigarettes.")
        st.write("- Stroke: Indicates whether the person has a history of stroke.")
        st.write("- Diabetes: Indicates whether the person has a history of diabetes, or is currently pre-diabetes, or suffers from some type of diabetes.")
        st.write("- PhysActivity: Indicates whether the person practices any type of physical activity on a daily basis.")
        st.write("- Fruit: Indicates whether the person consumes 1 or more fruit(s) daily.")
        st.write("- Veggies: Indicates whether the person consumes 1 or more vegetables daily.")
        st.write("- HvyAlcoholConsumption: Indicates whether the person drinks more than 14 drinks per week.")
        st.write("- AnyHealthcare: Indicates whether the person has any type of health plan.")
        st.write("- NoDocbcCost: Indicates if the person wanted to see a doctor in the last 1 year, but was unable to do so due to the cost.")
        st.write("- GenHealth: Indicates the person's response to how well their general health is, ranging from 1 (excellent) to 5 (poor).")
        st.write("- MentHealth: Indicates the number of days in the last 30 days that the person had mental health problems.")
        st.write("- PhysHealth: Indicates the number of days in the last 30 days that the person had physical health problems")
        st.write("- DiffWalk: Indicates whether the person has difficulty walking or climbing stairs.")
        st.write("- Sex: Indicates the person's gender, where 0 is female and 1 is male.")
        st.write("- Age: Indicates the person's age range, where 1 is 18 years old to 24 years old up to 13 which is 60 years old or more, each interval between has a 5 year increment.")
        st.write("- Education: Indicates the highest year of schooling completed, 1 year means having never attended or only attended kindergarten and 6 years, having attended 4 years of college or more.")
        st.write("- Income: Indicates the person's income ranging from 1 to 10, where 1 represents a poor category, 5 represents a middle class, and 10 represents the rich category. values above 1 represents well to do income and values above 5 represents upper middle class category.")
        st.write(" ")
        st.write(" ")
        st.write(" ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bp = st.selectbox("HighBP", ["Yes", "No"])
        
        if bp == 'Yes':
            bp = 1.00
        else:
            bp = 0.00    
        
    with col2:
        chol = st.selectbox("HighChol", ["Yes", "No"])
        
        if chol == 'Yes':
            chol = 1.00
        else:
            chol = 0.00    
        
    with col3:
        ck = st.selectbox("CholChek", ["Yes", "No"])
        
        if ck == 'Yes':
            ck = 1.00
        else:
             ck = 0.00    
        
    with col1:
        smoker = st.selectbox("Smoker", ["Yes", "No"])
        
        if smoker == 'Yes':
            smoker = 1.00
        else:
            smoker = 0.00    
        
    with col2:
        stroke = st.selectbox("Stroke", ["Yes", "No"])
        
        if stroke == 'Yes':
            stroke = 1.00
        else:
            stroke = 0.00   
            
    with col3:
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        
        if diabetes == 'Yes':
            diabetes = 1.00
        else:
            diabetes = 0.00        
        
    with col1:
         bmi = st.number_input('Body Mass Index', 15.00, 30.00, step=0.01)    
        
    with col2:
        genHealth = st.number_input('General Health', 0.00, 5.00, step=1.00)   
        
    with col3:
        mentHealth = st.number_input('Mental Health', 0.00, 30.00, step = 1.00)   
        
    with col1:
        physHealth = st.number_input('Physical Health', 0.00, 30.00, step = 1.00)   
        
    with col2:
        alcoholic = st.selectbox("AlcoholConsumption", ["Yes", "No"])
        
        if alcoholic == 'Yes':
            alcoholic = 1.00
        else:
            alcoholic = 0.00    
        
    with col3:
        physicalAct = st.selectbox("PhysicalActivity", ["Yes", "No"])
        
        if physicalAct == 'Yes':
            physicalAct = 1.00
        else:
            physicalAct = 0.00         
        
    with col1:
        female = 0.00
        male  = 0.00  
        
        gender = st.selectbox("Gender", ["male", "female"])
        
        if gender == 'female':
            female = 1.00
        elif gender == 'male':
            male = 1.00          
        
    with col2:
        age_1 = 0.00 
        age_2 = 0.00
        age_3 = 0.00
        age_4 = 0.00
        age_5 = 0.00
        age_6 = 0.00
        age_7 = 0.00
        age_8 = 0.00
        age_9 = 0.00
        age_10 = 0.00
        age_11 = 0.00
        age_12 = 0.00
        age_13 = 0.00
         
        age = st.selectbox("Select your age category", ['Age_18-24', 'Age_25-29', 'Age_30-34',
       'Age_35-39', 'Age_40-44', 'Age_45-49', 'Age_50-54', 'Age_55-59',
       'Age_60-64', 'Age_65-69', 'Age_70-74', 'Age_75-79', 'Age_80 or above'])
        
        if age == 'Age_18-24':
            age_1 = 1.00
        elif age == 'Age_25-29':
            age_2 = 1.00
        elif age == 'Age_30-34':
            age_3 = 1.00
        elif age ==  'Age_35-39':
            age_4 = 1.00
        elif age ==  'Age_40-44' :
            age_5 = 1.00
        elif age ==  'Age_45-49':
            age_6 = 1.00
        elif age ==  'Age_50-54':
            age_7 = 1.00
        elif age == 'Age_55-59':
            age_8 = 1.00
        elif age == 'Age_60-64':
            age_9 = 1.00
        elif age == 'Age_65-69':
            age_10 = 1.00
        elif age ==  'Age_70-74':
            age_11 = 1.00
        elif age == 'Age_75-79':
            age_12 = 1.00
        elif age == 'Age_80 or above':
            age_13 = 1.00   
        
    with col3:
        fruits = st.selectbox("fruits", ["Yes", "No"])
        
        if fruits == 'Yes':
            fruits = 1.00
        else:
            fruits = 0.00 
        
    with col1:
        veggies = st.selectbox("Veggies", ["Yes", "No"])
        
        if veggies == 'Yes':
            veggies = 1.00
        else:
            veggies = 0.00 
        
    with col2:
        DiffWalk = st.selectbox("DiffWalk", ["Yes", "No"])
        
        if DiffWalk == 'Yes':
            DiffWalk = 1.00
        else:
            DiffWalk = 0.00    
        
    with col3:
        education = st.number_input('Education', 1.00, 6.00, step = 1.00)
        
    with col1:
        income = st.number_input('Income', 1.00, 10.00, step=1.00)
        
    with col2:
        nodoc = st.selectbox("NoDocBcCost", ["Yes", "No"])
        
        if nodoc == 'Yes':
            nodoc = 1.00
        else:
            nodoc = 0.00    
        
    with col3:
        healthcare = st.selectbox("Any other health issue", ["Yes", "No"])
        
        if healthcare == 'Yes':
            healthcare = 1.00
        else:
            healthcare = 0.00
                                           
            
    # Create a button and store the click status in a variable
    clicked = st.button("Predict")

    # If the button is clicked, perform some action 
    if clicked:
      prediction = heart_disease.predict([[bp, chol, ck, bmi, smoker, stroke, diabetes, physicalAct, fruits, veggies, alcoholic, healthcare, nodoc, genHealth, mentHealth, physHealth, DiffWalk, education, income,female, male, age_1, age_2, age_3, age_4, age_5, age_6, age_7, age_8, age_9, age_10, age_11, age_12, age_13]])
    
      if prediction == 1:
          st.write("You have chances of having a heart disease")
      else:
          st.write("You do not posses chances of having any heart problem.")
          st.write("Althogh we suggest you to maintain a stress free and healthy lifestyle.") 
          
    st.write(" ")      
    st.write(" ")  
    st.write(" ")  
     
    st.image("hd.jpg")            
          

    # Liver Disease Prediction Page
if (selected == 'Liver Disease Prediction'):
    
    # page title
    st.title('Liver Disease Prediction using ML')
    
    st.write("We highly reccomend you to first get to know about the insights of the information asked here.")     
    if st.checkbox("Insights"):
        st.write("The chosen database includes the following structure, having different measures for different types of attributes.")
        st.write("- Total Bilirubin: total bilirubin in adults is 0.1 to 1.2 milligrams per deciliter (mg/dL).")
        st.write("- Alkphos Alkaline Phosphotase: The normal range is 44 to 147 international units per liter (IU/L) or 0.73 to 2.45 microkatal per liter (µkat/L).")
        st.write("- Sgpt Alamine Aminotransferase: SGPT (serum glutamic-pyruvic transaminase) or ALT (alanine aminotransferase) levels in the blood is typically 7–56 units per liter (U/L).")
        st.write("- Sgot Aspartate Aminotransferase: The normal range of serum glutamic-oxaloacetic transaminase (SGOT) aspartate, also known as aspartate aminotransferase (AST), is typically between 8 and 45 units per liter")
        st.write("- Total Protien: The normal range is 6.0 to 8.3 grams per deciliter (g/dL) or 60 to 83 g/L.")
        st.write("- ALB Albumin: The normal range for albumin in an adult's blood is 3.4–5.4 grams per deciliter (g/dL), or 34–54 grams per liter (g/L).")
        st.write("- A/G Ratio Albumin and Globulin Ratio: The normal range for albumin-to-globulin (A/G) ratio is between 1.1 and 2.5, but this can vary by lab.")
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', 30.00, 75.00, step=1.00)
        
    with col2:
        bilirubin = st.number_input('Total Bilirubin', 0.01, 5.20, step=0.01)
        
    with col3:
        Alkphos = st.number_input('Alkphos Alkaline Phosphotase', 22.00, 347.00, step= 1.00)
        
    with col1:
        sgpt = st.number_input('Sgpt Alamine Aminotransferase', 3.00, 76.00, step=1.00)
        
    with col2:
        sgpot = st.number_input('Sgot Aspartate Aminotransferase', 3.00, 85.00, step=1.00)
        
    with col3:
        protien = st.number_input('Total Protiens', 2.00, 15.30, step=0.01)
        
    with col1:
        ALB = st.number_input('ALB Albumin', 14.00, 84.00, step=1.00)                
            
    with col2:
        ratio = st.number_input('A/G Ratio Albumin and Globulin Ratio', 0.10, 5.50, step=0.01)
        
    with col3:
        female = 0.00
        male  = 0.00  
        
        gender = st.selectbox("Gender", ["male", "female"])
        
        if gender == 'female':
            female = 1.00
        elif gender == 'male':
            male = 1.00 
            
        
    # Create a button and store the click status in a variable
    clicked = st.button("Predict")

    # If the button is clicked, perform some action (replace with your logic)
    if clicked:
      prediction = liver_disease.predict([[age,bilirubin,Alkphos,sgpt,sgpot,protien,ALB,ratio,female,male]])
      
      if prediction == 1:
          st.write("You have chances of having a liver disease")
      else:
          st.write("You do not posses chances of having any liver problem.")
          st.write("Althogh we suggest you to maintain a stress free and healthy lifestyle.")
    st.write(" ")      
    st.write(" ")  
    st.write(" ")  
    
    st.image("liver.jpg")       
    
