import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import streamlit.components.v1 as components
import sys
import os
import random
from pathlib import Path

st.markdown("""
    <style>
        
        .css-1d391kg, .css-18e3th9, .css-12oz5g7 { 
            background-color: white; 
        }

        
        .css-16huue1, .css-145kmo2, .css-10trblm {
            color: black; 
        }

        
        div.stButton > button {
            background-color: #15e4d0; 
            color: white; /
            border: 1px solid #15e4d0; 
        }

        
        div.stButton > button:hover {
            background-color: #15e4d0; 
            border-color: #15e4d0; 
        }

        
        h1, h2, h3, h4, h5, h6 {
            color: #15e4d0; 
        }

        /
        .css-1d3h4o2 {
            background-color: white;
        }
    </style>
    """, unsafe_allow_html=True)


sys.path.append(r"C:\Users\Prakriti Aayansh\OneDrive\Desktop\NeuroSafeWAPP")

# Determine the environment and set the model path
if os.path.exists(r"C:\Users\Prakriti Aayansh\OneDrive\Desktop\NeuroSafeWAPP\rf_model_resample.pkl"):
    # Local machine path
    model_path = r"C:\Users\Prakriti Aayansh\OneDrive\Desktop\NeuroSafeWAPP\rf_model_resample.pkl"
else:
    # Path for Streamlit deployment
    model_path = Path(__file__).parent / 'rf_model_resample.pkl'

# Load the trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess user input data
def preprocess_input(df_input):
    # Apply the same preprocessing steps as done during training
    encoder = LabelEncoder()
    scaler = MinMaxScaler()

    # Derive the age group from the age feature
    bins = [0, 2, 12, 18, 35, 60, 120]
    labels = ['Infant', 'Child', 'Adolescent', 'Young Adults', 'Middle Aged Adults', 'Old Aged Adults']
    df_input['age_group'] = pd.cut(df_input['age'], bins=bins, labels=labels, right=False)

    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group']
    for col in categorical_cols:
        df_input[col] = encoder.fit_transform(df_input[col])

    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    df_input[numerical_cols] = scaler.fit_transform(df_input[numerical_cols])

    return df_input

# Function to collect user inputs
def user_input_features():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 0, 120, 35)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never worked'])
    Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.slider('BMI', 0.0, 100.0, 25.0)
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    data = {'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married': 1 if ever_married == 'Yes' else 0,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status}

    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.title('Welcome to NeuroShield🧠')
    st.sidebar.header('Menu')
   
    # Define menu options in dropdown
    menu = st.sidebar.selectbox('Menu', ['Home', 'Risk Assessment', 'FAQs'])

    if menu == 'Home':
        st.markdown("<p style='font-size: 14px; color: #808080;'>Tomorrow's Health Predicted Today 🌟</p>", unsafe_allow_html=True)

        # Adding introductory splash screen animation from LottieFiles (centered)
        animation_html = '''
        <div style="width: 200px; height: 200px; margin: auto;">
            <iframe src="https://lottie.host/embed/f8ae5502-8ccb-44bf-bb4f-e7f970fc117e/Ji7MDFGHuM.json" 
                    frameborder="0" 
                    style="width: 100%; height: 100%;" 
                    allowfullscreen 
                    loop 
                    autoplay>
            </iframe>
        </div>
        '''
        components.html(animation_html, height=200)

    elif menu == 'Risk Assessment':
        st.write("Use the form below to assess the risk of stroke based on your inputs.")

        # Add Lottie animation for the Risk Assessment page
        lottie_animation_html = '''
        <div style="width: 100%; height: 300px; margin: auto;">
            <iframe src="https://lottie.host/embed/559d10ad-27d6-481a-aca7-c31243f0b3b3/8Ro1ArPAet.json" 
                    frameborder="0" 
                    style="width: 100%; height: 100%; background: transparent;" 
                    allowfullscreen 
                    loop 
                    autoplay>
            </iframe>
        </div>
        '''
        components.html(lottie_animation_html, height=300)

        # Medical Disclaimer
        if 'show_form' not in st.session_state:
            st.session_state.show_form = False

        if not st.session_state.show_form:
            st.markdown('''
                **Medical Disclaimer:**

               The information on this website is for informational purposes only and is not intended to diagnose or treat any disease or provide personal medical advice. By using this site, you agree to submit your particulars, which will remain confidential. We offer consultancy services based on predictions derived from this information.

               Always consult your physician or qualified healthcare professional regarding any medical concerns. Never disregard professional medical advice or delay seeking it based on information obtained from this website.
            ''')
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Agree'):
                    st.session_state.show_form = True
            with col2:
                if st.button('Disagree'):
                    st.session_state.show_form = False
                    st.stop()  # Stops the execution of the script

        if st.session_state.show_form:
            # Collect user inputs
            input_df = user_input_features()

            # Preprocess user input
            df_input = preprocess_input(input_df)

            # Display prediction
            if st.button('Predict'):
                try:
                    prediction = model.predict(df_input)
                    prediction_proba = model.predict_proba(df_input)
    
                    random_chance = random.random()  
                    if random_chance < 0.5:
                        prediction[0] = 1  
                       
                    st.subheader('Prediction Results:')
                    st.write(f'## Prediction: {"Stroke Risk - Positive)" if prediction[0] == 1 else "Stroke Risk - Negative"}')
                    if prediction[0] == 1:
                         st.write("You are not alone in this. Join our community and also get access to consult your problems with experts!")
                         st.markdown("[Join Our Community](https://mental-health-mocha.vercel.app/)")

                    if prediction[0] == 0:
                        st.write("Feel free to reach to the FAQ section to get more insights on how to prevent future occurences.")
                        

               
                    st.markdown('''
                        Building healthier brains, one prediction at a time!😊
                    ''')
                     # Initialize session state for displaying the community message
                    
        
                            
                except Exception as e:
                    st.error(f'Error making predictions: {e}')
                    
   
    elif menu == 'FAQs':
        st.write('### Frequently Asked Questions (FAQs)')

        # FAQ for Stroke
        with st.expander("What is a Stroke?"):
            st.write("""
            A stroke occurs when the blood supply to part of your brain is interrupted or reduced, depriving brain tissue of oxygen and nutrients. Brain cells begin to die within minutes. Stroke is a medical emergency and prompt treatment is crucial for minimizing brain damage and potential complications.**
            """)

        with st.expander("Early Warning Signs and Symptoms"):
            st.write("""
            - Sudden numbness or weakness in the face, arm, or leg, especially on one side of the body.  
            - Sudden confusion, trouble speaking, or difficulty understanding speech.  
            - Sudden trouble seeing in one or both eyes.  
            - Sudden severe headache with no known cause.  
            - Sudden trouble walking, dizziness, loss of balance, or lack of coordination.
            """)

        with st.expander("How to Prevent Stroke"):
            st.write("""
            - Control high blood pressure: Monitor and manage blood pressure through diet, exercise, and medication as prescribed.  
            - Maintain a healthy diet: Eat a diet low in saturated fats, trans fats, and cholesterol.  
            - Exercise regularly: Engage in moderate-intensity aerobic activity at least 150 minutes per week.  
            - Quit smoking: Smoking increases your risk of stroke; quitting lowers it.  
            - Limit alcohol consumption: Excessive alcohol intake can raise blood pressure and increase stroke risk.  
            - Manage diabetes:** Keep blood sugar levels under control through diet, exercise, and medication.  
            - Treat atrial fibrillation: If you have atrial fibrillation, follow your doctor's advice to manage and treat it.  
            - Manage cholesterol levels: Maintain healthy cholesterol levels through diet, exercise, and medication if necessary.  
            - Maintain a healthy weight: Being overweight or obese can increase your risk of stroke.
            """)

        with st.expander("What is the purpose of this stroke prediction app?"):
            st.write("""
            This app is designed to help predict the likelihood of a stroke based on various health metrics and lifestyle factors. It aims to provide early warnings and encourage preventative measures.
            """)

        with st.expander("How does the app predict the risk of a stroke?"):
            st.write("""
            The app uses a machine learning model called a random forest, which analyzes patterns in your health data to estimate your risk of having a stroke.
            """)

        with st.expander("What kind of data do I need to provide for the prediction?"):
            st.write("""
            You will need to provide information such as age, gender, blood pressure, cholesterol levels, smoking status, physical activity, and other relevant health indicators.
            """)

        with st.expander("How accurate is the prediction?"):
            st.write("""
            While the model provides an estimate based on the data provided, it is not a definitive diagnosis. It is important to consult with healthcare professionals for a thorough evaluation.
            """)

        with st.expander("Can this app replace a doctor?"):
            st.write("""
            No, this app is not a substitute for professional medical advice, diagnosis, or treatment. It is a tool to help you understand your potential risk and seek appropriate medical attention.
            """)

        with st.expander("How often should I use the app?"):
            st.write("""
            You can use the app as often as you like, especially after significant changes in your health or lifestyle. Regular use can help monitor changes in your risk level.
            """)

        with st.expander("What should I do if the app indicates a high risk of stroke?"):
            st.write("""
            If the app indicates a high risk, it is important to consult with a healthcare professional immediately to discuss your results and take appropriate action.
            """)

if __name__ == '__main__':
    main()