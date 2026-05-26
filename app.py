import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("Life Expectancy Prediction App")

st.write("Enter health and economic details below.")

# User Inputs
status = st.selectbox(
    "Country Status",
    ["Developed", "Developing"]
)

status_developing = 1 if status == "Developing" else 0
year = st.number_input("Year", value=2015)
adult_mortality = st.number_input("Adult Mortality")
infant_deaths = st.number_input("Infant Deaths")
alcohol = st.number_input("Alcohol")
percentage_expenditure = st.number_input("Percentage Expenditure")
hepatitis_b = st.number_input("Hepatitis B")
measles = st.number_input("Measles")
bmi = st.number_input("BMI")
under_five_deaths = st.number_input("Under-Five Deaths")
polio = st.number_input("Polio")
total_expenditure = st.number_input("Total Expenditure")
diphtheria = st.number_input("Diphtheria")
hiv_aids = st.number_input("HIV/AIDS")
gdp = st.number_input("GDP")
population = st.number_input("Population")
thinness_1_19 = st.number_input("Thinness 1-19 years")
thinness_5_9 = st.number_input("Thinness 5-9 years")
income_composition = st.number_input("Income Composition of Resources")
schooling = st.number_input("Schooling")

# Create input dataframe
# Create transformed features
GDP_log = np.log1p(gdp)
Population_log = np.log1p(population)
Measles_log = np.log1p(measles)
HIV_AIDS_log = np.log1p(hiv_aids)
Adult_Mortality_log = np.log1p(adult_mortality)
infant_deaths_log = np.log1p(infant_deaths)
under_five_deaths_log = np.log1p(under_five_deaths)
percentage_expenditure_log = np.log1p(percentage_expenditure)

# Create dataframe
input_data = pd.DataFrame({

    'Year': [year],

    'Adult Mortality': [adult_mortality],

    'infant deaths': [infant_deaths],

    'Alcohol': [alcohol],

    'percentage expenditure': [percentage_expenditure],

    'Hepatitis B': [hepatitis_b],

    'Measles': [measles],

    'BMI': [bmi],

    'under-five deaths': [under_five_deaths],

    'Polio': [polio],

    'Total expenditure': [total_expenditure],

    'Diphtheria': [diphtheria],

    'HIV/AIDS': [hiv_aids],

    'GDP': [gdp],

    'Population': [population],

    'thinness  1-19 years': [thinness_1_19],

    'thinness 5-9 years': [thinness_5_9],

    'Income composition of resources': [income_composition],

    'Schooling': [schooling],

    'GDP_log': [GDP_log],

    'Population_log': [Population_log],

    'infant_deaths_log': [infant_deaths_log],

    'under_five_deaths_log': [under_five_deaths_log],

    'Measles_log': [Measles_log],

    'HIV_AIDS_log': [HIV_AIDS_log],

    'percentage_expenditure_log': [percentage_expenditure_log],

    'Adult_Mortality_log': [Adult_Mortality_log],

    'Status_Developing': [status_developing]
})

feature_order = [
    'Year',
    'Adult Mortality',
    'infant deaths',
    'Alcohol',
    'percentage expenditure',
    'Hepatitis B',
    'Measles',
    'BMI',
    'under-five deaths',
    'Polio',
    'Total expenditure',
    'Diphtheria',
    'HIV/AIDS',
    'GDP',
    'Population',
    'thinness  1-19 years',
    'thinness 5-9 years',
    'Income composition of resources',
    'Schooling',
    'GDP_log',
    'Population_log',
    'infant_deaths_log',
    'under_five_deaths_log',
    'Measles_log',
    'HIV_AIDS_log',
    'percentage_expenditure_log',
    'Adult_Mortality_log',
    'Status_Developing'
]

input_data = input_data[feature_order]

# Prediction button
if st.button("Predict Life Expectancy"):

    # Columns used during scaler fitting
    scale_columns = [
        'Year',
        'Adult Mortality',
        'infant deaths',
        'Alcohol',
        'percentage expenditure',
        'Hepatitis B',
        'Measles',
        'BMI',
        'under-five deaths',
        'Polio',
        'Total expenditure',
        'Diphtheria',
        'HIV/AIDS',
        'GDP',
        'Population',
        'thinness  1-19 years',
        'thinness 5-9 years',
        'Income composition of resources',
        'Schooling',
        'GDP_log',
        'Population_log',
        'infant_deaths_log',
        'under_five_deaths_log',
        'Measles_log',
        'HIV_AIDS_log',
        'percentage_expenditure_log',
        'Adult_Mortality_log'
    ]

    # Scale only scaler columns
    scaled_part = scaler.transform(input_data[scale_columns])

    # Convert back to dataframe
    scaled_df = pd.DataFrame(
        scaled_part,
        columns=scale_columns
    )

    # Add Status_Developing separately
    scaled_df['Status_Developing'] = status_developing

    # Reorder columns exactly
    final_order = [
        'Year',
        'Adult Mortality',
        'infant deaths',
        'Alcohol',
        'percentage expenditure',
        'Hepatitis B',
        'Measles',
        'BMI',
        'under-five deaths',
        'Polio',
        'Total expenditure',
        'Diphtheria',
        'HIV/AIDS',
        'GDP',
        'Population',
        'thinness  1-19 years',
        'thinness 5-9 years',
        'Income composition of resources',
        'Schooling',
        'GDP_log',
        'Population_log',
        'infant_deaths_log',
        'under_five_deaths_log',
        'Measles_log',
        'HIV_AIDS_log',
        'percentage_expenditure_log',
        'Adult_Mortality_log',
        'Status_Developing'
    ]

    scaled_df = scaled_df[final_order]

    # Prediction
    prediction = model.predict(scaled_df)

    st.success(
        f"Predicted Life Expectancy: {prediction[0]:.2f} years"
    )
