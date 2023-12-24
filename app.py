import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
# import xgboost


def preprosessor(frame):
    for i in frame.columns:
        q3 = np.percentile(frame[i], 75)
        q1 = np.percentile(frame[i], 25)

        iqr = q3 - q1

        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        frame[i] = np.where(frame[i] > upper, upper, np.where(frame[i] < lower, lower, frame[i]))

    frame["exhaust_vacuum"] = np.log1p(frame['exhaust_vacuum'])
    frame['r_humidity'] = np.square(frame['r_humidity'])

    return frame


with open("model_pipe.pkl", 'rb') as file:
    pipe = pkl.load(file)



st.write(
            f"<div style='text-align: center;'><h1 style='font-weight: bold;color: seagreen;font-size:50px;'> "
            f"Power Generation Predictor ",

            unsafe_allow_html=True,
        )


temperature = st.number_input("Enter Temperature (Degree Celcius)")

exhaust_vacuum = st.number_input("Enter Exhaust vaccum (cm Hg)")

amb_pressure = st.number_input("Enter Ambient Pressure (millibar)")

r_humidity = st.number_input("Enter Relative humdiity (percentage)")




st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
if st.button('Predict Energy Production'):



    input_df = pd.DataFrame({'temperature': [temperature],'exhaust_vacuum':[exhaust_vacuum] ,
                             'amb_pressure':[amb_pressure],'r_humidity': [r_humidity]})


    results = pipe.predict(input_df)



    st.write(
            f"<div style='text-align: center;'><h2 style='font-weight: bold;color: seagreen;font-size:30px;'> "
            f"The net Hourly Electrical Energy Produced will be between ",

            unsafe_allow_html=True,
        )

    st.write("<div style='text-align: center;'><h2 style='font-weight: bold;color: seagreen;font-size:30px;'> "
             f"{np.round(results[0] - 15, 2)}  - {np.round(results[0] + 15, 2)} </h2></div>",
             unsafe_allow_html=True,
             )
st.divider()
st.write("The working of Combined Cycle Power Plant")
st.video("https://youtu.be/KVjtFXWe9Eo?si=nRrOcDncgmZsNzRx")
st.caption("Credits - https://www.ge.com/gas-power/resources/education/combined-cycle-power-plants")
