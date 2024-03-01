from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle as pkl


app = Flask(__name__)

app.static_folder = 'static'

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

with open("./artifacts/model.pkl", 'rb') as file:
    pipe = pkl.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        exhaust_vacuum = float(request.form['exhaust_vacuum'])
        amb_pressure = float(request.form['amb_pressure'])
        r_humidity = float(request.form['r_humidity'])


        input_data = pd.DataFrame({'temperature': [temperature],'exhaust_vacuum':[exhaust_vacuum] ,
                             'amb_pressure':[amb_pressure],'r_humidity': [r_humidity]})


        prediction = pipe.predict(input_data)[0]

        return render_template('index.html', prediction=f'The Excpected  Power Gnerated Would be '
                                                        f'between <br> {np.round(prediction-15,1)}MW - {np.round(prediction+ 15,1)}MW')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)