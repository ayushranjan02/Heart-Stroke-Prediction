from flask import Flask, render_template, request
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 
import numpy as np

model = tf.keras.models.load_model('my_model.h5')

app = Flask(__name__)

# def predictor(X_new):
#     X_new = X_new.copy()
#     for i in range(len(X_new)):
#         X_new[i] = (X_new[i] - min(X_new)) / (max(X_new) - min(X_new)) 
#     X_new = np.array(X_new)
#     preds = model.predict(X_new.reshape(1, -1)) * 100
#     return preds[0][0]
def predictor(X_new):
    for i in range(len(X_new)):
        X_new[i] = (X_new[i] - min(X_new)) / (max(X_new) - min(X_new)) 
    X_new = np.array(X_new)
    preds = model.predict(X_new.reshape(1, -1)) * 100
    return preds[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    Age = request.form.get('Age')
    Sex = request.form.get('Sex')
    Chest_pain_type = request.form.get('CP')
    Resting_blood_pressure = request.form.get('RBP')
    Serum_cholestoral = request.form.get('SC')
    blood_sugar = request.form.get('FBS')
    resting_ECG_results = request.form.get('ECG')
    Maximunm_heart_rate = request.form.get('MHRA')
    EIA = request.form.get('EIA')
    ST_depression = request.form.get('ST')
    slope = request.form.get('Slope')
    number_of_major_vessels = request.form.get('MajorVessel')
    thalium = request.form.get('Thal')

    import sys
    # prediction
    data = [Age, Sex, Chest_pain_type, Resting_blood_pressure, Serum_cholestoral, blood_sugar, resting_ECG_results, Maximunm_heart_rate, EIA, float(ST_depression), slope, number_of_major_vessels, thalium]
    # data = [40, 1, 0, 152, 223, 0, 1, 181, 0, 0.0, 2, 0, 3]
    print(data, file=sys.stderr)
    data = [int(x) for x in data]
    print(data, file=sys.stderr)
    ans = predictor(data)
    print(ans, file=sys.stderr)

    ans = float(0 if ans is None else ans)

    message = 'You have the chances of getting a heart stroke!' if ans > 50 else 'You have less chances of getting a heart stroke!'
    return render_template('index.html', result=f'{ans:.2f}%', message=message)



if __name__=='__main__':
    app.run(debug=True)  