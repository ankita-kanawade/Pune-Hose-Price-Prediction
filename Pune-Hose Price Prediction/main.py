import pandas as pd
import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    total_bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    print(location, total_bath, sqft)
    input_data = pd.DataFrame([[location, sqft, total_bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0] * 1e5
    return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True, port=8000)
