from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your model (you should already have trained and saved it)
model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    return render_template('template.html', prediction_text='Machine Failure Prediction: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
