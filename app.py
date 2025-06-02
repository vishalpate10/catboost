from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the CatBoost model
with open('catboost.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return '''
        <h2>CatBoost Prediction</h2>
        <form action="/predict" method="post">
            Feature 1: <input type="text" name="f1"><br>
            Feature 2: <input type="text" name="f2"><br>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        features = np.array([[f1, f2]])
        prediction = model.predict(features)
        return f'<h3>Prediction: {prediction[0]}</h3>'
    except Exception as e:
        return f'<h3>Error: {str(e)}</h3>'

if __name__ == '__main__':
    app.run(debug=True)
