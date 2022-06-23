import pickle
import numpy as np
from flask import Flask,request,app,jsonify, url_for, render_template

app=Flask(__name__)
# Load pickle file
model = pickle.load(open('D:\\Subbu\\Learnings\\Data Science\\ML\\Practical\\Airfoil self noise\\model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():    # Function to predict the values
    data = request.json['data']    # data will be given in POSTMAN while calling the API. Any API calling the method should pass data in JSON format and with key as 'data'.
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict', methods = ['POST'])
def predict():    # Function to predict the values
    data = [float(x) for x in request.form.values()]    # request.form.values will retrieve values from the web form we created
    final_features = [np.array(data)]
    print(data)
    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text = 'Airfoil pressure is {}'.format(output))

if __name__ == '__main__':
    app.run(debug = True)

