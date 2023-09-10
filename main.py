from flask import Flask, render_template, request

import numpy as np

import pickle


app = Flask(__name__)

model = pickle.load(open('lnr_model.pkl', 'rb'))


@app.route('/')
def start():
    return render_template('model.html')


@app.route('/predict', methods=['POST'])
def predict():
     all_features = [int(x) for x in request.form.values()]
     features = [np.array(all_features)]
     predicted = model.predict(features)

     if predicted == 1:
         predicted = 'Dead'

     else:
         predicted='Alive'

     return render_template('result.html', prediction='You will {}'.format(predicted))


if __name__ == '__main__':
    app.run(debug=True)