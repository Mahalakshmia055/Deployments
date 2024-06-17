from flask import Flask, render_template, request
import pickle
from pyngrok import ngrok

port_no = 5000

ngrok.set_auth_token('2hzXLuGcPIUV8bQxESqx8QTfgTo_5EUuwBvWemuqppGabY8Rx')
public_url = ngrok.connect(port_no).public_url

app = Flask(__name__)

def predict_iris(sl,sw,pl,pw):

    predss = [[sl,sw,pl,pw]]

    with open('iris.pkl', 'rb') as our_model:
        for_pred_model = pickle.load(our_model)

    predicted = for_pred_model.predict(predss)

    return predicted

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    print(public_url)
    app.run()


