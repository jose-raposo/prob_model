import numpy as np
import pandas
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import load
from flask import Flask, jsonify, request, render_template

model = load('final_model.sav')

app = Flask(__name__)

@app.route('/')
def index():
	return 'Predição ML'

@app.route('/', methods = ['POST'])
def predictions():
	# Aqui entram as variáveis que serão recebidas do web app
	input1 = request.form['query1'] # query1 é a primeira variável
	input2 = request.form['query2'] # query1 é a primeira variável
	input3 = request.form['query3'] # query1 é a primeira variável
	input4 = request.form['query4'] # query1 é a primeira variável
	input5 = request.form['query5'] # query1 é a primeira variável
	input6 = request.form['query6'] # query1 é a primeira variável
	input7 = request.form['query7'] # query1 é a primeira variável
	input8 = request.form['query8'] # query1 é a primeira variável
	input9 = request.form['query9'] # query1 é a primeira variável

	variaveis = [input1, input2, input3,
	input4, input5, input6, input7, input8, input9]

	variaveis = np.array([variaveis])

	# predição de probabilidade positiva do modelo
	pred = model.predict_proba(variaveis)[:,1]
	res = int(pred*100)
	return jsonify(prediction = res)

if __name__ == '__main__':
	app.run(debug = True, host='0.0.0.0')