#Importamos los 3 metodos que utilizaremos
from flask import Flask, request, jsonify
import pandas as pd
#Importamos joblib para leer el modelo
import joblib
from flask_cors import CORS, cross_origin
#Creamos la instancia de flask
app = Flask(__name__)
CORS(app)
#Abrimos el archivo qe contienen el modelo
MODEL = joblib.load('iris-svc-model.pkl')
MODEL2=joblib.load('banquero-lr-model.pkl')
MODEL3=joblib.load('predicion-lr-model.pkl')
#Las etiquetas con las cuales se clasificaran nuevos datos
#Sabemos que son los nombres de los 3 tipos de iris.
MODEL_LABELS = ['setosa', 'versicolor', 'virginica']
MODEL_LABELS2=['desaprobara','aprobara']
#El metodo predict sera el encargado de clasificar y dar una respuesta a cualquier IP que le envie una peticion
@app.route('/predict')
def predict():
    #Declaramos cuales seran los parametros que recibe la peticion 
    #En este caso son las medidas de la flor a clasificar
    #longitud y tamaño de petalo y del sepalo
    sepal_length = request.args.get('sepal_length')
    sepal_width = request.args.get('sepal_width')
    petal_length = request.args.get('petal_length')
    petal_width = request.args.get('petal_width')
    #la lista de caracteristicas que se utilizaran  
    #para la prediccion
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    #print(features)
    #Utilizamos el model para la prediccion de los datos
    label_index = MODEL.predict(features)
    print(label_index)
    #Variable label contendra el resultado de la clasificacion
    label = MODEL_LABELS[label_index[0]]
    print(label)
    #Creamos y enviamos la respuesta al cliente
    return jsonify(status='clasificado completado', clasificacion=label)
@cross_origin
@app.route('/apiPrediccion',methods=['POST'])
def predict2():
    #datanew={'abierta':[1,0],'moroso':[1,0],'trabajo':[0,1]}
    #features=[[0,1,1]]
    cla_details=[]
    cla_details=request.get_json()
    arr2=[]
    for cla in cla_details:
        print('Cla',cla)
        abierta=cla['abierta']
        moroso=cla['moroso']
        trabajo=cla['trabajo']
        arr2.append([abierta,moroso,trabajo])
    print('Este es el array 2',arr2)
    #abierta=cla_details['abierta']
    #moroso=cla_details['moroso']
    #trabajo=cla_details['trabajo']
    #features2=[[abierta,moroso,trabajo]]
    clientesnew=pd.DataFrame(arr2,columns=['abierta','moroso','trabajo'])
    prediccion=MODEL2.predict(clientesnew);
    print('Data frame de prediccion')
    print(clientesnew)
    print('Prediccion')
    print(prediccion)
    arrClasificados=[]
    for n in prediccion:
        print('Este es el',n)
        arrClasificados.append(MODEL_LABELS2[n])   
    return jsonify(status='Clasificacion completada', clasificacion=arrClasificados)
#Api de prediccion escolar
@cross_origin
@app.route('/apiPrediccion3',methods=['POST'])
def predict3():
  
    cla_details=[]
    cla_details=request.get_json()
    arr2=[]
    for cla in cla_details:
        print('Cla',cla)
        SEXO_femenino=cla['SEXO_femenino']
        SEXO_masculino=cla['SEXO_masculino']
        PROCEDENCIA_urbano=cla['PROCEDENCIA_urbano']
        ECONV_Viven_juntos=cla['ECONV_Viven_juntos']
        APF_SI=cla['APF_SI']
        ACCINT_SI=cla['ACCINT_SI']
        CPART_SI=cla['CPART_SI']
        ACTEXTRA_SI=cla['ACTEXTRA_SI']
        PES_SI=cla['PES_SI']
        TFAMILIA=cla['TFAMILIA']
        EMADRE=cla['EMADRE']
        EPADRE=cla['EPADRE']
        EDAD=cla['EDAD']
        TVIAJE=cla['TVIAJE']
        TESTUDIO=cla['TESTUDIO']
        TLIBRE=cla['TLIBRE']
        FALTAS=cla['FALTAS']
        N1=cla['N1']
        N2=cla['N2']

        arr2.append([SEXO_femenino,SEXO_masculino,PROCEDENCIA_urbano,ECONV_Viven_juntos,APF_SI,ACCINT_SI,CPART_SI,ACTEXTRA_SI,PES_SI,TFAMILIA,EMADRE,EPADRE,EDAD,TVIAJE,TESTUDIO,TLIBRE,FALTAS,N1,N2])
    print('Este es el array 2',arr2)
    clientesnew=pd.DataFrame(arr2,columns=['SEXO_femenino','SEXO_masculino','PROCEDENCIA_urbano','ECONV_Viven_juntos','APF_SI','ACCINT_SI','CPART_SI','ACTEXTRA_SI','PES_SI','TFAMILIA','EMADRE','EPADRE','EDAD','TVIAJE','TESTUDIO','TLIBRE','FALTAS','N1','N2'])
    prediccion=MODEL3.predict(clientesnew);
    print('Data frame de prediccion')
    print(clientesnew)
    print('Prediccion')
    print(prediccion)
    arrClasificados=[]
    for n in prediccion:
        print('Este es el',n)
        arrClasificados.append(MODEL_LABELS2[n])   
    return jsonify(status='Clasificacion completada', clasificacion=arrClasificados) 
##########
@cross_origin
@app.route('/recepciona',methods=['POST'])
def insert_game():
    game_details=request.get_json();
    sepal_length = game_details["sepal_length"]
    sepal_width = game_details["sepal_width"]
    petal_length = game_details["petal_length"]
    petal_width= game_details['petal_width']
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    print(features)
    label_index = MODEL.predict(features)
    print (label_index)
    label = MODEL_LABELS[label_index[0]]
    print (label)
    return jsonify(status='clasificado completado', clasificacion=label)

@cross_origin
@app.route('/api/v1/users')
def get_users():
    response = {'message': 'ellanoteama'}
    return jsonify(response)

app.route('/api/v1/users/<id>', methods=['GET'])
def get_user(id):
    response = {'message': 'success'}
    return jsonify(response)

@app.route('/api/v1/users/', methods=['POST'])
def create_user():
    response = {'message': 'success'}
    return jsonify(response)

@app.route('/api/v1/users/<id>', methods=['PUT'])
def update_user(id):
    response = {'message': 'success'}
    return jsonify(response)

@app.route('/api/v1/users/<id>', methods=['DELETE'])
def delete_user(id):
    response = {'message': 'success'}
    return jsonify(response)
  
if __name__ == '__main__':
    #Iniciamo el servidor

    app.run(debug=True)