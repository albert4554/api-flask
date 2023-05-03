import requests

paramsVersicolor = {'sepal_length':5,'sepal_width':2.1,'petal_length':4.5,'petal_width':1.4}
paramsSetosa = {'sepal_length':5.1,'sepal_width':3.5,'petal_length':1.4,'petal_width':0.2}
paramsVirginica = {'sepal_length':6.3,'sepal_width':3.3,'petal_length':6.0,'petal_width':2.5}

response = requests.get('http://127.0.0.1:5000/predict', params=paramsVirginica)

if response.status_code == 200:
    print(response.text)

#este es mi nuevo cambio