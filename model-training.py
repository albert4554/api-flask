from sklearn.svm import SVC #Importamos el clasificador svc 

from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris #conjunto de datos iris
import joblib

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
#Modelo de prediccion

clf = SVC() #Instancia del clasificador

clf.fit(x_train, y_train) #entrenamos al modelo


#print(clf.score(x_test, y_test))
#joblib  nos permitira guardar el modelo entrenado en un archivo de texto.
#De esta manera nos ahorramos tiempo y recursos en estar creando y entrenando modelo cada minuto.
joblib.dump(clf,"iris-svc-model.pkl")