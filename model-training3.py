from cgi import test
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import joblib
#Extraccion de datos
#leemos el csv
datos = pd.read_csv("salida2.csv")
dataframe=pd.DataFrame(datos)
print(datos)
X=(dataframe[["SEXO_femenino","SEXO_masculino","PROCEDENCIA_urbano","ECONV_Viven juntos","APF_SI","ACCINT_SI","CPART_SI","ACTEXTRA_SI","PES_SI","TFAMILIA","EMADRE","EPADRE","EDAD","TVIAJE","TESTUDIO","TLIBRE","FALTAS","N1","N2"]])
y=(dataframe["APROBADO"])
#Entrenamiento
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0);
model=LogisticRegression();
model.fit(X_train,y_train);
#datanew={'abierta':[1,0],'moroso':[1,0],'trabajo':[0,1]}
#features=[[0,1,1]]
#clientesnew=pd.DataFrame(datanew,columns=['abierta','moroso','trabajo'])
#prediccion=model.predict(features);
#print(clientesnew);
#print(prediccion);
#print(model.score(X_test, y_test))
joblib.dump(model,"predicion-lr-model.pkl")