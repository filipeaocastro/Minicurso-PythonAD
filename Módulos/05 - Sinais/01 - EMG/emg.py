"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Classificação de sinal Eletromiográfico

"""


# Importando as bibliotecas
import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets

# Importando o dataset
dataset = pd.read_csv('emg_all_features_labeled.csv', header = None)
X = dataset.iloc[:, :-1].values    #Cria uma matriz com os valores X (independente)
y = dataset.iloc[:, -1].values      #Cria uma matriz com os valores y (dependente)

t = np.arange(0, len(X[:, 0]))


#Dividindo o dataset em Training set e Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Support Vector Machine (SVM) = 
from sklearn.svm import SVC
classifier = SVC(random_state = 0)
classifier = classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)