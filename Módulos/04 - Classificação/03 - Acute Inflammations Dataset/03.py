"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Classificação do Acute Inflammations Dataset

"""

# Importando as bibliotecas
import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets

# Importando o dataset
dataset = pd.read_csv('acute_inflammations.csv', sep = ';')
X = dataset.iloc[:, :-1].values     #Cria uma matriz com os valores X (independente)
y = dataset.iloc[:, -1].values      #Cria uma matriz com os valores y (dependente)

# Transformando dados categóricos
from sklearn.preprocessing import LabelEncoder
# Transformando a variável independente
labelencoder_X = LabelEncoder()
for i in range(1, 6):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])


df_X = pd.DataFrame(X)

#Dividindo o dataset em Training set e Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ****************  CLASSIFICADORES  **************** #

# Regressão Logística
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Support Vector Machine (SVM)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# *************************************************** #

# Predição do test set
y_pred = classifier.predict(X_test)
dfy = pd.DataFrame(y_pred)

# Fazendo a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)