"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Pré Processamento de Dados

"""

# Importando as bibliotecas
import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets

# Importando o dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    #Cria uma matriz com os valores X (independente)
y = dataset.iloc[:, 3].values      #Cria uma matriz com os valores y (dependente)

df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y)

# Trabalhando com dados ausentes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])   # Insere os dados ausentes na matriz
df_X = pd.DataFrame(X)


#Dividindo o dataset em Training set e Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
#Os algorítmos de ML são baseados na distancia euclidiana, portanto, é necessário colocar as 
#colunas na mesma escala
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)