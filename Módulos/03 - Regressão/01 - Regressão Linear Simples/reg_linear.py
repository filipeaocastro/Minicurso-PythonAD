"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Regressão Linear

"""

# Importando as bibliotecas
import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets


# Importando o dataset
dataset = pd.read_csv('salarios.csv')
X = dataset.iloc[:, :-1].values    #Cria uma matriz com os valores X (independente)
y = dataset.iloc[:, 1].values      #Cria uma matriz com os valores y (dependente)

plt.figure(1)
plt.plot(X, y)
plt.title('Salário vs Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

#Dividindo o dataset em Training set e Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#20 -> train e  10 -> teste


#REGRESSÃO LINEAR SIMPLES
#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)     # Modela o regressor de acordo com os traning sets

# Plotando os resultados gerais
y_pred = regressor.predict(X)   # Prediz o valor de y para cada valor de X

plt.figure(4)
plt.plot(X, y, color = 'blue')
plt.plot(X, y_pred, 'black')
plt.title('Salário vs Experiência (Training Set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

y_pred_6 = regressor.predict(6.5) # Prediz o valor de y para x = 6.5

plt.figure(5)
plt.plot(X, y, color = 'blue')
plt.plot(X, y_pred, color = 'black')
plt.scatter(6.5, y_pred_6, color = 'red')
plt.title('Salário vs Experiência (Training Set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()


# ********************************************** #


#Predição do y_train e y_test
y_pred_train = regressor.predict(X_train) 
y_pred_test = regressor.predict(X_test) 

#Plotando os resultados TRAINING
plt.figure(2)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_pred_train, color = 'blue')
plt.title('Salário vs Experiência (Training Set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

#Plotando os resultados TESTING
plt.figure(3)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred_test, color = 'blue')
plt.title('Salário vs Experiência (Test Set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()


