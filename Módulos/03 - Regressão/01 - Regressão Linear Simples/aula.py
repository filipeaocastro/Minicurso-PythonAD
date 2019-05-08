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
plt.plot(X, y, linewidth = 5)
plt.title('Salário vs Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')

#Dividindo o dataset em Training set e Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                        test_size = 0.2, random_state = 0)

# Regressão Linear Simples
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.figure(2)
plt.scatter(X_test, y_test, color = 'blue', linewidth = 5)
plt.plot(X_test, y_pred, color = 'black', linewidth = 5)
plt.title('Salário vs Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')

y_pred = regressor.predict(X)
y_pred_65 = regressor.predict([[6.5]])
plt.figure(3)
plt.scatter(X, y, color = 'blue', linewidth = 5)
plt.scatter(6.5, y_pred_65, color = 'red', linewidth = 7)
plt.plot(X, y_pred, color = 'black', linewidth = 5)
plt.title('Salário vs Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')














