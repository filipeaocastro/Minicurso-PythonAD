"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Regressão Polinomial

"""


# Importando as bibliotecas
import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets

#Importing the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #Colocou [1:2] pro sistema considerar X como uma matriz, não vetor
y = dataset.iloc[:, 2].values

plt.figure(1)
plt.plot(X,y, color = 'black')
plt.xlabel('Nível de posição')
plt.ylabel('Salário')
plt.show()


#Fazendo regressão Simples
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

#Predição com Linear Regression
y_pred_6_lin = lin_reg.predict(6.5)

#Plotando a Regressão Linear
plt.figure(2)
plt.plot(X, y, color = 'black', label = 'Dados reais')
plt.plot(X, y_pred_lin, color = 'blue', label = 'Regressão Linear')
plt.title('Regresão Linear')
plt.xlabel('Nível de posição')
plt.ylabel('Salário')
#plt.scatter(6.5, y_pred_6_lin, color = 'red', label = 'Predição p/ 6.5')
plt.legend(loc='upper left')
plt.show()

#Fazendo Regressão Polinomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2 = lin_reg_2.fit(X_poly, y)
y_pred = lin_reg_2.predict(X_poly)

y_pred_6_poly = lin_reg_2.predict(poly_reg.fit_transform(6.5))


#Plotando a Regressão Polinomial
plt.scatter(X, y, color = 'black', label = 'Dados reais')
plt.plot(X, y_pred, color = 'blue', label = 'Regressão Polinomial')
#plt.scatter(6.5, y_pred_6_poly, color = 'red', label = 'Regressão p/ 6.5')
plt.title('Regressão Polinomial')
plt.xlabel('Nível de posição')
plt.ylabel('Salário')
plt.legend(loc='upper left')
plt.show()



# ************************ #

# Plotando gráfico suavizado

# Cria um vetor com o mesmo intervalo X com, só que com o step de 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) # Transforma o vetor em uma matriz de uma coluna

plt.scatter(X, y, color = 'black', label = 'Dados reais')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue', 
         label = 'Regressão Polinomial')
plt.scatter(6.5, y_pred_6_poly, color = 'red', label = 'Regressão p/ 6.5')
plt.title('Regressão Polinomial')
plt.xlabel('Nível de posição')
plt.ylabel('Salário')
plt.legend(loc='upper left')
plt.show()






















